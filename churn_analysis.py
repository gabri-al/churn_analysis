import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn import metrics
from scipy.stats import boxcox
from scipy.stats import multivariate_normal

plt.style.use('bmh')
%matplotlib inline
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.colheader_justify', 'left')
pd.set_option('display.max_colwidth', 150)
pd.set_option('display.float_Format', '{:,.2f}'.format)
np.set_printoptions(suppress=True)

########################################################################################################################################
# DATA UPLOAD
########################################################################################################################################
col_list = ['discount_ratio','mrr_ratio','mrr_ratio_A','mrr_ratio_B','support_chats','subs_A','subs_B']
col_list_full = ['obs_date', 'is_churn'] + col_list + ['has_addon']

_path = r'/Users/yourname/Churn_simulated_data.xlsx'
churn_simulated_metrics = pd.read_excel(_path, sheet_name = 'Sheet1')
churn_simulated_metrics = churn_simulated_metrics.astype({'is_churn':'int',
                                                          'discount_ratio':'float',
                                                          'mrr_ratio':'float',
                                                          'mrr_ratio_A':'float','mrr_ratio_B':'float',
                                                          'support_chats':'float','subs_A':'float',
                                                          'subs_B':'float',
                                                          'has_addon':'int'})
print(churn_simulated_metrics.head(-2))


########################################################################################################################################
# COHORT ANALYSIS
########################################################################################################################################

def cohort_plot(sourcedf, xmetric_name, ymetric_name, ncohort = 10, min_threshold = 5):
    '''Cohorts are formed against xmetric and then means of each cohort is plotted for x and y metrics'''
    if len(sourcedf[xmetric_name].unique()) < min_threshold:
        cohort_xmeans = list(sourcedf[xmetric_name].unique())
        cohort_ymeans = sourcedf.groupby(xmetric_name)[ymetric_name].mean()
        df_to_plot = pd.DataFrame({xmetric_name: cohort_xmeans, ymetric_name: cohort_ymeans.values})
    else:
        groups_ = pd.qcut(sourcedf[xmetric_name], q=ncohort, duplicates = 'drop')
        cohort_xmeans = sourcedf.groupby(groups_)[xmetric_name].mean()
        cohort_ymeans = sourcedf.groupby(groups_)[ymetric_name].mean()
        df_to_plot = pd.DataFrame({xmetric_name: cohort_xmeans.values, ymetric_name: cohort_ymeans.values})
    
    plt.figure(figsize = (6,4))
    plt.plot(xmetric_name, ymetric_name, data = df_to_plot, marker = 'o', linewidth = 2)
    plt.xlabel('Cohort average of '+xmetric_name)
    plt.ylabel('Cohort average of '+ymetric_name)
    plt.grid(visible=True)
    plt.title("Relationship between "+xmetric_name+" and "+ymetric_name)
    plt.show()
    
exclude_ = ['datetime64[ns]'] # data type to exclude from the analysis
cols_ = []
for i in range(0, len(churn_simulated_metrics.dtypes), 1):
    if (str(churn_simulated_metrics.dtypes[i]) not in exclude_) & (churn_simulated_metrics.columns[i] != 'is_churn'):
        cols_.append(churn_simulated_metrics.columns[i])

for m in cols_:
    cohort_plot(churn_simulated_metrics, m, 'is_churn', 10, 5)

########################################################################################################################################
# DATA SPLIT
########################################################################################################################################
churn_simulated_metrics['obs_date'] = pd.to_datetime(churn_simulated_metrics['obs_date'], format = '%Y-%m-%d')
churn_simulated_metrics.sort_values(by = 'obs_date', ascending = True, inplace = True)
churn_simulated_metrics.reset_index(inplace = True, drop = True)

# Prepare features and target labels
if 'has_addon' not in col_list:
    col_list = col_list+['has_addon']
X = np.array(churn_simulated_metrics.loc[:, col_list])
y = np.array(churn_simulated_metrics.loc[:, 'is_churn'])

# Time series split
tscv = TimeSeriesSplit(n_splits = 3)


########################################################################################################################################
# LOGISTIC REGRESSION
########################################################################################################################################
# Hyperparam selection
base_model = LogisticRegression(solver = 'liblinear')
test_par = {'C': [.005, .01, .04, .08, .16, .32, .64, .75, .95],
            'penalty' : ['l1','l2']          
           }
LR_gsearch = GridSearchCV(base_model, param_grid = test_par, scoring = 'roc_auc', cv = tscv, verbose = 1, n_jobs = -1)
LR_gsearch.fit(X,y)
LR_result_df = pd.DataFrame(LR_gsearch.cv_results_)

print(LR_gsearch.best_score_)
print(LR_gsearch.best_params_)

########################################################################################################################################
# RANDOM FOREST
########################################################################################################################################
# Hyperparam selection
base_model = RandomForestClassifier(n_jobs = -1)
test_par = {'max_depth' : [2,5,10],
            'max_features': ['sqrt', 'log2'],
            'n_estimators': [5,10,100,500,1000]}
RF_gsearch = GridSearchCV(base_model, param_grid = test_par, scoring = 'roc_auc', cv=tscv, verbose=1, n_jobs=-1, refit=True)
RF_gsearch.fit(X,y)
RF_result_df = pd.DataFrame(RF_gsearch.cv_results_)

print(RF_gsearch.best_score_)
print(RF_gsearch.best_params_)

# Check feature importance
X_feature = col_list
RF_feat = pd.DataFrame(data = RF_gsearch.best_estimator_.feature_importances_, index = X_feature, columns=['importance'])
RF_feat.sort_values('importance').plot(kind='barh', title = 'RF Feature Importance')
plt.show()

########################################################################################################################################
# XGBOOST
########################################################################################################################################
# Hyperparam selection
base_model = xgb.XGBClassifier(objective='binary:logistic', n_jobs = -1, eval_metric = 'auc')
test_par = {'max_depth' : [2,10,20,25,40],
            'learning_rate': [.1,.2,.5],
            'n_estimators': [5,10,20,30,50],
            'min_child_weight': [.05,.15,.5,1]}
XGB_gsearch = GridSearchCV(base_model, param_grid = test_par, scoring = 'roc_auc', cv=tscv, verbose=1, n_jobs=-1, refit=True)
XGB_gsearch.fit(X,y)
XGB_result_df = pd.DataFrame(XGB_gsearch.cv_results_)

print(XGB_gsearch.best_score_)
print(XGB_gsearch.best_params_)

# Check feature importance
X_feature = col_list
RF_feat = pd.DataFrame(data = XGB_gsearch.best_estimator_.feature_importances_, index = X_feature, columns=['importance'])
RF_feat.sort_values('importance').plot(kind='barh', title = 'XGB Feature Importance')
plt.show()
