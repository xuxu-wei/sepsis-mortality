# %%
'''
训练机器学习模型，预测 28天死亡率及院内死亡率
'''
import os, sys
from datetime import datetime
import argparse
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold 

# 检测运行环境
IN_NOTEBOOK = None
def in_notebook():
    return 'IPKernelApp' in getattr(globals().get('get_ipython', lambda: None)(), 'config', {})
    
if in_notebook():
    from IPython.display import clear_output, display
    notebook_dir = os.getcwd()
    src_path = os.path.abspath(os.path.join(notebook_dir, '..'))
    IN_NOTEBOOK = True
else:
    src_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    parser = argparse.ArgumentParser(description='')
    sys_args = parser.parse_args()
    IN_NOTEBOOK = False

sys.path.append(src_path) if src_path not in sys.path else None

from src.utils import *
from src.model_utils import *
from src.metrix import cal_ci, format_ci
from src.setup import *
from risk_setup import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'current device: {DEVICE}')
risk_ml_models = f'{MODELS}/risk_models/ML_ref_model/'
os.makedirs(risk_ml_models, exist_ok=True)

# %%
df = pd.read_csv(f'{DATA}/imputed/MIMIC_IV_clean_imputed.tsv.gz', sep='\t', index_col='ID')
df = df.sample(frac=0.1) if IN_NOTEBOOK else df
features, _, _, outcomes = get_risk_model_features()
X, y = load_data(df, outcome_ix=0) # 这里加载了 28-d mortality 作为预测目标

# load multi-task y
y = df[outcomes].copy() 

# standardization of X
std_processor = StandardScaler()
X_array = std_processor.fit_transform(X)
X = pd.DataFrame(X_array, index=X.index, columns=X.columns)
joblib.dump(std_processor, f'{risk_ml_models}/MIMIC_StandardScaler.joblib')

print(f'training data: {X.shape}')

# %%
# Initialize a stacking of LR, KNN, SVM, XGB, and RF
base_estimators = [('LR', LogisticRegression(penalty='l2', random_state=19960816)),
                   ('KNN', KNeighborsClassifier()),
                   ('SVM', SVC(probability=True, random_state=19960816)),
                   ('RF', RandomForestClassifier(max_depth=np.log2(X.shape[1]), bootstrap=True, n_jobs=-1, random_state=19960816)),
                #    ('XGB', XGBClassifier(n_estimators=100, n_jobs=-1, device=DEVICE, random_state=19960816))
                  ]

# stacking of base models
stacking_model = StackingClassifier(base_estimators,
                                    final_estimator=LogisticRegression(random_state=19960816), # learn weights for each base estimator
                                    stack_method='predict_proba',
                                    n_jobs=-1,
                                    passthrough=False,
                                    verbose=1)

# Multi-task Model
multi_task_model = MultiOutputClassifier(stacking_model, n_jobs=-1)

# %%
# Cross validation (performance estimation)
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=19960816)
aucs_1 = []
aucs_2 = []
for i, (train_index, val_index) in enumerate(kf.split(X, y.iloc[:,0])): # stratified by primary outcome (28d-mortality)
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    # train
    multi_task_model.fit(X_train, y_train)
    
    # eval
    y_hat_1 = multi_task_model.predict_proba(X_val)[0][:, 1] # predict proba for 28d-mortality
    y_hat_2 = multi_task_model.predict_proba(X_val)[1][:, 1] # predict proba for in-hospital mortality
    
    y_val_1 = np.array(y_val.iloc[:, 0]) # true 28d-death label
    y_val_2 = np.array(y_val.iloc[:, 1]) # true in-hospital death label
    
    auc_1 = roc_auc_score(y_val_1, y_hat_1)
    auc_2 = roc_auc_score(y_val_2, y_hat_2)
    
    aucs_1.append(auc_1)
    aucs_2.append(auc_2)
    
    print(f'Fold {i+1}: AUC of y1: {auc_1:.3f}, AUC of y2: {auc_2:.3f}')

# assume t-distribution for 95% CI calculation
mean_auc_1, lower_1, upper_1 = cal_ci(aucs_1, alpha=0.05, method='t')
mean_auc_2, lower_2, upper_2 = cal_ci(aucs_2, alpha=0.05, method='t')
print(f'AUC of {outcomes[0]}: {format_ci(mean_auc_1, lower_1, upper_1, 3)}')
print(f'AUC of {outcomes[1]}: {format_ci(mean_auc_2, lower_2, upper_2, 3)}')

# %%
# train reference model on whole training set and save for external validation.
multi_task_model.fit(X, y)
joblib.dump(multi_task_model, f'{risk_ml_models}/MIMIC_stacking.joblib')


