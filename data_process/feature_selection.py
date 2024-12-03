# %%
'''
feature selection with Boruta
'''
import os, sys
from datetime import datetime
import argparse
import pandas as pd
import numpy as np
import torch
import optuna
from sklearn.model_selection import KFold
from boruta import BorutaPy
from BorutaShap import BorutaShap
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import json
import pickle
# 检测运行环境
def in_notebook():
    return 'IPKernelApp' in getattr(globals().get('get_ipython', lambda: None)(), 'config', {})

if in_notebook():
    from IPython.display import clear_output, display
    notebook_dir = os.getcwd()
    src_path = os.path.abspath(os.path.join(notebook_dir, '..'))
    N_TRIAL = 100 # boruta 特征选择次数
    OUTCOME_IX = 0
    IMPORTANCE_MEASURE = 'shap' # gini, shap, perm
else:
    src_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-n',metavar= 50, type=int, default=50,help='''optuna优化尝试次数''')
    parser.add_argument('-outcome_ix',metavar=0, type=int, default=0,help='''选择预测结局, 为 `get_ite_features()`返回的预设 outcomes 列表的索引''')
    parser.add_argument('-importance',metavar='gini', type=str, default='shap',help='''特征重要性度量方式''')
    sys_args = parser.parse_args()
    N_TRIAL = sys_args.n
    OUTCOME_IX = sys_args.outcome_ix
    IMPORTANCE_MEASURE = sys_args.importance

sys.path.append(src_path) if src_path not in sys.path else None
from src.utils import *
from src.model_utils import *
from src.setup import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'current device: {DEVICE}')

# %%
df = pd.read_csv(f'{DATA}/imputed/MIMIC_IV_clean_imputed.tsv.gz', sep='\t', index_col='ID')
cate_vars, cont_vars, outcomes = get_cleaned_vars('MIMIC_IV')

X = df[[*cate_vars, *cont_vars]].copy()
y = df[outcomes[OUTCOME_IX]].copy()

# %%
# model = XGBClassifier(n_estimators=200, 
#                       objective='binary:logistic', eval_metric='logloss',
#                       tree_method="hist",
#                       booster='gbtree',
#                       n_jobs=-1,
#                       device = DEVICE)

model = RandomForestClassifier(n_jobs=-1)

# %%
feature_selector = BorutaShap(
                              # model=model, 
                              # importance_measure='shap',
                              importance_measure=IMPORTANCE_MEASURE,
                              classification=True,
                              pvalue=0.05)

feature_selector.fit(X, y, 
                     n_trials=N_TRIAL, 
                     random_state=19960816,
                     sample=True,
                     train_or_test='test', 
                     normalize=True,
                     verbose=True)

# %%
feature_selector_path = f'{MODELS}/MIMIC_IV_boruta_risk_model_{outcomes[OUTCOME_IX]}_{IMPORTANCE_MEASURE}.pkl'
with open(feature_selector_path, 'wb') as file:
    pickle.dump(feature_selector, file)
print(f"Feature selector saved: {feature_selector_path}")


