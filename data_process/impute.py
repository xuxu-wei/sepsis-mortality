
import argparse
import os, sys, re, gc, time, datetime
from IPython.display import display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Tuple, Set, List, Any, Iterable, Optional
import pickle
from sklearn.experimental import enable_iterative_imputer #启用多重填补工具
from sklearn.impute import IterativeImputer,MissingIndicator #导入多重填补器,缺失指示器
from sklearn.linear_model import BayesianRidge,LogisticRegression

sns.set_theme('paper')
# 检测运行环境
def in_notebook():
    return 'IPKernelApp' in getattr(globals().get('get_ipython', lambda: None)(), 'config', {})

SCRIPT_MODE = 'notebook' if in_notebook() else 'shell'
if SCRIPT_MODE == 'notebook':
    notebook_dir = os.getcwd()
    src_path = os.path.abspath(os.path.join(notebook_dir, '..'))
    DATASET = 'EXIT_SEP'

elif SCRIPT_MODE == 'shell':
    # 设置目录(.py)
    src_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', metavar= 'EXIT_SEP', type=str, default=None, help='''指定要填补的数据集: MIMIC_IV, eICU, EXIT_SEP, EXIT_SEP_worst_both, EXIT_SEP_worst_xbj''')

    sys_args = parser.parse_args()
    DATASET = sys_args.data

# 添加到 sys.path
if src_path not in sys.path:
    sys.path.append(src_path)

from src.utils import *
from src.setup import *

imputation_path = f'{DATA}/imputed/'
if not os.path.exists(imputation_path):
    os.makedirs(imputation_path)

data_file_dict = {
    'EXIT_SEP':'EXIT_SEP_clean.tsv.gz',
    'EXIT_SEP_worst_both':'EXIT_SEP_worse_case_both_die.tsv.gz',
    'EXIT_SEP_worst_xbj':'EXIT_SEP_worse_case_xbj_die.tsv.gz',
    'eICU':'eICU.tsv.gz',
    'MIMIC_IV':'MIMIC_IV.tsv.gz',
}
data_file = data_file_dict[DATASET]
fname = data_file.split('.')[0]
imputation_output = f'{imputation_path}/{fname}_imputed.tsv.gz'
print(f'output file: {os.path.realpath(imputation_output)}')

df = pd.read_csv(f'{DATA}/{data_file}', sep='\t', index_col='ID')
cate_vars, cont_vars, outcomes = get_cleaned_vars(DATASET)

# 排除主要结局缺失的数据
mask_primary_outcome = df[outcomes[0]].notna()
df = df[mask_primary_outcome].copy()
gc.collect()

# 规定缺失填充的下界
def infer_lower(x: pd.Series, cate_vars: List[str], cont_vars: List[str]):
    """
    获取每一列的填充下界。
    
    参数:
    x (pd.Series): 数据列
    cate_vars (List[str]): 分类变量列表
    cont_vars (List[str]): 连续变量列表
    
    返回:
    float: 填充下界
    """
    # 若为分类变量，直接使用最小值
    if x.name in cate_vars:
        return x.min()
    # 若为连续变量，下界为左侧2.5% - IQR
    if x.name in cont_vars:
        return x.quantile(0.025) - 2*(x.quantile(0.75)-x.quantile(0.25))
    else:
        return x.min()

# 规定缺失填充的上界
def infer_upper(x: pd.Series, cate_vars: List[str], cont_vars: List[str]):
    """
    获取每一列的填充上界。
    
    参数:
    x (pd.Series): 数据列
    cate_vars (List[str]): 分类变量列表
    cont_vars (List[str]): 连续变量列表
    
    返回:
    float: 填充上界
    """
    # 若为分类变量，直接使用最大值
    if x.name in cate_vars:
        return x.max()
    # 若为连续变量，上界为右侧97.5% +2倍四分位距
    if x.name in cont_vars:
        return x.quantile(0.975) + 2*(x.quantile(0.75)-x.quantile(0.25))
    else:
        return x.max()

# 打印当前数据缺失信息
def print_missing_info(data: pd.DataFrame, cate_vars: List[str], cont_vars: List[str]):
    """
    打印当前数据的缺失信息。
    
    参数:
    data (pd.DataFrame): 输入数据框
    cate_vars (List[str]): 分类变量列表
    cont_vars (List[str]): 连续变量列表
    """
    print('----------------------当前缺失情况----------------------')
    print('--------------------分类变量--------------------')
    print(data[cate_vars].isna().sum().sort_values(ascending=False)/len(df))
    print(f'分类变量无缺失：{data[cate_vars].notna().all().all()}')
    print('--------------------连续变量--------------------')
    print(data[cont_vars].isna().sum().sort_values(ascending=False)/len(df))
    print(f'连续变量无缺失：{data[cont_vars].notna().all().all()}')


# %%
# 敏感性分析数据集处理(仅完整数据，不填补)

# 主数据集分析
print(f'缺失填补前')
print_missing_info(df, cate_vars, cont_vars)

df_impute_model = df.copy()
imput_lower = df_impute_model.agg(lambda x: infer_lower(x, cate_vars, cont_vars))
imput_upper = df_impute_model.agg(lambda x: infer_upper(x, cate_vars, cont_vars))

cont_estimator = BayesianRidge()
cate_estimator = LogisticRegression(penalty='l2', max_iter=10000)

cont_imputer = IterativeImputer(
    estimator=cont_estimator,
    initial_strategy='median', # 连续变量采用中位数初始化，其他参数设置与分类变量相同
    imputation_order='ascending', # 从缺失最少的变量开始填补
    n_nearest_features=None, # 使用所有可用变量
    max_iter=200,
    tol=0.001,
    verbose=2,
    add_indicator=False,
    random_state=19960816,
    min_value=imput_lower,
    max_value=imput_upper,
    )

cate_imputer = IterativeImputer(
    estimator=cate_estimator,
    initial_strategy='most_frequent', # 连续变量采用中位数初始化，其他参数设置与分类变量相同
    imputation_order='ascending', # 从缺失最少的变量开始填补
    n_nearest_features=None, # 使用所有可用变量
    max_iter=200,
    tol=0.001,
    verbose=2,
    add_indicator=False,
    random_state=19960816,
    skip_complete=True,
    min_value=imput_lower,
    max_value=imput_upper,
    )


# 连续变量填补
if (not cont_vars) or (df_impute_model[cont_vars].notna().all().all()):
    print('连续变量无缺失， 无需填补')
else:
    print('执行 连续变量填补...')
    X = cont_imputer.fit_transform(df_impute_model) # 利用整个数据集
    df_cont_imputed = pd.DataFrame(X, index=df_impute_model.index, columns=df_impute_model.columns)
    for var in cont_vars:
        df[var] = df_cont_imputed[var] # 将填补后的连续变量赋值给原始数据
        df_impute_model[var] = df_cont_imputed[var]  # 将填补后的连续变量赋值给建模数据 继续用于分类变量填补
    # # 打印填补信息
    print(f'填补变量：{len(cont_imputer.feature_names_in_)}个\n',cont_imputer.feature_names_in_)
    print_missing_info(df, cate_vars, cont_vars)
    del df_cont_imputed, cont_imputer
    gc.collect()

# 分类变量填补
if (not cate_vars) or (df_impute_model[cate_vars].notna().all().all()):
    print('连续变量无缺失， 无需填补')
else:
    print('执行 分类变量填补...')
    X = cate_imputer.fit_transform(df_impute_model)
    print('填补完成')
    df_all_imputed = pd.DataFrame(X, index=df_impute_model.index, columns=df_impute_model.columns)
    for var in cate_vars:
        df[var] = df_all_imputed[var]
    # 打印填补信息
    print(f'填补变量：{len(cate_imputer.feature_names_in_)}个\n',cate_imputer.feature_names_in_)
    print_missing_info(df, cate_vars, cont_vars)
    del df_all_imputed, cate_imputer, df_impute_model
    gc.collect()

df.to_csv(imputation_output, sep='\t', compression='gzip')
print(f'输出填补数据: {imputation_output}')

# %%
print('重新读取填补后数据进行检查')
df_test = pd.read_csv(imputation_output, sep='\t', index_col='ID')
print_missing_info(df_test, cate_vars, cont_vars)
print(df_test.describe())
