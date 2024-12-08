# %%
import os, sys
from datetime import datetime
import argparse
import pandas as pd
import torch
import optuna
from sklearn.model_selection import KFold
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
    RUN_MODE = 'tuning' # reload: 重现study; tuning 搜索超参数
    N_TRIAL = 3
    OUTCOME_IX = 1
else:
    src_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-n',metavar= 50, type=int, default=50,help='''optuna优化尝试次数''')
    parser.add_argument('-outcome_ix',metavar= 0, type=int, default=0,help='''选择预测结局, 为 `get_ite_features()`返回的预设 outcomes 列表的索引''')
    sys_args = parser.parse_args()
    N_TRIAL = sys_args.n
    OUTCOME_IX = sys_args.outcome_ix
    RUN_MODE = 'tuning' # 脚本模式只做tuing!

sys.path.append(src_path) if src_path not in sys.path else None

from src.utils import *
from src.model_utils import *
from src.setup import *
from ite_setup import *
from ganite_mod import Ganite, GaniteRegressor
from ganite_mod.utils.metrics import *

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'current device: {DEVICE}')

# %%
df = pd.read_csv(f'{DATA}/imputed/EXIT_SEP_clean_imputed.tsv.gz', sep='\t', index_col='ID')
features, _, _, treatment, outcomes = get_ite_features()
current_outcome = outcomes[OUTCOME_IX] # 设置预测目标

df_train = df.sample(frac=0.7, random_state=19960816)
df_test = df[~df.index.isin(df_train.index)].copy()
X, W, y = load_data(df, outcome_ix=OUTCOME_IX)

X = np.array(X)
W = np.array(W)
y = np.array(y)

# %% [markdown]
# # 超参数搜索

# %%
if RUN_MODE=='tuning':
    # 建立文件输出路径
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M")

    optuna_result_path = f'{MODELS}/GANITE_optuna-{current_outcome}-{formatted_time}-{sys.platform}/'
    # optuna_fig_path = f'{FIGS}/GANITE_optuna-{current_outcome}-{formatted_time}-{sys.platform}/'
    optuna_fig_path = optuna_result_path # 统一输出路径

    os.makedirs(optuna_fig_path, exist_ok=True)
    os.makedirs(optuna_result_path, exist_ok=True)


    def objective(trial):
        # 定义需要调优的超参数范围
        dim_hidden = trial.suggest_int("dim_hidden", 70, 400, step=10)
        alpha = trial.suggest_float("alpha", 0.0, 3.0, step=0.05)
        beta = trial.suggest_float("beta", 0.0, 3.0, step=0.05)
        depth = trial.suggest_int("depth", 1, 5, step=1)
        minibatch_size = trial.suggest_categorical("minibatch_size", [200, 300, 400])
        num_iterations = trial.suggest_int("num_iterations", 1000, 2500, step=500)
        num_discr_iterations = trial.suggest_categorical("num_discr_iterations", [1, 2, 3])

        # 初始化模型
        model = GaniteRegressor(
            dim_in=X.shape[1],
            binary_y=True,
            dim_hidden=dim_hidden,
            alpha=alpha,
            beta=beta,
            depth=depth,
            minibatch_size=minibatch_size,
            num_iterations=num_iterations,
            num_discr_iterations=num_discr_iterations,
        )

        # 实现交叉验证
        kf = KFold(n_splits=10, shuffle=True, random_state=19960816)
        aucs = []
        scores = []
        ate_losses_ob = []
        ate_losses = []

        for train_index, val_index in kf.split(X):
            # 划分训练集和验证集
            X_train, X_val = X[train_index], X[val_index]
            T_train, T_val = W[train_index], W[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            # 训练模型
            model.fit((X_train, T_train), y_train)
            
            # 验证模型并记录分数
            score_temp = model.score((X_val, T_val), y_val)  # 默认负均方误差
            neg_ate_l1_loss_ob_temp = model.ate_l1_loss((X_val, T_val), y_val, eval_strategy='observed_only')  # 负 ATE 误差, 仅比较观测组间误差
            neg_ate_l1_loss_temp = model.ate_l1_loss((X_val, T_val), y_val, eval_strategy='mean_ITE')  # 负 ATE 误差
            auc = model.roc_auc((X_val, T_val), y_val, average='weighted')

            aucs.append(auc)
            scores.append(score_temp)
            ate_losses_ob.append(neg_ate_l1_loss_ob_temp)
            ate_losses.append(neg_ate_l1_loss_temp)

        # 返回平均交叉验证分数（负均方误差）
        return np.mean(aucs), np.mean(scores), np.mean(ate_losses_ob), np.mean(ate_losses)

    # 日志功能：设置 Optuna 的日志级别
    # optuna.logging.set_verbosity(optuna.logging.INFO)

    # 实时打印当前最佳结果
    def trial_callback(study, trial):
        if in_notebook():
            clear_output(wait=True)  # 清除之前的输出
            df_trials = study.trials_dataframe()  # 获取当前的试验数据
            display(df_trials)  # 动态显示最新的 dataframe

            # 多目标优化时 打印 Pareto 前沿解
            if len(study.directions) > 1:
                pareto_front = study.best_trials
                print(f"Number of Pareto optimal solutions: {len(pareto_front)}")
                for i, trial in enumerate(pareto_front):
                    print(f"Pareto solution {i}: Values {trial.values}, Params {trial.params}")
            else:
                print(f"Current best value: {study.best_value}")
                print(f"Current best parameters: {study.best_params}")
                print(f"Current best trial: {study.best_trials}")
        else:
            # 多目标优化时 打印首选 Pareto 前沿解的结果和参数
            if len(study.directions) > 1:
                pareto_front = study.best_trials
                best_trial = pareto_front[0]
                print(f"Trial {trial.number}/{N_TRIAL} finished with value: {trial.values} and parameters: {trial.params} | crrent best value: {best_trial.values} and parameters: {best_trial.params} | Number of Pareto optimal solutions: {len(pareto_front)}")
            else:
                print(f"Trial {trial.number}/{N_TRIAL} finished with value: {trial.value} and parameters: {trial.params} | crrent best value: {study.best_value} and parameters: {study.best_params}")


    # 使用 Optuna 优化
    study = optuna.create_study(directions=["maximize", "maximize", "maximize", "maximize",])  # 或 "minimize"，取决于评分标准
    study.optimize(objective, n_trials=N_TRIAL, callbacks=[trial_callback])

    # 保存实验结果
    with open(f"{optuna_result_path}/optuna_study.pkl", "wb") as f:
        print('调参结束，正在保存optuna调参试验结果')
        pickle.dump(study, f)

    # 获取 Pareto 前沿解
    pareto_front = study.best_trials
    print("Pareto Front Solutions:")
    for trial in pareto_front:
        print(f"Trial {trial.number}: Values {trial.values}, Params {trial.params}")

    # 保存 Pareto 解到文件
    pareto_data = [
        {"trial_number": trial.number, "values": trial.values, "params": trial.params}
        for trial in pareto_front
    ]
    with open(f"{optuna_result_path}/pareto_solutions.json", "w") as f:
        json.dump(pareto_data, f)


    # 保存完整调参历史为 xlsx 文件
    df_trials = study.trials_dataframe()
    df_trials.to_excel(f"{optuna_result_path}/tuning_history.xlsx", index=False)

    # 使用一个 Pareto 最优解重新初始化模型
    best_trial = pareto_front[0] # 选择第一个 Pareto 解
    best_params = best_trial.params
    # 使用最佳参数重新初始化模型
    best_model = GaniteRegressor(
        dim_in=X.shape[1],
        binary_y=True,
        dim_hidden=best_params["dim_hidden"],
        alpha=best_params["alpha"],
        beta=best_params["beta"],
        depth=best_params["depth"],
        num_iterations=best_params["num_iterations"],
        num_discr_iterations=best_params["num_discr_iterations"],
    )

    # 训练最佳模型
    print('使用最佳参数在全集上模型')
    best_model.fit((X, W), y)

    # 保存最佳模型
    print('训练完成，保存模型参数')
    torch.save(best_model.state_dict(), f"{optuna_result_path}/GANITE_weights.pth")

# %%
if RUN_MODE=='reload':
    optuna_result_path = f'{MODELS}/GANITE_optuna-28d_mortality-2024-11-28_23-14-linux/'
    with open(f"{optuna_result_path}/optuna_study.pkl", "rb") as f:
        print('正在加载指定optuna调参试验结果')
        study = pickle.load(f)

from optuna.visualization import (
    plot_parallel_coordinate,
    plot_param_importances,
    plot_contour,
    plot_slice,
    plot_optimization_history,
    plot_pareto_front,
)

# 绘制不同的图表
target_args_1 = dict(target = lambda t: -t.values[0], target_name="AUC")
target_args_2 = dict(target = lambda t: -t.values[1], target_name="Berier Score")
target_args_3 = dict(target = lambda t: -t.values[2], target_name="ATE_observed L1-loss")
target_args_4 = dict(target = lambda t: -t.values[3], target_name="ATE L1-loss")
targets_args = dict(targets = lambda t: [t.values[0], -t.values[3]], target_names=["AUC", "ΔATE"])

# 并行坐标图
parallel_coordinate_fig = plot_parallel_coordinate(study, **target_args_1)
parallel_coordinate_fig.update_layout(width=800, height=600)
parallel_coordinate_fig.show() if in_notebook() else None
parallel_coordinate_fig.write_image(f"{optuna_fig_path}/parallel_coordinate_fig.svg", format='svg', scale=2, width=700, height=500) if not in_notebook() else None

# 参数重要性图
param_importance_fig = plot_param_importances(study, **target_args_1)
param_importance_fig.update_layout(width=800, height=600)
param_importance_fig.show() if in_notebook() else None
param_importance_fig.write_image(f"{optuna_fig_path}/param_importance_fig.svg", format='svg', scale=2, width=700, height=500) if not in_notebook() else None

# 平行曲面图
contour_fig = plot_contour(study, **target_args_1)
contour_fig.update_layout(width=1200, height=1200)
contour_fig.show() if in_notebook() else None
contour_fig.write_image(f"{optuna_fig_path}/contour_fig.svg", format='svg', scale=2, width=1200, height=1200) if not in_notebook() else None

# 超参数分布图
slice_fig = plot_slice(study, **target_args_1)
slice_fig.show() if in_notebook() else None
slice_fig.write_image(f"{optuna_fig_path}/slice_fig.svg", format='svg', scale=2, width=2500, height=400) if not in_notebook() else None

# 优化历史图
optimization_history_fig = plot_optimization_history(study, **target_args_1)
optimization_history_fig.update_layout(width=700, height=500)
optimization_history_fig.show() if in_notebook() else None
optimization_history_fig.write_image(f"{optuna_fig_path}/optimization_history_fig.svg", format='svg', scale=2, width=700, height=500) if not in_notebook() else None

# Pareto 前沿图（仅适用于多目标优化）
if len(study.directions) > 1:
    pareto_fig = plot_pareto_front(study, **targets_args)
    pareto_fig.update_layout(width=1200, height=800)
    pareto_fig.show() if in_notebook() else None
    pareto_fig.write_image(f"{optuna_fig_path}/pareto_fig.svg", format='svg', scale=2, width=1200, height=1200) if not in_notebook() else None


# %% [markdown]
# # 手动调参

# %%
# X_train, W_train, y_train = load_data(df_train)
# X_test, W_test, y_test = load_data(df_test)

# # modified GANITE
# model = Ganite(dim_in=X.shape[1],
#                binary_y=True,
#                dim_hidden=300,
#                alpha = 0.3,
#                beta = 0.3,
#                depth = 3,
#                minibatch_size = 200,
#                num_iterations=2500,
#                num_discr_iterations=3,
#                )

# if RUN_MODE == 'train':
#     model = model.fit(X_train, W_train, y_train)
#     torch.save(model.state_dict(), f"{MODELS}/GANITE.pth")
# else:
#     model.load_state_dict(torch.load(f"{MODELS}/GANITE_best_weights_manual.pth", weights_only=True))
#     model.eval()  # 切换到评估模式（重要！）
#     print("模型参数已加载！")

# # 测试集测试
# Y_1_test, Y_0_test, ITE_test = model(X_test)
# df_test['potential_y1'] = Y_1_test.cpu()
# df_test['potential_y0'] = Y_0_test.cpu()
# df_test['ITE'] = ITE_test.cpu()
# df_test['y_pred_observed'] = df_test.apply(lambda row: row['potential_y1'] if row[treatment]==1 else row['potential_y0'], axis=1)

# ATE_test = RCT_ATE(df_test[treatment], df_test[current_outcome])
# ATE_pred_ob = RCT_ATE(df_test[treatment], df_test['y_pred_observed'])
# ATE_pred = df_test['ITE'].mean()

# print(f'实际ATE: {ATE_test:.4f}, 预测实际ATE: {ATE_pred_ob:.4f}, ATE误差: {ATE_test - ATE_pred_ob:.4f}, 预测组间ATE: {ATE_pred:.4f}')


