import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet   # 新增
import warnings

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# ==================== 参数配置 ====================
INPUT_DIR = "温湿度数据csv"
OUTPUT_DIR = "温室预测结果"
WINDOW_SIZE = 30
TEST_SPLIT_RATIO = 0.2
TARGET_COLS = ["温度(℃)", "湿度(%)"]
ML_MODELS = {
    "线性回归": LinearRegression(),
    "随机森林": RandomForestRegressor(n_estimators=200, random_state=42),
    "梯度提升": GradientBoostingRegressor(n_estimators=200, random_state=42),
    "SVR": SVR(),
    "XGBoost": XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5,
                            subsample=0.8, colsample_bytree=0.8, random_state=42)
}
# 时间序列模型（单输出），现在包含 Prophet
TS_MODELS = ['ARIMA', 'ExpSmoothing', 'Prophet']

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)

# ==================== 辅助函数（保持不变） ====================
def create_features(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

def evaluate(y_true, y_pred):
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    metrics = {}
    for i in range(y_true.shape[1]):
        metrics[f'目标{i}'] = {
            'MAE': mean_absolute_error(y_true[:, i], y_pred[:, i]),
            'MSE': mean_squared_error(y_true[:, i], y_pred[:, i]),
            'R2': r2_score(y_true[:, i], y_pred[:, i])
        }
    avg_mae = np.mean([m['MAE'] for m in metrics.values()])
    avg_mse = np.mean([m['MSE'] for m in metrics.values()])
    avg_r2 = np.mean([m['R2'] for m in metrics.values()])
    return {'per_target': metrics, 'avg_mae': avg_mae, 'avg_mse': avg_mse, 'avg_r2': avg_r2}

def plot_comparison(truth, preds, title, save_path, target_names):
    if truth.ndim == 1:
        truth = truth.reshape(-1, 1)
        preds = preds.reshape(-1, 1)
    n_targets = truth.shape[1]
    for i in range(n_targets):
        plt.figure(figsize=(12, 6))
        plt.plot(truth[:, i], label="真实值", linewidth=2)
        plt.plot(preds[:, i], label="预测值", linewidth=2)
        plt.xlabel("样本序号")
        plt.ylabel(target_names[i] if i < len(target_names) else f"目标{i+1}")
        plt.title(f"{title} - {target_names[i] if i < len(target_names) else f'目标{i+1}'}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        name_part = target_names[i].replace('(℃)', '').replace('(%)', '') if i < len(target_names) else f"目标{i+1}"
        save_path_i = save_path.replace('.png', f'_{name_part}.png')
        os.makedirs(os.path.dirname(save_path_i), exist_ok=True)
        plt.savefig(save_path_i, dpi=150)
        plt.close()

def plot_model_comparison(results_dict, metric='avg_r2', save_path=None):
    models = list(results_dict.keys())
    values = [results_dict[m][metric] for m in models]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, values, color='skyblue')
    plt.title(f'各模型{metric}对比')
    plt.xlabel('模型')
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(values),
                 f'{val:.4f}', ha='center', va='bottom')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close()

# ==================== 模型训练 ====================
def train_ml_model(model, X_train, y_train, X_test):
    multi_model = MultiOutputRegressor(model)
    multi_model.fit(X_train, y_train)
    pred = multi_model.predict(X_test)
    return pred

def train_arima(train, test):
    try:
        from pmdarima import auto_arima
        model = auto_arima(train, seasonal=True, m=24, stepwise=True, trace=False,
                           error_action='ignore', suppress_warnings=True)
        pred = model.predict(n_periods=len(test))
    except ImportError:
        model = ARIMA(train, order=(5,1,0))
        model_fit = model.fit()
        pred = model_fit.forecast(steps=len(test))
    return pred

def train_exp_smoothing(train, test):
    model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=24)
    try:
        model_fit = model.fit()
        pred = model_fit.forecast(steps=len(test))
    except:
        model = ExponentialSmoothing(train, trend='add', seasonal=None)
        model_fit = model.fit()
        pred = model_fit.forecast(steps=len(test))
    return pred

def train_prophet(train, test, freq='H', dates=None):
    """
    train, test: 一维 numpy 数组
    dates: 可选，与 train+test 等长的日期列表（用于拟合 Prophet）
           如果为 None，则自动生成从 2020-01-01 开始的虚拟日期
    """
    n_train = len(train)
    if dates is None:
        # 生成虚拟日期，假设频率为 freq
        start = pd.Timestamp('2020-01-01 00:00:00')
        all_dates = pd.date_range(start=start, periods=n_train + len(test), freq=freq)
    else:
        all_dates = dates
        if len(all_dates) < n_train + len(test):
            # 如果提供的日期不足，扩展
            last = all_dates[-1]
            all_dates = all_dates.tolist() + pd.date_range(start=last + pd.Timedelta(1, unit=freq),
                                                           periods=len(test), freq=freq).tolist()
    train_dates = all_dates[:n_train]
    test_dates = all_dates[n_train:]

    df_train = pd.DataFrame({'ds': train_dates, 'y': train})
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    model.fit(df_train)
    future = pd.DataFrame({'ds': test_dates})
    forecast = model.predict(future)
    return forecast['yhat'].values

# ==================== 核心预测函数（增加 date_index 参数） ====================
def process_series(series_dict, base_name, output_subdir="figures", date_index=None):
    """
    series_dict: dict, key为目标名，value为对应的一维numpy数组（时间序列）
    base_name: 结果文件前缀
    output_subdir: 图片保存子目录
    date_index: 可选，与序列等长的日期索引（用于 Prophet 模型）
    """
    full_subdir = os.path.join(OUTPUT_DIR, output_subdir)
    os.makedirs(full_subdir, exist_ok=True)

    target_names = list(series_dict.keys())
    series_list = [series_dict[name] for name in target_names]
    series_multi = np.column_stack(series_list)

    # 构造 ML 特征
    all_features, all_targets = [], []
    for i in range(len(series_multi) - WINDOW_SIZE):
        all_features.append(series_multi[i:i+WINDOW_SIZE].flatten())
        all_targets.append(series_multi[i+WINDOW_SIZE])
    all_features = np.array(all_features)
    all_targets = np.array(all_targets)

    split_idx = int(len(all_features) * (1 - TEST_SPLIT_RATIO))
    X_train, X_test = all_features[:split_idx], all_features[split_idx:]
    y_train, y_test = all_targets[:split_idx], all_targets[split_idx:]

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    scaler_y = MinMaxScaler()
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_test_scaled = scaler_y.transform(y_test)

    results = {}

    # 机器学习模型
    for name, model in ML_MODELS.items():
        print(f"  正在运行ML模型: {name}")
        pred_scaled = train_ml_model(model, X_train_scaled, y_train_scaled, X_test_scaled)
        pred = scaler_y.inverse_transform(pred_scaled)
        truth = scaler_y.inverse_transform(y_test_scaled)
        metrics = evaluate(truth, pred)
        results[name] = {
            'avg_mae': metrics['avg_mae'],
            'avg_mse': metrics['avg_mse'],
            'avg_r2': metrics['avg_r2'],
            'per_target': metrics['per_target']
        }
        print(f"    完成 {name} - 平均MAE: {metrics['avg_mae']:.4f}, 平均R²: {metrics['avg_r2']:.4f}")

        img_name = f"{base_name}_{name}_对比图.png"
        img_path = os.path.join(OUTPUT_DIR, output_subdir, img_name)
        plot_comparison(truth, pred, f"{base_name} - {name} 预测结果", img_path, target_names)

    # 时间序列模型（ARIMA, ExpSmoothing, Prophet）
    for name in TS_MODELS:
        print(f"  正在运行时序模型: {name}")
        target_metrics = {}
        all_pred = []
        for i, target_name in enumerate(target_names):
            series_single = series_dict[target_name]
            split_idx_single = int(len(series_single) * (1 - TEST_SPLIT_RATIO))
            train_orig = series_single[:split_idx_single]
            test_orig = series_single[split_idx_single:]

            if name == 'ARIMA':
                pred = train_arima(train_orig, test_orig)
            elif name == 'ExpSmoothing':
                pred = train_exp_smoothing(train_orig, test_orig)
            elif name == 'Prophet':
                # 提取日期索引（如果提供了且长度匹配）
                if date_index is not None and len(date_index) == len(series_single):
                    dates = date_index
                else:
                    dates = None
                pred = train_prophet(train_orig, test_orig, freq='H', dates=dates)
            else:
                raise ValueError(f"Unknown time series model: {name}")

            all_pred.append(pred)
            mae = mean_absolute_error(test_orig, pred)
            mse = mean_squared_error(test_orig, pred)
            r2 = r2_score(test_orig, pred)
            target_metrics[target_name] = {'MAE': mae, 'MSE': mse, 'R2': r2}
        avg_mae = np.mean([target_metrics[t]['MAE'] for t in target_names])
        avg_mse = np.mean([target_metrics[t]['MSE'] for t in target_names])
        avg_r2 = np.mean([target_metrics[t]['R2'] for t in target_names])
        results[name] = {
            'avg_mae': avg_mae,
            'avg_mse': avg_mse,
            'avg_r2': avg_r2,
            'per_target': target_metrics
        }
        print(f"    完成 {name} - 平均MAE: {avg_mae:.4f}, 平均R²: {avg_r2:.4f}")

        # 绘图
        for i, target_name in enumerate(target_names):
            pred = all_pred[i]
            truth = series_dict[target_name][int(len(series_dict[target_name]) * (1 - TEST_SPLIT_RATIO)):]
            img_name = f"{base_name}_{name}_{target_name.replace('(℃)', '').replace('(%)', '')}_对比图.png"
            img_path = os.path.join(OUTPUT_DIR, output_subdir, img_name)
            plot_comparison(truth, pred, f"{base_name} - {name} 预测结果 ({target_name})", img_path, [target_name])

    return results

# ==================== 聚合所有传感器数据 ====================
def aggregate_sensors(input_dir, target_cols, freq='H'):
    all_agg = {}
    for target in target_cols:
        all_dfs = []
        for filename in os.listdir(input_dir):
            if not filename.endswith('.csv'):
                continue
            filepath = os.path.join(input_dir, filename)
            df = pd.read_csv(filepath, parse_dates=['日期'], encoding='utf-8-sig')
            if target not in df.columns:
                continue
            sub = df[['日期', target]].set_index('日期')
            sub = sub.resample(freq).mean()
            all_dfs.append(sub)
        if not all_dfs:
            raise ValueError(f"未找到目标列 '{target}' 的数据")
        combined = pd.concat(all_dfs, axis=1)
        agg_series = combined.mean(axis=1, skipna=True)
        agg_series = agg_series.fillna(method='ffill')
        all_agg[target] = agg_series
    agg_df = pd.DataFrame(all_agg).reset_index()
    agg_df.columns = ['ds'] + target_cols
    return agg_df

# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser(description='温室温湿度预测脚本')
    parser.add_argument('--mode', type=str, default='all', choices=['aggregate', 'single', 'all'],
                        help='运行模式：aggregate=整体大棚预测，single=单个传感器预测，all=两者都运行（默认）')
    args = parser.parse_args()

    # 整体预测模式
    if args.mode in ['aggregate', 'all']:
        print("\n" + "="*50)
        print("=== 整体大棚预测模式 ===")
        print("正在聚合所有传感器数据...")
        agg_df = aggregate_sensors(INPUT_DIR, TARGET_COLS)
        if agg_df.empty:
            print("聚合失败，未找到有效数据。")
        else:
            agg_df.to_csv(os.path.join(OUTPUT_DIR, "整体大棚_聚合数据.csv"), index=False, encoding='utf-8-sig')
            print(f"聚合数据长度：{len(agg_df)} 条，时间范围：{agg_df['ds'].min()} 至 {agg_df['ds'].max()}")
            series_dict = {col: agg_df[col].values for col in TARGET_COLS}
            # 传递日期索引（用于 Prophet）
            date_index = agg_df['ds'].values
            print("开始对整体大棚进行预测...")
            results = process_series(series_dict, base_name="整体大棚", output_subdir="整体大棚_图表", date_index=date_index)

            summary = []
            for model, metrics in results.items():
                summary.append({
                    '模型': model,
                    '平均MAE': metrics['avg_mae'],
                    '平均MSE': metrics['avg_mse'],
                    '平均R²': metrics['avg_r2']
                })
            summary_df = pd.DataFrame(summary)
            summary_df.to_csv(os.path.join(OUTPUT_DIR, "整体大棚_评估结果.csv"), index=False, encoding='utf-8-sig')

            plot_model_comparison(results, metric='avg_r2',
                                  save_path=os.path.join(OUTPUT_DIR, "整体大棚_图表", "模型_R2对比.png"))
            plot_model_comparison(results, metric='avg_mae',
                                  save_path=os.path.join(OUTPUT_DIR, "整体大棚_图表", "模型_MAE对比.png"))
            print("整体大棚预测完成！")

    # 单个传感器预测模式
    if args.mode in ['single', 'all']:
        print("\n" + "="*50)
        print("=== 单个传感器预测模式 ===")
        files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
        if not files:
            print(f"未在 {INPUT_DIR} 中找到任何CSV文件")
        else:
            all_results = []
            for filename in files:
                filepath = os.path.join(INPUT_DIR, filename)
                base_name = os.path.splitext(filename)[0]
                print(f"\n处理文件: {base_name}")
                df = pd.read_csv(filepath, parse_dates=['日期'], encoding='utf-8-sig')
                series_dict = {col: df[col].values for col in TARGET_COLS if col in df.columns}
                # 传递日期列
                date_index = df['日期'].values
                results = process_series(series_dict, base_name=base_name, output_subdir="figures", date_index=date_index)
                for model, metrics in results.items():
                    all_results.append({
                        '文件': base_name,
                        '模型': model,
                        '平均MAE': metrics['avg_mae'],
                        '平均MSE': metrics['avg_mse'],
                        '平均R²': metrics['avg_r2']
                    })
            summary_df = pd.DataFrame(all_results)
            summary_df.to_csv(os.path.join(OUTPUT_DIR, "所有传感器评估汇总.csv"), index=False, encoding='utf-8-sig')
            print("\n所有传感器处理完成！")

    print(f"\n所有结果已保存到：{OUTPUT_DIR}")

if __name__ == "__main__":
    main()