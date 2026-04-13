import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# 设置 matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 参数配置 ====================
INPUT_DIR = "温湿度数据csv"
OUTPUT_DIR = "预测结果"                 # 所有结果存放根目录
WINDOW_SIZE = 24                       # 使用过去24小时预测下一小时

# 创建必要的目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)

# ==================== 聚合函数 ====================
def aggregate_sensors(input_dir, target_col, freq='h'):
    """读取所有CSV，重采样到统一频率，计算每个时间点的平均值"""
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"输入目录 '{input_dir}' 不存在，请创建并放入CSV文件。")
    all_dfs = []
    for filename in os.listdir(input_dir):
        if not filename.endswith('.csv'):
            continue
        filepath = os.path.join(input_dir, filename)
        df = pd.read_csv(filepath, parse_dates=['日期'], encoding='utf-8-sig')
        if target_col not in df.columns:
            continue
        sub = df[['日期', target_col]].set_index('日期')
        sub = sub.resample(freq).mean()
        all_dfs.append(sub)
    if not all_dfs:
        raise ValueError(f"未找到目标列 '{target_col}' 的数据")
    combined = pd.concat(all_dfs, axis=1)
    agg_series = combined.mean(axis=1, skipna=True)
    agg_series = agg_series.ffill()
    agg_df = agg_series.reset_index()
    agg_df.columns = ['ds', 'y']
    return agg_df

# ==================== 线性回归预测函数 ====================
def predict_linear(df, target_name, periods=7*24, window=WINDOW_SIZE, save_img_path=None):
    """
    使用线性回归（滑动窗口）预测未来 periods 小时
    返回预测结果 DataFrame
    """
    series = df['y'].values
    # 构造训练数据
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i+window])
        y.append(series[i+window])
    X = np.array(X)
    y = np.array(y)

    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 训练模型
    model = LinearRegression()
    model.fit(X_scaled, y)

    # 多步预测：使用最后 window 个真实值作为初始输入
    last_window = series[-window:].copy()
    predictions = []
    for _ in range(periods):
        last_scaled = scaler.transform(last_window.reshape(1, -1))
        pred = model.predict(last_scaled)[0]
        predictions.append(pred)
        last_window = np.append(last_window[1:], pred)

    # 生成未来日期
    last_date = df['ds'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(1, unit='h'), periods=periods, freq='h')

    # 裁剪预测值，使其符合物理意义
    if target_name == '温度(℃)':
        predictions = np.clip(predictions, 0, 40)
        lower = np.clip(np.array(predictions) - 2, 0, 40)
        upper = np.clip(np.array(predictions) + 2, 0, 40)
    else:  # 湿度(%)
        predictions = np.clip(predictions, 0, 100)
        lower = np.clip(np.array(predictions) - 5, 0, 100)
        upper = np.clip(np.array(predictions) + 5, 0, 100)

    pred_df = pd.DataFrame({
        '时间': future_dates,
        '预测值': predictions,
        '下限': lower,
        '上限': upper
    })

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(df['ds'], df['y'], label='历史数据', linewidth=2)
    plt.plot(pred_df['时间'], pred_df['预测值'], label='预测值', linewidth=2, color='red')
    plt.fill_between(pred_df['时间'], pred_df['下限'], pred_df['上限'], alpha=0.2, color='red')
    plt.title(f'{target_name} 未来 {periods//24} 天预测 (线性回归)')
    plt.xlabel('时间')
    plt.ylabel(target_name)
    plt.legend()
    plt.grid(True)
    if save_img_path:
        plt.savefig(save_img_path, dpi=150, bbox_inches='tight')
        print(f'  已保存图片: {save_img_path}')
    plt.close()
    return pred_df

# ==================== 整体大棚预测 ====================
def run_aggregate():
    print("\n=== 整体大棚预测模式 ===")
    for target in ['温度(℃)', '湿度(%)']:
        print(f"  正在聚合 {target} 数据...")
        agg_df = aggregate_sensors(INPUT_DIR, target, freq='h')
        print(f"  聚合数据长度：{len(agg_df)}，时间范围：{agg_df['ds'].min()} 至 {agg_df['ds'].max()}")

        # 整体大棚图片保存到 OUTPUT_DIR 根目录
        img_name = f"整体大棚_{target.replace('(℃)', '').replace('(%)', '')}_预测_未来7天.png"
        img_path = os.path.join(OUTPUT_DIR, img_name)

        pred = predict_linear(agg_df, target_name=target, periods=7*24, save_img_path=img_path)

        # 整体大棚 CSV 保存到 OUTPUT_DIR 根目录
        csv_name = f"整体大棚_{target.replace('(℃)', '').replace('(%)', '')}_预测_7天.csv"
        csv_path = os.path.join(OUTPUT_DIR, csv_name)
        pred.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  已保存 CSV: {csv_path}")

# ==================== 单个传感器预测 ====================
def run_single():
    print("\n=== 单个传感器预测模式 ===")
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR, exist_ok=True)
        print(f"已创建输入目录 '{INPUT_DIR}'，请将 CSV 文件放入后再运行。")
        return

    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
    if not files:
        print(f"未在 '{INPUT_DIR}' 中找到任何CSV文件。当前工作目录: {os.getcwd()}")
        print("请确保 '温湿度数据csv' 文件夹存在且包含 CSV 文件。")
        return

    print(f"找到 {len(files)} 个CSV文件: {files[:5]}{'...' if len(files)>5 else ''}")

    for filename in files:
        filepath = os.path.join(INPUT_DIR, filename)
        base_name = os.path.splitext(filename)[0]
        print(f"\n处理文件: {base_name}")

        df = pd.read_csv(filepath, parse_dates=['日期'], encoding='utf-8-sig')
        for target in ['温度(℃)', '湿度(%)']:
            if target not in df.columns:
                print(f"  警告: 文件缺少列 {target}，跳过")
                continue

            data = df[['日期', target]].rename(columns={'日期': 'ds', target: 'y'})
            data = data.set_index('ds').resample('h').mean().reset_index()
            data['y'] = data['y'].ffill()

            # 图片和 CSV 都保存到 OUTPUT_DIR/figures 下
            img_name = f"{base_name}_{target.replace('(℃)', '').replace('(%)', '')}_预测_未来7天.png"
            img_path = os.path.join(OUTPUT_DIR, "figures", img_name)

            pred = predict_linear(data, target_name=target, periods=7*24, save_img_path=img_path)

            csv_name = f"{base_name}_{target.replace('(℃)', '').replace('(%)', '')}_预测_7天.csv"
            csv_path = os.path.join(OUTPUT_DIR, "figures", csv_name)
            pred.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"  已保存 CSV: {csv_path}")

# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser(description='温室温湿度未来预测（线性回归）')
    parser.add_argument('--mode', type=str, default='all', choices=['aggregate', 'single', 'all'],
                        help='运行模式：aggregate=整体大棚预测，single=单个传感器预测，all=两者都运行（默认）')
    args = parser.parse_args()

    if args.mode in ['aggregate', 'all']:
        run_aggregate()
    if args.mode in ['single', 'all']:
        run_single()

    print(f"\n所有预测完成！结果保存在：")
    print(f"  整体大棚 CSV 和图片：{OUTPUT_DIR}")
    print(f"  单个传感器 CSV 和图片：{OUTPUT_DIR}/figures/")

if __name__ == "__main__":
    main()