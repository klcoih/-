import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

# 设置 matplotlib 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 参数配置 ====================
INPUT_DIR = "温湿度数据csv"
OUTPUT_DIR = "预测结果"                     # 所有结果存放根目录
PREDICT_DAYS = 30                          # 预测天数
PREDICT_PERIODS = PREDICT_DAYS * 24         # 小时数

# 创建必要的目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)

# ==================== 聚合函数 ====================
def aggregate_sensors(input_dir, target_col, freq='h'):
    """读取所有CSV，重采样到统一频率，计算每个时间点的平均值，返回 DataFrame (ds, y)"""
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

# ==================== Prophet 预测函数 ====================
def predict_prophet(df, target_name, periods=PREDICT_PERIODS, freq='h', save_img_path=None):
    """
    df: 包含 ds 和 y 列的 DataFrame
    target_name: 用于图片标题和文件名
    periods: 预测步数
    freq: 频率
    save_img_path: 图片保存路径，如果为 None 则不保存
    返回预测结果 DataFrame
    """
    # 确保数据按时间排序
    df = df.sort_values('ds').reset_index(drop=True)

    # 创建 Prophet 模型
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    model.fit(df)

    # 生成未来日期
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)

    # 绘图
    fig = model.plot(forecast)
    plt.title(f'{target_name} 预测 (未来 {periods//24} 天)')
    plt.xlabel('时间')
    plt.ylabel(target_name)
    if save_img_path:
        plt.savefig(save_img_path, dpi=150, bbox_inches='tight')
        print(f'  已保存图片: {save_img_path}')
    plt.close(fig)

    # 提取预测结果（最后 periods 行）
    pred = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
    pred.columns = ['时间', '预测值', '下限', '上限']
    return pred

# ==================== 整体大棚预测 ====================
def run_aggregate():
    print("\n=== 整体大棚预测模式 ===")
    for target in ['温度(℃)', '湿度(%)']:
        print(f"  正在聚合 {target} 数据...")
        agg_df = aggregate_sensors(INPUT_DIR, target, freq='h')
        print(f"  聚合数据长度：{len(agg_df)}，时间范围：{agg_df['ds'].min()} 至 {agg_df['ds'].max()}")

        # 整体大棚图片和 CSV 都保存到 OUTPUT_DIR 根目录
        img_name = f"整体大棚_{target.replace('(℃)', '').replace('(%)', '')}_预测_{PREDICT_DAYS}天.png"
        img_path = os.path.join(OUTPUT_DIR, img_name)

        pred = predict_prophet(agg_df, target_name=target, periods=PREDICT_PERIODS, freq='h', save_img_path=img_path)

        csv_name = f"整体大棚_{target.replace('(℃)', '').replace('(%)', '')}_预测_{PREDICT_DAYS}天.csv"
        csv_path = os.path.join(OUTPUT_DIR, csv_name)
        pred.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  已保存 CSV: {csv_path}")

# ==================== 单个传感器预测 ====================
def run_single():
    print("\n=== 单个传感器预测模式 ===")
    files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
    if not files:
        print(f"未在 {INPUT_DIR} 中找到任何CSV文件")
        return

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
            img_name = f"{base_name}_{target.replace('(℃)', '').replace('(%)', '')}_预测_{PREDICT_DAYS}天.png"
            img_path = os.path.join(OUTPUT_DIR, "figures", img_name)

            pred = predict_prophet(data, target_name=target, periods=PREDICT_PERIODS, freq='h', save_img_path=img_path)

            csv_name = f"{base_name}_{target.replace('(℃)', '').replace('(%)', '')}_预测_{PREDICT_DAYS}天.csv"
            csv_path = os.path.join(OUTPUT_DIR, "figures", csv_name)
            pred.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"  已保存 CSV: {csv_path}")

# ==================== 主程序 ====================
def main():
    parser = argparse.ArgumentParser(description='温室温湿度未来预测（Prophet）')
    parser.add_argument('--mode', type=str, default='all', choices=['aggregate', 'single', 'all'],
                        help='运行模式：aggregate=整体大棚预测，single=单个传感器预测，all=两者都运行（默认）')
    args = parser.parse_args()

    if args.mode in ['aggregate', 'all']:
        run_aggregate()
    if args.mode in ['single', 'all']:
        run_single()

    print(f"\n所有预测完成！结果保存在：{OUTPUT_DIR}（整体）和 {OUTPUT_DIR}/figures/（单个传感器）")

if __name__ == "__main__":
    main()