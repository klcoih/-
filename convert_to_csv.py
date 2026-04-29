"""
xlsx 文件转换模块
处理宣威市气象数据 Excel 文件，转换为统一 CSV 格式
支持预测字段：温度、湿度、风速、降水量
"""
import os
import csv
import pandas as pd

# xlsx 列名映射到统一列名
COLUMN_MAPPING = {
    '收集时间': '日期',
    '大气温度': '温度',
    '大气湿度': '湿度',
    '风速': '风速',
    '降雨量': '降水量',
}

# 需要保留的列（统一格式）
REQUIRED_COLUMNS = ['日期', '温度', '湿度', '风速', '降水量']


def parse_xlsx_data(filepath):
    """
    解析单个 xlsx 文件，返回清洗后的数据 DataFrame
    支持的预测类型：温度、湿度、风速、降水量
    """
    try:
        df = pd.read_excel(filepath, sheet_name=0)
    except Exception as e:
        raise ValueError(f"无法读取 Excel 文件: {e}")

    # 检查必要的列是否存在
    source_cols = {col.strip(): col for col in df.columns}
    
    # 查找并重命名列
    rename_map = {}
    for target_col, source_col_pattern in COLUMN_MAPPING.items():
        found = False
        for col in df.columns:
            if target_col in col or source_col_pattern in col:
                rename_map[col] = target_col
                found = True
                break
        if not found and target_col != '日期':
            raise ValueError(f"缺少必要列: {target_col}")

    if rename_map:
        df = df.rename(columns=rename_map)

    # 确保日期列存在并转换格式
    date_col = None
    for col in ['收集时间', '日期', '时间']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is None:
        raise ValueError("未找到日期列（收集时间/日期/时间）")
    
    df['日期'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=['日期'])

    # 确保数值列为数值类型
    for col in ['温度', '湿度', '风速', '降水量']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 选择并补齐必要列
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    result_df = df[REQUIRED_COLUMNS].copy()
    
    # 数据清洗
    result_df = _clean_data(result_df)
    
    if len(result_df) == 0:
        raise ValueError("过滤后无有效数据")

    return result_df


def _clean_data(df):
    """
    数据清洗规则
    """
    n = len(df)
    to_delete = [False] * n

    # 规则①：温度==0 且 湿度==0 且 风速==0 且 降水量==0
    for i in range(n):
        temp = df.iloc[i]['温度']
        hum = df.iloc[i]['湿度']
        wind = df.iloc[i]['风速']
        rain = df.iloc[i]['降水量']
        
        is_all_zero = (
            (pd.notna(temp) and temp == 0) or temp is None
        ) and (
            (pd.notna(hum) and hum == 0) or hum is None
        ) and (
            (pd.notna(wind) and wind == 0) or wind is None
        ) and (
            (pd.notna(rain) and rain == 0) or rain is None
        )
        if is_all_zero:
            # 只有当所有值都为0时才删除
            if (pd.notna(temp) and temp == 0) and (pd.notna(hum) and hum == 0):
                to_delete[i] = True

    # 规则②：温度 > 50（异常高温）
    for i in range(n):
        temp = df.iloc[i]['温度']
        if pd.notna(temp) and temp > 50:
            to_delete[i] = True

    # 规则③：湿度 < 1（异常低湿度）
    for i in range(n):
        hum = df.iloc[i]['湿度']
        if pd.notna(hum) and hum < 1:
            to_delete[i] = True

    # 规则④：降雨量 < 0（异常值）
    for i in range(n):
        rain = df.iloc[i]['降水量']
        if pd.notna(rain) and rain < 0:
            to_delete[i] = True

    # 规则⑤：风速 < 0（异常值）
    for i in range(n):
        wind = df.iloc[i]['风速']
        if pd.notna(wind) and wind < 0:
            to_delete[i] = True

    # 规则⑥：连续3行温度均为0
    temp_zero_run = 0
    for i in range(n):
        temp = df.iloc[i]['温度']
        if pd.notna(temp) and temp == 0:
            temp_zero_run += 1
        else:
            if temp_zero_run >= 3:
                for j in range(i - temp_zero_run, i):
                    to_delete[j] = True
            temp_zero_run = 0
    if temp_zero_run >= 3:
        for j in range(n - temp_zero_run, n):
            to_delete[j] = True

    # 规则⑦：连续3行湿度均为0
    hum_zero_run = 0
    for i in range(n):
        hum = df.iloc[i]['湿度']
        if pd.notna(hum) and hum == 0:
            hum_zero_run += 1
        else:
            if hum_zero_run >= 3:
                for j in range(i - hum_zero_run, i):
                    to_delete[j] = True
            hum_zero_run = 0
    if hum_zero_run >= 3:
        for j in range(n - hum_zero_run, n):
            to_delete[j] = True

    # 保留有效行
    result_df = df[~pd.Series(to_delete)].copy()
    
    # 清理后的数据再处理缺失值
    result_df = result_df.ffill()  # 前向填充
    result_df = result_df.bfill()  # 后向填充
    
    return result_df


def process_single_xlsx_to_csv(xlsx_path, output_dir=None, output_filename=None):
    """
    处理单个 xlsx 文件，返回生成的 csv 文件路径
    xlsx_path: 输入 xlsx 文件路径
    output_dir: 输出目录，若为 None 则自动创建 '温湿度数据csv' 文件夹
    output_filename: 自定义输出文件名（不含扩展名），若为 None 则使用原文件名
    """
    if output_dir is None:
        output_dir = "温湿度数据csv"
    os.makedirs(output_dir, exist_ok=True)

    # 解析 xlsx 数据
    df = parse_xlsx_data(xlsx_path)
    
    # 生成输出文件名
    if output_filename is None:
        base_name = os.path.splitext(os.path.basename(xlsx_path))[0]
    else:
        base_name = output_filename
    csv_filename = base_name + ".csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    # 写入 CSV
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    return csv_path


def convert_xlsx_to_csv(input_dir="uploads", output_dir="温湿度数据csv"):
    """
    批量转换 input_dir 中的所有 xlsx 文件到 output_dir
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith((".xlsx", ".xls")):
            continue
        filepath = os.path.join(input_dir, filename)
        print(f"\n正在处理: {filepath}")
        try:
            csv_path = process_single_xlsx_to_csv(filepath, output_dir)
            print(f"  已生成: {csv_path}")
        except Exception as e:
            print(f"  处理失败: {e}")


# 保持向后兼容的函数名
def process_single_txt_to_csv(txt_path, output_dir=None, output_filename=None):
    """向后兼容：使用新的 xlsx 处理逻辑"""
    return process_single_xlsx_to_csv(txt_path, output_dir, output_filename)


def parse_txt_data(filepath):
    """向后兼容：使用新的 xlsx 解析逻辑"""
    df = parse_xlsx_data(filepath)
    col_names = list(df.columns)
    rows = [tuple(row) for row in df.values]
    return col_names, rows


if __name__ == "__main__":
    # 命令行批量转换模式
    convert_xlsx_to_csv()
