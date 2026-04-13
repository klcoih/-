import os
import csv
import re

def detect_encoding(filepath):
    """检测文件编码，支持 utf-8, utf-8-sig, gbk, gb2312"""
    for enc in ['utf-8', 'utf-8-sig', 'gbk', 'gb2312']:
        try:
            with open(filepath, 'r', encoding=enc) as f:
                f.read()
            return enc
        except UnicodeDecodeError:
            continue
    return None

def parse_txt_data(filepath):
    """
    解析单个 txt 文件，返回清洗后的数据行列表和列名
    返回: (col_names, kept_rows)
        col_names: list, 列名
        kept_rows: list of tuple (序号, 温度, 湿度, 日期)
    """
    enc = detect_encoding(filepath)
    if enc is None:
        raise ValueError(f"无法识别文件编码: {filepath}")

    with open(filepath, 'r', encoding=enc) as f:
        lines = f.readlines()

    # 定位表头
    header_idx = None
    for i, line in enumerate(lines):
        if all(k in line for k in ['序号', '温度', '湿度', '日期']):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("未找到数据表头")

    # 提取列名
    header_line = lines[header_idx].strip()
    parts = re.split(r'\t+', header_line)
    non_empty = [p.strip() for p in parts if p.strip()]
    if len(non_empty) >= 4:
        col_names = non_empty[:4]
    else:
        col_names = ["序号", "温度(℃)", "湿度(%)", "日期"]

    # 解析数据行
    data_rows = []
    for idx_line, line in enumerate(lines[header_idx+1:], start=header_idx+1):
        line = line.strip()
        if not line:
            continue
        # 按一个或多个制表符分割
        parts = re.split(r'\t+', line)
        if len(parts) < 4:
            # 尝试按任意空白分割
            parts = re.split(r'\s+', line)
        if len(parts) < 4:
            continue
        idx_str = parts[0].strip()
        temp_str = parts[1].strip()
        hum_str = parts[2].strip()
        dt_str = parts[3].strip()
        # 转换数值
        temp = None
        hum = None
        try:
            temp = float(temp_str)
        except:
            pass
        try:
            hum = float(hum_str)
        except:
            pass
        if idx_str and dt_str:
            data_rows.append((idx_str, temp, hum, dt_str))

    if not data_rows:
        raise ValueError("未找到任何有效数据行")

    n = len(data_rows)
    to_delete = [False] * n

    # 规则①：温度==0且湿度==0
    for i in range(n):
        temp, hum = data_rows[i][1], data_rows[i][2]
        if temp is not None and hum is not None and temp == 0.0 and hum == 0.0:
            to_delete[i] = True

    # 规则③：温度 > 50
    for i in range(n):
        temp = data_rows[i][1]
        if temp is not None and temp > 50.0:
            to_delete[i] = True

    # 规则④：湿度 < 1
    for i in range(n):
        hum = data_rows[i][2]
        if hum is not None and hum < 1.0:
            to_delete[i] = True

    # 规则②：连续三行温度均为0
    temp_zero_run = 0
    for i in range(n):
        temp = data_rows[i][1]
        if temp is not None and temp == 0.0:
            temp_zero_run += 1
        else:
            if temp_zero_run >= 3:
                for j in range(i - temp_zero_run, i):
                    to_delete[j] = True
            temp_zero_run = 0
    if temp_zero_run >= 3:
        for j in range(n - temp_zero_run, n):
            to_delete[j] = True

    # 规则②：连续三行湿度均为0
    hum_zero_run = 0
    for i in range(n):
        hum = data_rows[i][2]
        if hum is not None and hum == 0.0:
            hum_zero_run += 1
        else:
            if hum_zero_run >= 3:
                for j in range(i - hum_zero_run, i):
                    to_delete[j] = True
            hum_zero_run = 0
    if hum_zero_run >= 3:
        for j in range(n - hum_zero_run, n):
            to_delete[j] = True

    # 收集保留的行（丢弃数值无效的行）
    kept_rows = []
    for i in range(n):
        if to_delete[i]:
            continue
        idx, temp, hum, dt = data_rows[i]
        if temp is None or hum is None:
            continue
        kept_rows.append((idx, temp, hum, dt))

    if not kept_rows:
        raise ValueError("过滤后无有效数据")

    return col_names, kept_rows

def write_csv(output_path, col_names, rows):
    """将数据写入 CSV 文件"""
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(col_names)
        writer.writerows(rows)

def process_single_txt_to_csv(txt_path, output_dir=None, output_filename=None):
    """
    处理单个 txt 文件，返回生成的 csv 文件路径
    txt_path: 输入 txt 文件路径
    output_dir: 输出目录，若为 None 则自动创建 '温湿度数据csv' 文件夹
    output_filename: 自定义输出文件名（不含扩展名），若为 None 则使用原文件名（保留中文）
    """
    if output_dir is None:
        output_dir = "温湿度数据csv"
    os.makedirs(output_dir, exist_ok=True)

    col_names, kept_rows = parse_txt_data(txt_path)
    if output_filename is None:
        base_name = os.path.splitext(os.path.basename(txt_path))[0]
    else:
        base_name = output_filename
    csv_filename = base_name + ".csv"
    csv_path = os.path.join(output_dir, csv_filename)
    write_csv(csv_path, col_names, kept_rows)
    return csv_path

def convert_txt_to_csv(input_dir="温湿度数据txt", output_dir="温湿度数据csv"):
    """批量转换 input_dir 中的所有 txt 文件到 output_dir"""
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(".txt"):
            continue
        filepath = os.path.join(input_dir, filename)
        print(f"\n正在处理: {filepath}")
        try:
            csv_path = process_single_txt_to_csv(filepath, output_dir)
            print(f"  已生成: {csv_path}")
        except Exception as e:
            print(f"  处理失败: {e}")

if __name__ == "__main__":
    # 命令行批量转换模式
    convert_txt_to_csv()