import pandas as pd
import numpy as np

# 读取 tsf 文件
tsf_file = 'data/SOLAR/solar_10_minutes_dataset.tsf'
series_data = []

print("开始读取文件...")
with open(tsf_file, 'r') as f:
    line_count = 0
    for line in f:
        line = line.strip()
        line_count += 1
        if line_count <= 5:
            print(f"第{line_count}行: {line}")
        
        if not line:
            continue
        if line.startswith('T'):
            # 拆分出 ID、时间戳 和 数据部分
            try:
                first_colon = line.index(':')
                second_colon = line.index(':', first_colon + 1)
                values_str = line[second_colon + 1:]
                values = values_str.strip().split(',')
                numeric_values = [float(x) if x != '?' else np.nan for x in values]
                if numeric_values:
                    series_data.append(numeric_values)
                    if len(series_data) <= 3:
                        print(f"处理后的数据 {len(series_data)}: {numeric_values[:5]}...")
            except Exception as e:
                print(f"解析出错，跳过该行: {e}")
                continue

print(f"总共读取了 {line_count} 行")
print(f"成功处理了 {len(series_data)} 个数据序列")

# 构建DataFrame
if series_data:
    df = pd.DataFrame(series_data)
    df = df.transpose()
    df.columns = [f'T{i+1}' for i in range(len(series_data))]
    
    # 添加数据统计信息
    print("\n=== 数据统计信息 ===")
    print(f"数据形状: {df.shape} (行数: 时间点数, 列数: 变量数)")
    
    # 计算每个时间序列的统计信息
    stats = pd.DataFrame({
        'mean': df.mean(),
        'std': df.std(),
        'min': df.min(),
        'max': df.max(),
        'median': df.median(),
        'missing_values': df.isna().sum()
    })
    
    print("\n=== 每个时间序列的统计信息 ===")
    print(stats)
    
    # 计算全局统计信息
    print("\n=== 全局统计信息 ===")
    print(f"所有数据的均值: {df.values.mean():.4f}")
    print(f"所有数据的标准差: {df.values.std():.4f}")
    print(f"所有数据的最小值: {df.values.min():.4f}")
    print(f"所有数据的最大值: {df.values.max():.4f}")
    print(f"所有数据的中位数: {np.median(df.values):.4f}")
    print(f"缺失值总数: {df.isna().sum().sum()}")
    
    # 保存统计信息到文件
    stats.to_csv('data/SOLAR/solar_data_statistics.csv')
    print("\n统计信息已保存到 'solar_data_statistics.csv'")
    
    # 保存原始数据
    df.to_csv('data/SOLAR/solar_10_minutes_dataset.csv', index=False)
    print("原始数据已保存为 'solar_10_minutes_dataset.csv'")
    
else:
    print("错误：文件中没有找到有效数据。")
