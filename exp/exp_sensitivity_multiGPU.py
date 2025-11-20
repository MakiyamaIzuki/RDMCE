import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

# 定义颜色加深函数
def darken_color(color, factor=0.3):
    rgb = to_rgb(color)
    return tuple(max(0, c - factor) for c in rgb)

# 手动指定方法数组，可根据实际情况修改
METHODS = ['RDMCE-4090', 'RDMCE-A100', 'RDMCE-H800']  # 请替换为实际方法名


def main():
    # 读取 CSV 文件
    file_path = 'res_overall.csv'
    df = pd.read_csv(file_path)

    # 确保使用 abbr 列作为标签，并删除包含 NaN 的行
    if 'abbr' in df.columns:
        # 删除 abbr 列中包含 NaN 的行
        df = df.dropna(subset=['abbr'])
        x_labels = df['abbr'].copy()  # 使用 copy 确保 x_labels 不被后续操作影响
    else:
        df = df.dropna(subset=[df.columns[0]])
        x_labels = df.iloc[:, 0].copy()  # 使用 copy 确保 x_labels 不被后续操作影响

    # 将非数字值转换为 NaN
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    # 设置图片尺寸，降低高度，增加宽度以适配 figure*
    plt.figure(figsize=(6, 3))  # 高度降低到 4

    # 设置基础字体大小
    plt.rcParams.update({'font.size': 14})

    # 定义颜色列表
    colors = ['lightcoral', 'darkseagreen', 'wheat', 'thistle', 'lightblue', 'lightgrey', '#8B4513']

    # 绘制柱状图，添加黑色边框
    x = range(len(x_labels))
    width = 0.8 / len(METHODS)
    for i, method in enumerate(METHODS):
        if method in df.columns:
            valid_mask = df[method] > 0
            x_values = [pos + i * width for pos in x if valid_mask.iloc[pos]]
            y_values = df.loc[valid_mask, method]
            plt.bar(x_values, y_values, width=width, label=method, color=colors[i % len(colors)], edgecolor='black')

    # 标记缺失数据，显示对应原因并使用加深后的颜色
    for i, method in enumerate(METHODS):
        if method in df.columns:
            invalid_mask = df[method].isna() | (df[method] <= 0)
            if invalid_mask.any(): 
                x_values = [pos + i * width for pos in x if invalid_mask.iloc[pos]]
                for idx, x_val in enumerate(x_values):
                    if method.startswith('mce-gpu'):
                        reason = 'MOB'
                    elif method == 'G2-AIMD':
                        reason = 'MLE'
                    else:
                        reason = 'Not run'
                    plt.text(
                        x_val, 0.7, reason, 
                        ha='center', va='bottom', 
                        color=darken_color(colors[i % len(colors)]), 
                        fontsize=10, rotation=90, 
                        fontweight='bold',
                        family='Arial'
                    )

    # 设置 y 轴为对数坐标
    plt.yscale('log')
    plt.ylabel('Running time(s)')
    # 移除横轴名称
    # plt.xlabel('Test Cases')

    # 确保 x 轴标签能正确显示
    if len(x_labels) > 0:
        plt.xticks([pos + width * (len(METHODS) - 1) / 2 for pos in x], x_labels)

    plt.legend(fontsize=12)
    plt.tight_layout()
    
    import os
    script_name = os.path.splitext(os.path.basename(__file__))[0]

    # 保存为 PDF 文件
    plt.savefig(f'{script_name}.pdf')
    plt.close()

if __name__ == '__main__':
    main()