import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

# 定义颜色加深函数
def darken_color(color, factor=0.3):
    rgb = to_rgb(color)
    return tuple(max(0, c - factor) for c in rgb)


def main():
    # 读取 CSV 文件 - 修改分隔符为制表符并正确提取数值列
    file_path = 'res_share_bound.csv'
    df = pd.read_csv(file_path, sep=',')  # 指定制表符分隔

    # 提取所有数值类型的列名并按数字排序（排除第一列标签列）
    METHODS = []
    for col in df.columns[1:]:  # 跳过第一列标签列
        try:
            # 尝试将列名转换为整数，筛选数值型列
            int(col)
            METHODS.append(col)
        except ValueError:
            continue
    METHODS.sort(key=lambda x: int(x))  # 按数值排序

    # 添加错误处理：确保至少有一个方法列
    if not METHODS:
        raise ValueError("未找到有效的方法列，请检查CSV格式")

    # 确保使用第一列作为标签，并删除包含 NaN 的行
    df = df.dropna(subset=[df.columns[0]])
    x_labels = df[df.columns[0]].copy()

    # 定义目标方法，将其颜色设置为lightcoral
    TARGET_METHOD = '24'  # 用户可修改此处为目标方法名

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
    colors = ['lightcoral', 'linen', 'bisque', 'lavender', 'aliceblue', 'gainsboro', '#D2B48C']

    # 绘制柱状图，添加黑色边框
    x = range(len(x_labels))
    width = 0.8 / len(METHODS)
    # 为非目标方法准备颜色列表（排除lightcoral）
    other_colors = [c for c in colors if c != 'lightcoral']
    for i, method in enumerate(METHODS):
        if method in df.columns:
            valid_mask = df[method] > 0
            x_values = [pos + i * width for pos in x if valid_mask.iloc[pos]]
            y_values = df.loc[valid_mask, method]
            # 根据方法名确定颜色
            if method == TARGET_METHOD:
                color = 'lightcoral'
            else:
                color = other_colors[i % len(other_colors)]
            plt.bar(x_values, y_values, width=width, label=f'τ={method}', color=color, edgecolor='black')

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

    plt.legend(fontsize=12, ncol=2)
    plt.tight_layout()
    
    import os
    script_name = os.path.splitext(os.path.basename(__file__))[0]

    # 保存为 PDF 文件
    plt.savefig(f'{script_name}.pdf')
    plt.close()

if __name__ == '__main__':
    main()