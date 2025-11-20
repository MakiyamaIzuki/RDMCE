import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import os

# Function to darken a color
def darken_color(color, factor=0.3):
    rgb = to_rgb(color)
    return tuple(max(0, c - factor) for c in rgb)

# Define methods array
METHODS = ['RDMCE', 'mce-gpu-P', 'mce-gpu-PX', 'G2-AIMD']

def main():
    # Disable LaTeX rendering to avoid font issues
    plt.rcParams.update({'text.usetex': False})
    # Use standard fonts
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans']})
    
    # Read CSV file
    file_path = 'res_overall.csv'
    df = pd.read_csv(file_path)

    # Ensure abbr column is used for labels and remove rows with NaN
    if 'abbr' in df.columns:
        df = df.dropna(subset=['abbr'])
        x_labels = df['abbr'].copy()
    else:
        df = df.dropna(subset=[df.columns[0]])
        x_labels = df.iloc[:, 0].copy()

    # Convert non-numeric values to NaN
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

    # Set figure size
    plt.figure(figsize=(12, 3))

    # Set base font size
    plt.rcParams.update({'font.size': 14})

    # Define color list
    colors = ['lightcoral', 'darkseagreen', 'wheat', 'thistle', 'lightblue', 'lightgrey', '#8B4513']

    # Plot bar chart with black borders
    x = range(len(x_labels))
    width = 0.8 / len(METHODS)
    for i, method in enumerate(METHODS):
        if method in df.columns:
            valid_mask = df[method] > 0
            x_values = [pos + i * width for pos in x if valid_mask.iloc[pos]]
            y_values = df.loc[valid_mask, method]
            if method == 'G2-AIMD':
                method_lb = 'G²-AIMD'  # Use Unicode superscript instead of LaTeX
            else:
                method_lb = method
            plt.bar(x_values, y_values, width=width, label=method_lb, color=colors[i % len(colors)], edgecolor='black')

    # Mark missing data with reasons
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
                        fontweight='bold'
                    )

    # Set y-axis to logarithmic scale
    plt.yscale('log')
    plt.ylabel('Running time(s)')

    # Ensure x-axis labels are displayed correctly
    if len(x_labels) > 0:
        plt.xticks([pos + width * (len(METHODS) - 1) / 2 for pos in x], x_labels)

    plt.legend(fontsize=12)
    plt.tight_layout()

    # Save as PDF file
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    plt.savefig(f'{script_name}.pdf')
    plt.close()

if __name__ == '__main__':
    main()