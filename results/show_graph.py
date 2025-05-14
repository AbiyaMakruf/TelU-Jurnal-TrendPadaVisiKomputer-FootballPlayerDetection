import pandas as pd
import matplotlib.pyplot as plt

# Data dictionary
data = {
    'yolo11m': {
        'Class': ['all', 'player', 'goalkeeper', 'referee', 'ball'],
        'Precision': [0.821, 0.925, 0.97, 0.83, 0.558],
        'Recall': [0.743, 0.962, 0.933, 0.941, 0.137],
        'mAP50': [0.762, 0.971, 0.976, 0.944, 0.156],
        'mAP50-95': [0.452, 0.631, 0.553, 0.557, 0.0486]
    },
    'yolo11s': {
        'Class': ['all', 'player', 'goalkeeper', 'referee', 'ball'],
        'Precision': [0.783, 0.908, 0.945, 0.791, 0.486],
        'Recall': [0.735, 0.958, 0.954, 0.935, 0.0941],
        'mAP50': [0.748, 0.968, 0.974, 0.942, 0.107],
        'mAP50-95': [0.439, 0.614, 0.53, 0.576, 0.0337]
    },
    'yolo11n': {
        'Class': ['all', 'player', 'goalkeeper', 'referee', 'ball'],
        'Precision': [0.793, 0.903, 0.889, 0.795, 0.585],
        'Recall': [0.716, 0.951, 0.916, 0.927, 0.0706],
        'mAP50': [0.737, 0.963, 0.962, 0.924, 0.101],
        'mAP50-95': [0.419, 0.595, 0.511, 0.54, 0.0302]
    },
    'yolo11m-LCA': {
        'Class': ['all', 'player', 'goalkeeper', 'referee', 'ball'],
        'Precision': [0.846, 0.936, 0.977, 0.895, 0.577],
        'Recall': [0.778, 0.969, 0.989, 0.985, 0.169],
        'mAP50': [0.794, 0.975, 0.994, 0.988, 0.22],
        'mAP50-95': [0.482, 0.645, 0.598, 0.61, 0.0773]
    },
    'yolo11s-LCA': {
        'Class': ['all', 'player', 'goalkeeper', 'referee', 'ball'],
        'Precision': [0.848, 0.927, 0.984, 0.869, 0.613],
        'Recall': [0.77, 0.968, 0.987, 0.974, 0.151],
        'mAP50': [0.789, 0.974, 0.994, 0.928, 0.205],
        'mAP50-95': [0.472, 0.628, 0.601, 0.595, 0.0638]
    },
    'yolo11n-LCA': {
        'Class': ['all', 'player', 'goalkeeper', 'referee', 'ball'],
        'Precision': [0.824, 0.888, 0.967, 0.85, 0.59],
        'Recall': [0.761, 0.962, 0.984, 0.958, 0.139],
        'mAP50': [0.774, 0.97, 0.989, 0.963, 0.174],
        'mAP50-95': [0.448, 0.618, 0.535, 0.581, 0.0577]
    }
}

# Convert to DataFrames
dfs = {key: pd.DataFrame(value) for key, value in data.items()}

# Metrics to plot
metrics = ['Precision', 'Recall', 'mAP50', 'mAP50-95']

# Create a plot for each metric
for metric in metrics:
    fig, ax = plt.subplots(figsize=(12, 6))

    for model, df in dfs.items():
        linestyle = '--' if 'LCA' in model else '-'
        ax.plot(df['Class'], df[metric], marker='o', linestyle=linestyle, label=model)

    ax.set_title(f'{metric} per Class for Each YOLOv11 Variant')
    ax.set_xlabel('Class')
    ax.set_ylabel(metric)
    ax.legend(loc='lower left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
