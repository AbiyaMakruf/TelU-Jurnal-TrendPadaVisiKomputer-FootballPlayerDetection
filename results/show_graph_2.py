import matplotlib.pyplot as plt

# Model parameter counts (actual values)
parameter_counts = {
    'yolo11n': 2582000,
    'yolo11s': 9413000,
    'yolo11m': 20030000,
    'yolo11n-LCA': 2585000,
    'yolo11s-LCA': 9424000,
    'yolo11m-LCA': 20052000,
}

# Data dictionary
data = {
    'yolo11m': {
        'Class': ['all', 'player', 'goalkeeper', 'referee', 'ball'],
        'mAP50': [0.762, 0.971, 0.976, 0.944, 0.156],
    },
    'yolo11s': {
        'Class': ['all', 'player', 'goalkeeper', 'referee', 'ball'],
        'mAP50': [0.748, 0.968, 0.974, 0.942, 0.107],
    },
    'yolo11n': {
        'Class': ['all', 'player', 'goalkeeper', 'referee', 'ball'],
        'mAP50': [0.737, 0.963, 0.962, 0.924, 0.101],
    },
    'yolo11m-LCA': {
        'Class': ['all', 'player', 'goalkeeper', 'referee', 'ball'],
        'mAP50': [0.794, 0.975, 0.994, 0.988, 0.22],
    },
    'yolo11s-LCA': {
        'Class': ['all', 'player', 'goalkeeper', 'referee', 'ball'],
        'mAP50': [0.789, 0.974, 0.994, 0.928, 0.205],
    },
    'yolo11n-LCA': {
        'Class': ['all', 'player', 'goalkeeper', 'referee', 'ball'],
        'mAP50': [0.774, 0.97, 0.989, 0.963, 0.174],
    }
}

# Model groups
original_models = ['yolo11n', 'yolo11s', 'yolo11m']
lca_models = ['yolo11n-LCA', 'yolo11s-LCA', 'yolo11m-LCA']
all_model_keys = original_models + lca_models

# Helper function to get mAP50 for specific class
def get_map50(model_key, target_class):
    idx = data[model_key]['Class'].index(target_class)
    return data[model_key]['mAP50'][idx]

# Extract data points
def extract_points(model_keys, target_class):
    # Mengonversi jumlah parameter ke ribuan
    x_vals = [parameter_counts[m] / 1000 for m in model_keys]
    y_vals = [get_map50(m, target_class) for m in model_keys]
    return x_vals, y_vals

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10)) # 2 baris, 1 kolom subplot

# --- Grafik untuk Kelas 'all' ---
target_class_all = 'all'
x_orig_all, y_orig_all = extract_points(original_models, target_class_all)
ax1.plot(x_orig_all, y_orig_all, 'o-', color='gray', label=f'Original - mAP50 ({target_class_all})')
for i, model_name in enumerate(original_models):
    ax1.text(x_orig_all[i], y_orig_all[i] + 0.005, model_name, fontsize=9, ha='center')

x_lca_all, y_lca_all = extract_points(lca_models, target_class_all)
ax1.plot(x_lca_all, y_lca_all, 'o--', color='tab:blue', label=f'LCA - mAP50 ({target_class_all})')
for i, model_name in enumerate(lca_models):
    ax1.text(x_lca_all[i], y_lca_all[i] + 0.005, model_name, fontsize=9, ha='center')

ax1.set_title('Perbandingan Kinerja Model YOLOv11 untuk Kelas "all"')
ax1.set_xlabel('Jumlah Parameter (ribuan)')
ax1.set_ylabel('mAP50')
ax1.grid(True)
ax1.legend()

# Menentukan rentang x untuk subplot 'all'
max_param_all = max(x_orig_all + x_lca_all)
ax1.set_xlim(0, max_param_all + max_param_all * 0.05) # Mulai dari 0 dan sedikit padding di kanan
min_map_all = min(y_orig_all + y_lca_all)
max_map_all = max(y_orig_all + y_lca_all)
ax1.set_ylim(min_map_all - 0.05, max_map_all + 0.05)


# --- Grafik untuk Kelas 'ball' ---
target_class_ball = 'ball'
x_orig_ball, y_orig_ball = extract_points(original_models, target_class_ball)
ax2.plot(x_orig_ball, y_orig_ball, 's-', color='gray', linestyle=':', label=f'Original - mAP50 ({target_class_ball})')
for i, model_name in enumerate(original_models):
    ax2.text(x_orig_ball[i], y_orig_ball[i] + 0.003, model_name, fontsize=9, ha='center')

x_lca_ball, y_lca_ball = extract_points(lca_models, target_class_ball)
ax2.plot(x_lca_ball, y_lca_ball, 's--', color='tab:blue', linestyle=':', label=f'LCA - mAP50 ({target_class_ball})')
for i, model_name in enumerate(lca_models):
    ax2.text(x_lca_ball[i], y_lca_ball[i] + 0.003, model_name, fontsize=9, ha='center')

ax2.set_title('Perbandingan Kinerja Model YOLOv11 untuk Kelas "ball"')
ax2.set_xlabel('Jumlah Parameter (ribuan)')
ax2.set_ylabel('mAP50')
ax2.grid(True)
ax2.legend()

# Menentukan rentang x untuk subplot 'ball'
max_param_ball = max(x_orig_ball + x_lca_ball)
ax2.set_xlim(0, max_param_ball + max_param_ball * 0.05) # Mulai dari 0 dan sedikit padding di kanan
min_map_ball = min(y_orig_ball + y_lca_ball)
max_map_ball = max(y_orig_ball + y_lca_ball)
ax2.set_ylim(min_map_ball - 0.02, max_map_ball + 0.02)

# Judul utama untuk figure
fig.suptitle('Perbandingan Kinerja Model YOLOv11 Original vs LCA\nterhadap Jumlah Parameter dan mAP50', fontsize=16)

# Menyesuaikan layout agar tidak tumpang tindih
plt.tight_layout(rect=[0, 0, 1, 0.96]) # rect untuk memberi ruang bagi suptitle
plt.show()