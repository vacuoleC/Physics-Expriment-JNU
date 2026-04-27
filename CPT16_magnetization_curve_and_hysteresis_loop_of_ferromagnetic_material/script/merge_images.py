"""
铁磁材料实验曲线合并脚本（双Y轴）

将 B-H 磁化曲线和 μ-H 磁导率曲线合并到一张图中
使用双Y轴，左右各一个Y轴，共享X轴(H)
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 实验数据
U = np.array([0.5, 1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0])
U1 = np.array([0.1056, 0.2140, 0.2840, 0.4544, 0.6451, 0.8131, 0.9946, 1.251, 1.499, 1.650])
U2 = np.array([0.04677, 0.07958, 0.09347, 0.1169, 0.1332, 0.1419, 0.1498, 0.1575, 0.1640, 0.1670])

# 实验常数
N = 30
L = 0.060
R1 = 2.0
n = 150
S = 80e-6
C2 = 20e-6
R2 = 10e3

# 计算H, B, μ
H = (N / L) * (U1 / R1)
B = (C2 * R2 / (n * S)) * U2
mu = B / H

# 添加起点(H=0, μ=0)以显示完整的μ-H曲线趋势
H_plot = np.concatenate([[0], H])
B_plot = np.concatenate([[0], B])
mu_plot = np.concatenate([[0], mu])

# 创建图形
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制B-H曲线（左Y轴）
color1 = '#1f77b4'
ax1.set_xlabel('磁场强度 H (A/m)', fontsize=12)
ax1.set_ylabel('磁感应强度 B (T)', fontsize=12, color=color1)
line1, = ax1.plot(H_plot, B_plot, 'o-', linewidth=2, markersize=8, color=color1, label='B-H 磁化曲线')
ax1.tick_params(axis='y', labelcolor=color1)

# 创建右Y轴，绘制μ-H曲线
ax2 = ax1.twinx()
color2 = '#ff7f0e'
ax2.set_ylabel('磁导率 μ (H/m)', fontsize=12, color=color2)
line2, = ax2.plot(H_plot, mu_plot, 's-', linewidth=2, markersize=8, color=color2, label='μ-H 磁导率曲线')
ax2.tick_params(axis='y', labelcolor=color2)

# 标注饱和点（使用原始数据索引，在_plot数据上+1）
max_B_idx = np.argmax(B)
ax1.scatter(H_plot[max_B_idx+1], B[max_B_idx], color='red', s=100, zorder=5, marker='*')
ax1.annotate(f'Bs={B[max_B_idx]:.3f}T', 
             xy=(H_plot[max_B_idx+1], B[max_B_idx]),
             xytext=(H_plot[max_B_idx+1]*1.1, B[max_B_idx]*0.9),
             fontsize=10, color='red')

# 标注最大磁导率点
max_mu_idx = np.argmax(mu)
ax2.scatter(H_plot[max_mu_idx+1], mu[max_mu_idx], color='green', s=100, zorder=5, marker='*')
ax2.annotate(f'μmax={mu[max_mu_idx]:.2e}H/m', 
             xy=(H_plot[max_mu_idx+1], mu[max_mu_idx]),
             xytext=(H_plot[max_mu_idx+1]*1.1, mu[max_mu_idx]*0.6),
             fontsize=10, color='green')
# 添加图例
lines = [line1, line2]
labels = ['B-H 磁化曲线', 'μ-H 磁导率曲线']
ax1.legend(lines, labels, loc='center right', fontsize=10)

# 标题
plt.title('铁磁材料 B-H 磁化曲线与 μ-H 磁导率曲线', fontsize=14)

# 网格
ax1.grid(True, linestyle='--', alpha=0.7)

# 调整布局
fig.tight_layout()

# 保存
script_dir = r'd:\git_prog\Physics_Expriment_JNU\CPT16_magnetization_curve_and_hysteresis_loop_of_ferromagnetic_material\script'
output_path = rf'{script_dir}\output\combined_curves.png'
plt.savefig(output_path, dpi=300)
plt.close()

print(f"[OK] Saved: {output_path}")