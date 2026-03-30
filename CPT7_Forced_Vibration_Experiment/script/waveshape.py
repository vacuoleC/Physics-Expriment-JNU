import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# 基础设置：中文字体、图像风格


# 1. 重新读取原始CSV文件，保留原始格式
df_raw = pd.read_csv('D:/git_prog/Physics_Expriment_JNU/CPT7_Forced_Vibration_Experiment/data/oringin_data/Free_Vibration/滤波数据.csv')
print(f"原始数据基本信息：")
print(f"- 数据形状：{df_raw.shape} (行数, 列数)")
print(f"- 列名：{df_raw.columns.tolist()}")
print(f"- 数据类型：\n{df_raw.dtypes}")
print(f"- 前10行原始数据：\n{df_raw.head(10)}")

# 2. 数据预处理：提取数值列（处理可能的列名异常）
# 检查列名是否为数值（可能是数据首行被当作列名）
try:
    # 尝试将列名转换为数值，判断是否为数据首行
    float(df_raw.columns[0])
    # 若列名是数值，重新构建数据（列名设为"信号值"，原列名作为第一行数据）
    first_row = pd.DataFrame({df_raw.columns[0]: [float(df_raw.columns[0])]})
    df_clean = pd.concat([first_row, df_raw], ignore_index=True)
    df_clean.columns = ['信号值']
except:
    # 若列名正常，直接使用
    df_clean = df_raw.rename(columns={df_raw.columns[0]: '信号值'})

# 取前20000个数据点（按之前需求）
data_20000 = df_clean['信号值'].iloc[:200000].values
print(f"\n前20000个数据点统计：")
print(f"- 数值范围：{data_20000.min():.3f} ~ {data_20000.max():.3f}")
print(f"- 均值：{data_20000.mean():.3f}")
print(f"- 标准差：{data_20000.std():.3f}")

# 3. 按1000Hz采样频率计算时间轴（用户指定采样率）
fs = 1000  # 采样频率（Hz）
time = np.arange(len(data_20000)) / fs  # 时间轴（单位：s）
print(f"\n时间轴信息：")
print(f"- 采样频率：{fs} Hz")
print(f"- 时间范围：0 ~ {time[-1]:.2f} s")
print(f"- 时间分辨率：{1/fs:.6f} s")

# 4. 绘制核心波形图（分3个尺度展示，确保细节清晰）
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 15))

# 子图1：完整前20000点波形（0~20s）
ax1.plot(time, data_20000, color='#2E86AB', linewidth=0.8, alpha=0.9, label=f'前20000个数据点（0~20s）')
ax1.axhline(y=data_20000.mean(), color='#A23B72', linestyle='--', linewidth=1.2, alpha=0.8, label=f'数据均值（{data_20000.mean():.3f}）')
ax1.set_title(f'滤波数据完整波形（采样频率{fs}Hz，前20000点）', fontsize=14, fontweight='bold', pad=20)
ax1.set_xlabel('时间（s）', fontsize=12)
ax1.set_ylabel('信号值', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax1.set_xlim(0, time[-1])

# 子图2：中期尺度波形（取中间5s：7.5~12.5s，展示完整周期细节）
mid_start = int(7.5 * fs)
mid_end = int(12.5 * fs)
ax2.plot(time[mid_start:mid_end], data_20000[mid_start:mid_end], color='#F18F01', linewidth=1.2, alpha=1.0, label=f'7.5~12.5s波形')
# 标注峰值（便于观察周期）
peaks_mid, _ = find_peaks(data_20000[mid_start:mid_end], distance=int(fs*0.5))  # 峰值间隔至少0.5s
ax2.scatter(time[mid_start:mid_end][peaks_mid], data_20000[mid_start:mid_end][peaks_mid], 
           color='#C73E1D', s=60, zorder=5, label='局部峰值')
ax2.set_title(f'滤波数据中期波形（7.5~12.5s，标注峰值）', fontsize=14, fontweight='bold', pad=20)
ax2.set_xlabel('时间（s）', fontsize=12)
ax2.set_ylabel('信号值', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax2.set_xlim(7.5, 12.5)

# 子图3：细节尺度波形（取1s片段：9~10s，展示振动细节）
detail_start = int(9 * fs)
detail_end = int(10 * fs)
ax3.plot(time[detail_start:detail_end], data_20000[detail_start:detail_end], color='#3F88C5', linewidth=1.5, alpha=1.0, label=f'9~10s细节波形')
ax3.axhline(y=0, color='black', linestyle=':', linewidth=1.0, alpha=0.7, label='零参考线')
ax3.set_title(f'滤波数据细节波形（9~10s，展示振动细节）', fontsize=14, fontweight='bold', pad=20)
ax3.set_xlabel('时间（s）', fontsize=12)
ax3.set_ylabel('信号值', fontsize=12)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax3.set_xlim(9, 10)

# 调整布局，保存图像
plt.tight_layout(pad=3.0)
plt.savefig('D:/git_prog/Physics_Expriment_JNU/CPT7_Forced_Vibration_Experiment', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"\n✅ 波形图已生成完成！")
print(f"- 包含3个尺度：完整20s波形、中期5s波形（标注峰值）、细节1s波形")
print(f"- 严格按1000Hz采样频率绘制时间轴")
print(f"- 图像已保存为：filter_data_waveforms_20000points.png")