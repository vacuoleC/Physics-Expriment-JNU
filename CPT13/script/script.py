# 导入所需库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ===================== 全局设置：解决中文/负号显示问题 =====================
plt.rcParams['font.sans-serif'] = ['SimHei']   # 中文黑体
plt.rcParams['axes.unicode_minus'] = False     # 正常显示负号

# ===================== 核心：os.path.join 拼接文件路径 =====================
# 自动拼接当前目录下的3个CSV文件路径（无需手动写路径，跨平台兼容）
base_path = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在目录
data_path = os.path.join(base_path, "data")  # 数据文件夹路径
pic_path = os.path.join(base_path, "pic")    # 图片输出文件夹路径

# 确保输出目录存在
os.makedirs(pic_path, exist_ok=True)

file_is = os.path.join(data_path, "vh-is.csv")    # VH-IS 数据路径
file_im = os.path.join(data_path, "vh-im.csv")    # VH-IM 数据路径
file_xb = os.path.join(data_path, "b-x.csv")       # B-x  数据路径

# ===================== 读取CSV数据 =====================
df_is = pd.read_csv(file_is)   # 读取VH-IS数据
df_im = pd.read_csv(file_im)   # 读取VH-IM数据
df_xb = pd.read_csv(file_xb)  # 读取B-x数据

# 提取数据列
IS = df_is["IS(mA)"]
VH_IS = df_is["VH(mV)"]

IM = df_im["IM(A)"]
VH_IM = df_im["VH(mV)"]

x = df_xb["x(cm)"]
B = df_xb["B(mT)"]

# ===================== 绘制子图1：VH - IS 关系图 =====================
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.scatter(IS, VH_IS, color='red', s=60, label='实验数据', zorder=3)

# 线性回归拟合
IS_arr = np.array(IS)
VH_IS_arr = np.array(VH_IS)
k1, b1 = np.polyfit(IS_arr, VH_IS_arr, 1)  # k为斜率，b为截距
VH_fit1 = k1 * IS_arr + b1
ax1.plot(IS_arr, VH_fit1, color='blue', linewidth=2, label=f'线性拟合: y={k1:.4f}x+{b1:.4f}')

ax1.set_title(r'霍尔电压 - 工作电流特性 ($V_H-I_S$)', fontsize=14, pad=10)
ax1.set_xlabel('工作电流 $I_S$ (mA)', fontsize=12)
ax1.set_ylabel('霍尔电压 $V_H$ (mV)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()
plt.tight_layout()
plt.savefig(os.path.join(pic_path, 'VH-IS关系图.png'), dpi=300, bbox_inches='tight')
plt.close()

# ===================== 绘制子图2：VH - IM 关系图 =====================
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.scatter(IM, VH_IM, color='green', s=60, label='实验数据', zorder=3)

# 线性回归拟合
IM_arr = np.array(IM)
VH_IM_arr = np.array(VH_IM)
k2, b2 = np.polyfit(IM_arr, VH_IM_arr, 1)  # k为斜率，b为截距
VH_fit2 = k2 * IM_arr + b2
ax2.plot(IM_arr, VH_fit2, color='orange', linewidth=2, label=f'线性拟合: y={k2:.4f}x+{b2:.4f}')

ax2.set_title(r'霍尔电压 - 励磁电流特性 ($V_H-I_M$)', fontsize=14, pad=10)
ax2.set_xlabel('励磁电流 $I_M$ (A)', fontsize=12)
ax2.set_ylabel('霍尔电压 $V_H$ (mV)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(pic_path, 'VH-IM关系图.png'), dpi=300, bbox_inches='tight')
plt.close()

# ===================== 绘制子图3：B - x 磁场分布图 =====================
# 建立以 x=11 为原点的坐标系
x_center = 11  # 磁场中心位置
x_new = np.array(x) - x_center  # 新坐标系
B_arr = np.array(B)

fig3, ax3 = plt.subplots(figsize=(8, 6))

# 绘制散点数据
ax3.scatter(x_new, B_arr, color='black', s=50, zorder=3, label='实验数据')

# 用折线连接数据点
ax3.plot(x_new, B_arr, color='purple', linewidth=1.5, alpha=0.7)

ax3.set_title(r'螺线管轴向磁场分布 ($B-x$)', fontsize=14, pad=10)
ax3.set_xlabel('轴向位置 $x$ (cm)，原点为 x=11cm', fontsize=12)
ax3.set_ylabel('磁感应强度 $B$ (mT)', fontsize=12)
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.legend()
ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)  # 添加原点参考线
plt.tight_layout()
plt.savefig(os.path.join(pic_path, 'B-x磁场分布图.png'), dpi=300, bbox_inches='tight')
plt.close()

print("三张图片已分别保存到 pic 文件夹中")
