"""
铁磁材料磁化曲线与磁滞回线实验数据分析脚本

功能：
1. 基本磁化曲线分析（B-H曲线、μ-H曲线）
2. 磁滞回线分析（饱和磁滞回线、关键参数提取）
3. 结果输出（图表 + 数据表格）

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import os
import sys
import io

# 设置标准输出编码为UTF-8，解决Windows控制台中文乱码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============================================================
# 1. 数据输入类（配置类 + 示例数据）
# ============================================================

class ExperimentConfig:
    """实验常数配置类"""
    
    N = 30          # 励磁线圈匝数
    L = 0.060       # 样品平均磁路长度(m)
    R1 = 2.0        # 励磁回路取样电阻(Ω)
    n = 150         # 测量线圈匝数
    S = 80e-6       # 样品截面积(m²) - 80 mm² = 80×10⁻⁶ m²
    C2 = 20e-6      # 积分电容(F)
    R2 = 10e3       # 积分电阻(Ω)
    mu0 = 4 * np.pi * 1e-7  # 真空磁导率(H/m)


class MagnetizationData:
    """磁化曲线数据输入类
    
    数据说明：
    - U: 励磁电压(V)
    - U1: 励磁回路取样电阻电压(V) → 用于计算x轴磁场强度H
    - U2: 积分电容两端电压(V) → 用于计算y轴磁感应强度B
    
    注意：U1对应x轴H，U2对应y轴B
    """
    
    # =====================================================================
    # 磁滞回线关键点数据
    # 数据说明：
    # - U1x: 励磁回路取样电阻电压，用于计算磁场强度H（x轴）
    # - U2x: 积分电容两端电压，用于计算磁感应强度B（y轴）
    # =====================================================================
    
    # 饱和磁滞回线关键点数据
    # 正向：S(饱和点)、R(剩磁点)、D(矫顽力点)
    # 反向：S'(反向饱和点)、R'(反向剩磁点)、D'(反向矫顽力点)
    sample_hysteresis = {
        
        'U1S': 1.699, 'U2S': 169.0e-3,       # 正向饱和点：最大励磁时的数据
        
        'U1R': 9.880e-3, 'U2R': 36.80e-3,    # 正向剩磁点：H=0时的磁感应强度
        
        'U1D': 60.84e-3, 'U2D': 0.0,         # 正向矫顽力点：B=0时的磁场强度
        
        'U1S_prime': 1.660, 'U2S_prime': 167.0e-3,  # 反向饱和点
        
        'U1R_prime': 0.0, 'U2R_prime': 37.60e-3,    # 反向剩磁点
        
        'U1D_prime': 65.0e-3, 'U2D_prime': 0.0         # 反向矫顽力点
    }
    
    # =====================================================================
    # 基本磁化曲线数据
    # 数据格式：[U(V), U1(V), U2(V)]
    # - U1: 励磁回路取样电阻电压，对应x轴磁场强度H
    # - U2: 积分电容两端电压，对应y轴磁感应强度B
    # =====================================================================
    
    # 基本磁化曲线数据
    sample_magnetization = np.array([
        
        [0.5,  105.6e-3, 46.77e-3],
        
        [1.0,  214.0e-3, 79.58e-3],
        
        [1.2,  284.0e-3, 93.47e-3],
        
        [1.5,  454.4e-3, 116.9e-3],
        
        [1.8,  645.1e-3, 133.2e-3],
        
        [2.0,  813.1e-3, 141.9e-3],
        
        [2.2,  994.6e-3, 149.8e-3],
        
        [2.5,  1.251,    157.5e-3],
        
        [2.8,  1.499,    164.0e-3],
        
        [3.0,  1.650,    167.0e-3],
    ])


# ============================================================
# 2. 数据计算函数
# ============================================================

def calculate_magnetic_parameters(U1, U2, config):
    """
    计算磁场强度H、磁感应强度B、磁导率μ
    
    参数:
        U1: 励磁回路取样电阻电压(V) - 可以是标量或数组
        U2: 积分电容两端电压(V) - 可以是标量或数组
        config: 实验配置类
    
    返回:
        H: 磁场强度(A/m)
        B: 磁感应强度(T)
        mu: 磁导率(H/m)
    """
    H = (config.N / config.L) * (U1 / config.R1)
    B = (config.C2 * config.R2 / (config.n * config.S)) * U2
    
    # 处理数组情况
    if hasattr(H, '__len__') and len(H) > 1:
        mu = np.where(H != 0, B / H, 0.0)
    elif H == 0:
        mu = 0.0
    else:
        mu = B / H
    
    return H, B, mu


def calculate_magnetization_curve(data, config):
    """
    计算基本磁化曲线数据
    
    参数:
        data: 原始数据 [U, U1, U2]
        config: 实验配置类
    
    返回:
        DataFrame包含所有计算结果
    """
    U = data[:, 0]
    U1 = data[:, 1]
    U2 = data[:, 2]
    
    H, B, mu = calculate_magnetic_parameters(U1, U2, config)
    
    result = pd.DataFrame({
        'U(V)': U,
        'U1(V)': U1,
        'U2(V)': U2,
        'H(A/m)': H,
        'B(T)': B,
        'mu(H/m)': mu
    })
    
    return result


def calculate_hysteresis_parameters(hysteresis_data, config):
    """
    计算磁滞回线关键参数
    
    参数:
        hysteresis_data: 磁滞回线关键点数据字典
        config: 实验配置类
    
    返回:
        包含Bs, Br, Hc的字典
    """
    # 正向饱和点
    H_S, B_S, _ = calculate_magnetic_parameters(
        hysteresis_data['U1S'], hysteresis_data['U2S'], config
    )
    
    # 正向剩磁点
    H_R, B_R, _ = calculate_magnetic_parameters(
        hysteresis_data['U1R'], hysteresis_data['U2R'], config
    )
    
    # 正向矫顽力点
    H_D, B_D, _ = calculate_magnetic_parameters(
        hysteresis_data['U1D'], hysteresis_data['U2D'], config
    )
    
    # 反向饱和点
    H_S_prime, B_S_prime, _ = calculate_magnetic_parameters(
        hysteresis_data['U1S_prime'], hysteresis_data['U2S_prime'], config
    )
    
    # 反向剩磁点
    H_R_prime, B_R_prime, _ = calculate_magnetic_parameters(
        hysteresis_data['U1R_prime'], hysteresis_data['U2R_prime'], config
    )
    
    # 反向矫顽力点
    H_D_prime, B_D_prime, _ = calculate_magnetic_parameters(
        hysteresis_data['U1D_prime'], hysteresis_data['U2D_prime'], config
    )
    
    return {
        'Bs': B_S,
        'Br': B_R,
        'Hc': H_D,
        'Bs_prime': B_S_prime,
        'Br_prime': B_R_prime,
        'Hc_prime': H_D_prime,
        'H_S': H_S,
        'H_R': H_R,
        'H_D': H_D,
        'H_S_prime': H_S_prime,
        'H_R_prime': H_R_prime,
        'H_D_prime': H_D_prime
    }


# ============================================================
# 3. 绘图函数
# ============================================================

def plot_magnetization_curve(result_df, sample_name, save_path):
    """
    绘制B-H基本磁化曲线
    
    参数:
        result_df: 计算结果DataFrame
        sample_name: 样品名称
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    H = result_df['H(A/m)'].values
    B = result_df['B(T)'].values
    
    plt.plot(H, B, 'o-', linewidth=2, markersize=8, color='#1f77b4', label='实验数据')
    
    # 标注饱和点
    max_idx = np.argmax(B)
    plt.scatter(H[max_idx], B[max_idx], color='red', s=100, zorder=5, marker='*')
    plt.annotate(f'Bs={B[max_idx]:.3f}T', 
                 xy=(H[max_idx], B[max_idx]),
                 xytext=(H[max_idx]*1.1, B[max_idx]*0.9),
                 fontsize=10, color='red')
    
    plt.xlabel('磁场强度 H (A/m)', fontsize=12)
    plt.ylabel('磁感应强度 B (T)', fontsize=12)
    plt.title(f'{sample_name} 基本磁化曲线', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✓ 已保存: {save_path}")


def plot_permeability_curve(result_df, sample_name, save_path):
    """
    绘制μ-H磁导率曲线
    
    参数:
        result_df: 计算结果DataFrame
        sample_name: 样品名称
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    H = result_df['H(A/m)'].values
    mu = result_df['mu(H/m)'].values
    
    # 添加起点(H=0, μ=0)以显示完整的μ-H曲线趋势
    H = np.concatenate([[0], H])
    mu = np.concatenate([[0], mu])
    
    plt.plot(H, mu, 's-', linewidth=2, markersize=8, color='#ff7f0e', label='实验数据（含起点)')
    
    # 标注最大磁导率点
    max_idx = np.argmax(mu)
    plt.scatter(H[max_idx], mu[max_idx], color='red', s=100, zorder=5, marker='*')
    plt.annotate(f'μmax={mu[max_idx]:.2e}H/m', 
                 xy=(H[max_idx], mu[max_idx]),
                 xytext=(H[max_idx]*1.1, mu[max_idx]*0.8),
                 fontsize=10, color='red')
    
    plt.xlabel('磁场强度 H (A/m)', fontsize=12)
    plt.ylabel('磁导率 μ (H/m)', fontsize=12)
    plt.title(f'{sample_name} 磁导率曲线', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✓ 已保存: {save_path}")


def plot_hysteresis_loop(params, sample_name, save_path):
    """
    绘制饱和磁滞回线
    
    参数:
        params: 磁滞回线参数字典
        sample_name: 样品名称
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 8))
    
    # 提取关键点
    H_S = params['H_S']
    B_S = params['Bs']
    H_R = params['H_R']
    B_R = params['Br']
    H_D = params['H_D']
    B_D = params['Bc'] = 0  # 矫顽力点B=0
    
    H_S_prime = params['H_S_prime']
    B_S_prime = params['Bs_prime']
    H_R_prime = params['H_R_prime']
    B_R_prime = params['Br_prime']
    H_D_prime = params['H_D_prime']
    B_D_prime = params['Bc_prime'] = 0
    
    # 磁滞回线关键点定义（标准顺时针顺序）
    # 真实的磁滞回线路径：
    # 1. 从原点(0,0)出发
    # 2. 沿基本磁化曲线到正向饱和点(Hs, Bs)
    # 3. 退磁曲线：从(Hs, Bs)到正向剩磁点(0, Br)
    # 4. 继续退磁：从(0, Br)到正向矫顽力点(Hc, 0)
    # 5. 反向磁化：从(Hc, 0)到反向饱和点(-Hs, -Bs)
    # 6. 反向退磁：从(-Hs, -Bs)到反向剩磁点(0, -Br)
    # 7. 继续反向退磁：从(0, -Br)到反向矫顽力点(-Hc, 0)
    # 8. 回到正向饱和点(Hs, Bs)
    
    # 磁滞回线8个关键点（顺时针顺序）：
    # 1. 正向饱和点 (H_S, B_S)
    # 2. 正向剩磁点 (0, B_R)
    # 3. 正向矫顽力点 (H_D, 0)
    # 4. 反向饱和点 (-H_S_prime, -B_S_prime)
    # 5. 反向剩磁点 (0, -B_R_prime)
    # 6. 反向矫顽力点 (-H_D_prime, 0)
    H_points = np.array([H_S, 0, H_D, -H_S_prime, 0, -H_D_prime, H_S])
    B_points = np.array([B_S, B_R, 0, -B_S_prime, -B_R_prime, 0, B_S])
    
    # 使用样条插值使曲线更平滑
    from scipy import interpolate
    t = np.linspace(0, 1, len(H_points))
    t_new = np.linspace(0, 1, 300)
    
    # 插值函数
    H_interp = interpolate.interp1d(t, H_points, kind='cubic')
    B_interp = interpolate.interp1d(t, B_points, kind='cubic')
    H_smooth = H_interp(t_new)
    B_smooth = B_interp(t_new)
    
    # 绘制磁滞回线
    plt.plot(H_smooth, B_smooth, '-', linewidth=2.5, color='#2ca02c')
    plt.fill(H_smooth, B_smooth, alpha=0.15, color='#2ca02c')
    
    # 标注关键点
    plt.scatter([H_S], [B_S], color='red', s=120, marker='*', zorder=5, label='饱和点')
    plt.annotate(f'Bs={B_S:.3f}T', xy=(H_S, B_S), xytext=(H_S*0.55, B_S*1.15), 
                 fontsize=10, color='red', fontweight='bold')
    
    plt.scatter([0], [B_R], color='blue', s=100, marker='o', zorder=5, label='剩磁点')
    plt.annotate(f'Br={B_R:.3f}T', xy=(0, B_R), xytext=(H_S*0.15, B_R*1.1), 
                 fontsize=10, color='blue', fontweight='bold')
    
    plt.scatter([H_D], [0], color='purple', s=100, marker='s', zorder=5, label='矫顽力点')
    plt.annotate(f'Hc={H_D:.1f}A/m', xy=(H_D, 0), xytext=(H_D*1.15, B_S*0.15), 
                 fontsize=10, color='purple', fontweight='bold')
    
    plt.xlabel('磁场强度 H (A/m)', fontsize=12)
    plt.ylabel('磁感应强度 B (T)', fontsize=12)
    plt.title(f'{sample_name} 饱和磁滞回线', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(y=0, color='black', linewidth=0.5)
    plt.axvline(x=0, color='black', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✓ 已保存: {save_path}")


# ============================================================
# 4. 主函数
# ============================================================

def main():
    """主函数"""
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 创建输出目录
    output_dir = os.path.join(script_dir, 'output')
    results_dir = os.path.join(script_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 初始化配置和数据
    config = ExperimentConfig()
    data = MagnetizationData()
    
    print("=" * 70)
    print("铁磁材料磁化曲线与磁滞回线实验数据分析")
    print("=" * 70)
    
    # -------------------- 基本磁化曲线分析 --------------------
    print("\n" + "-" * 50)
    print("【一】基本磁化曲线分析")
    print("-" * 50)
    
    # 计算磁化曲线
    result = calculate_magnetization_curve(data.sample_magnetization, config)
    print(f"\n基本磁化曲线计算结果：")
    print(result.to_string(index=False))
    
    # 保存结果表格
    result_path = os.path.join(results_dir, 'magnetization_results.csv')
    result.to_csv(result_path, index=False)
    print(f"\n✓ 已保存: {result_path}")
    
    # 绘制图表
    plot_magnetization_curve(result, '样品', 
                            os.path.join(output_dir, 'magnetization_curve.png'))
    plot_permeability_curve(result, '样品', 
                            os.path.join(output_dir, 'permeability_curve.png'))
    
    # -------------------- 磁滞回线分析 --------------------
    print("\n" + "-" * 50)
    print("【二】磁滞回线分析")
    print("-" * 50)
    
    # 计算磁滞回线参数
    params = calculate_hysteresis_parameters(data.sample_hysteresis, config)
    print(f"\n磁滞回线关键参数：")
    print(f"  饱和磁感应强度 Bs = {params['Bs']:.4f} T")
    print(f"  剩磁 Br = {params['Br']:.4f} T")
    print(f"  矫顽力 Hc = {params['Hc']:.2f} A/m")
    
    # 绘制磁滞回线
    plot_hysteresis_loop(params, '样品', 
                        os.path.join(output_dir, 'hysteresis_loop.png'))
    
    # -------------------- 结果汇总 --------------------
    print("\n" + "-" * 50)
    print("【三】结果汇总")
    print("-" * 50)
    
    # 输出汇总表格
    summary_df = pd.DataFrame({
        '参数': ['饱和磁感应强度 Bs (T)', '剩磁 Br (T)', '矫顽力 Hc (A/m)'],
        '数值': [f"{params['Bs']:.4f}", f"{params['Br']:.4f}", f"{params['Hc']:.2f}"]
    })
    
    print(f"\n磁性参数汇总：")
    print(summary_df.to_string(index=False))
    
    summary_path = os.path.join(results_dir, 'summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ 已保存: {summary_path}")
    
    print("\n" + "=" * 70)
    print("✓ 所有分析完成！")
    print(f"  图表保存目录: {output_dir}")
    print(f"  数据保存目录: {results_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
