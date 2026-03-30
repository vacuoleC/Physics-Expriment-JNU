import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.signal as signal

import os
from script.Free_Vibration import FreeVibrationProcessor


def test_fit_quality():
    """
    测试函数：验证 fit_theta0_T0 和 plot_results 的拟合准确性
    """
    print("\n" + "="*30)
    print("开始运行拟合质量测试...")
    print("="*30 + "\n")

    # 1. 生成模拟参数
    fs = 1000
    dt = 1 / fs
    t = np.arange(0, 10, dt)
    
    # 真实物理参数：theta0 = 150, beta = 0.15
    real_theta0 = 150.0
    real_beta = 0.15
    freq = 2.0 # 振动频率 2Hz
    
    # 生成模拟信号：衰减正弦波 + 噪声
    np.random.seed(42)
    noise = np.random.normal(0, 1.5, len(t))
    signal_sim = real_theta0 * np.exp(-real_beta * t) * np.cos(2 * np.pi * freq * t) + noise

    # 2. 创建一个临时配置和处理器
    class MockConfig:
        FREE_VIBRATION = "" 
        FILTERED_DATA = "dummy.csv" # 只是为了满足 os.path.join 不报错，实际不会被读取
        RAW_DATA = "dummy.csv"
        PHI_DATA = "dummy.txt"
        DAMPED_VIBRATION = ""
        FORCED_VIBRATION = ""
    
    processor = FreeVibrationProcessor(MockConfig()) # type: ignore
    
    processor = FreeVibrationProcessor(MockConfig()) # type: ignore
    
    # 3. 手动注入模拟数据（跳过 load_data）
    processor.angle = signal_sim
    processor.time = t
    processor.fs = fs
    processor.dt = dt

    # 4. 运行处理流程
    processor.find_peak()
    processor.cut_below_30()
    processor.compute_results()
    processor.fit_theta0_T0()
    
    # 5. 绘制对比图（在原有 plot_results 基础上叠加真实包络线）
    plt.figure(figsize=(14, 6))
    
    # 绘制原始模拟信号
    plt.plot(processor.time, processor.angle, color='#888888', alpha=0.5, label='Simulated Noisy Signal')
    
    # 绘制真实包络线 (理论值)
    t_env = np.linspace(0, np.max(processor.time), 1000)
    real_envelope = real_theta0 * np.exp(-real_beta * t_env)
    plt.plot(t_env, real_envelope, 'k--', linewidth=2, alpha=0.7, label=f'Real Envelope: {real_theta0}·exp(-{real_beta}t)')
    
    # 绘制代码计算出的拟合包络线
    # 注意：这里我们需要重新计算一下拟合线，因为 plot_results 里是局部变量
    # 为了方便，我们直接利用 compute_results 里的数据重新画
    if processor.vaild_peaks_idx is not None and len(processor.vaild_peaks_idx) > 2: # type: ignore
        t_peaks = processor.time[processor.vaild_peaks_idx] # type: ignore
        ang_peaks = processor.vaild_angle # type: ignore
        log_ang = np.log(ang_peaks) # type: ignore
        coeffs_exp = np.polyfit(t_peaks, log_ang, 1)
        fit_beta = -coeffs_exp[0]
        fit_theta0 = np.exp(coeffs_exp[1])
        fit_envelope = fit_theta0 * np.exp(-fit_beta * t_env)
        
        plt.plot(t_env, fit_envelope, color='#0066cc', linewidth=2.5, label=f'Fitted Envelope: {fit_theta0:.1f}·exp(-{fit_beta:.4f}t)')
        
        # 标注拟合参数对比
        info_text = (
            f"真实参数:\n"
            f"  Theta0 = {real_theta0}\n"
            f"  Beta   = {real_beta}\n\n"
            f"拟合参数:\n"
            f"  Theta0 = {fit_theta0:.2f}\n"
            f"  Beta   = {fit_beta:.4f}\n"
            f"  R²     = {processor.fit_r_squared:.4f}"
        )
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.scatter(processor.time[processor.all_peaks_idx], processor.angle[processor.all_peaks_idx], 
                color='red', s=15, label='Detected Peaks', zorder=5)
    plt.title(f'Fitting Quality Test (Noise Level: 1.5)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()

def main():
    test_fit_quality()

if __name__ == "__main__":
    main()