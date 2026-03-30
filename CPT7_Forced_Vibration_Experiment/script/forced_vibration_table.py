import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']  # 优先使用黑体/微软雅黑/宋体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# ===================== 配置类 =====================
class ExperimentConfig:
    DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'oringin_data') # type: ignore
    FREE_VIBRATION = os.path.join(DATA_ROOT, 'Free_Vibration') # type: ignore
    FORCED_VIBRATION = os.path.join(DATA_ROOT, 'Forced_Vibration') # type: ignore
    FILTERED_DATA = '滤波数据.csv'
    PHI_DATA = 'dphi.txt'

# ===================== 工具函数：安全读取文件，绝对不返回文件对象 =====================
def safe_read_phi(file_path: str) -> float:
    """【核心安全函数】只返回float，绝对不返回文件对象"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}")
    content = ""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # 极端清洗：只保留数字和小数点
    clean_content = ''.join([c for c in content if c.isdigit() or c == '.' or c == '-'])
    phi = float(clean_content)
    return phi

def safe_read_theta(csv_path: str, fs: int) -> float:
    """【核心安全函数】只返回float振幅"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"文件不存在：{csv_path}")
    df = pd.read_csv(csv_path, header=None) # type: ignore
    raw_angle = df.iloc[:, 0].values # type: ignore
    angle = signal.detrend(raw_angle) # type: ignore
    b, a = signal.butter(4, 20, 'low', fs=fs) # type: ignore
    angle = signal.filtfilt(b, a, angle) # type: ignore
    
    steady_start = int(len(angle) * 0.3)
    steady_angle = angle[steady_start:] # type: ignore
    peaks, _ = signal.find_peaks( # type: ignore
        steady_angle, distance=int(fs*0.2),
        height=np.max(steady_angle)*0.5 # type: ignore
    )
    if len(peaks) < 3: # type: ignore
        raise ValueError("峰值不足")
    
    theta = float(np.mean(steady_angle[peaks])) # type: ignore
    return theta

# ===================== 自由振动拟合 =====================
class FreeVibrationFitter:
    def __init__(self, config: ExperimentConfig, sample_rate: int = 100):
        self.data_path = os.path.join(config.FREE_VIBRATION, config.FILTERED_DATA) # type: ignore
        self.fs = sample_rate
        self.fit_k = 0.0
        self.fit_b = 0.0

    def load_and_process(self):
        df = pd.read_csv(self.data_path, header=None) # type: ignore
        raw_angle = df.iloc[:, 0].values # type: ignore
        angle = signal.detrend(raw_angle) # type: ignore
        b, a = signal.butter(4, 10, 'low', fs=self.fs) # type: ignore
        angle = signal.filtfilt(b, a, angle) # type: ignore
        time = np.arange(len(angle)) / self.fs # type: ignore

        rough_peaks, _ = signal.find_peaks(angle, distance=int(self.fs*0.5), height=np.max(angle)*0.05) # type: ignore
        min_dist = int(np.mean(np.diff(rough_peaks)) * 0.7) if len(rough_peaks)>=2 else 70 # type: ignore
        peaks, _ = signal.find_peaks(angle, distance=min_dist, height=np.max(angle)*0.1) # type: ignore
        
        T0_list = []
        theta0_list = []
        for i in range(len(peaks)-1): # type: ignore
            t1, t2 = peaks[i], peaks[i+1] # type: ignore
            a1, a2 = angle[t1], angle[t2] # type: ignore
            T0_list.append(float((t2 - t1) / self.fs))
            theta0_list.append(float((a1 + a2) / 2))
        
        coeffs = np.polyfit(np.array(T0_list), np.array(theta0_list), 1) # type: ignore
        self.fit_k, self.fit_b = float(coeffs[0]), float(coeffs[1])
        
        print("✅ 自由振动拟合完成")
        print(f"   θ0 = {self.fit_k:.4f} * T0 + {self.fit_b:.4f}")
        return self.fit_k, self.fit_b

    def get_T0(self, theta: float) -> float:
        theta_f = float(theta)
        T0 = (theta_f - self.fit_b) / self.fit_k
        return float(T0)

# ===================== 主程序 =====================
def main():
    config = ExperimentConfig()
    fs = 100

    # 1. 自由振动
    try:
        fitter = FreeVibrationFitter(config, fs)
        fitter.load_and_process()
    except Exception as e:
        print(f"❌ 自由振动失败：{e}")
        return

    # 2. 受迫振动数据收集
    forced_dir = config.FORCED_VIBRATION
    if not os.path.exists(forced_dir):
        print(f"❌ 受迫振动目录不存在：{forced_dir}")
        return

    freq_dirs = [d for d in os.listdir(forced_dir) if os.path.isdir(os.path.join(forced_dir, d))] # type: ignore
    if len(freq_dirs) == 0:
        print("❌ 未找到频率文件夹")
        return

    data_list = []
    print(f"\n===== 开始加载数据 =====")
    for d_name in freq_dirs:
        try:
            # 【关键】每一步都强制转float，确保没有文件对象
            f = float(d_name)
            csv_p = os.path.join(forced_dir, d_name, config.FILTERED_DATA) # type: ignore
            phi_p = os.path.join(forced_dir, d_name, config.PHI_DATA) # type: ignore
            
            theta = safe_read_theta(csv_p, fs)
            phi = safe_read_phi(phi_p)
            omega = float(2 * np.pi * f)
            
            data_list.append({
                "f": f, "omega": omega, "theta": theta, "phi": phi
            })
            print(f"   ✅ {f:.0f}Hz | θ={theta:.2f}° | φ={phi:.0f}°")
        except Exception as e:
            print(f"   ⚠️  跳过 {d_name}：{e}")
            continue

    if len(data_list) == 0:
        print("❌ 无有效数据")
        return

    # 3. 找共振点
    data_list = sorted(data_list, key=lambda x: x["f"]) # type: ignore
    theta_list = [d["theta"] for d in data_list] # type: ignore
    res_idx = np.argmax(theta_list) # type: ignore
    res_data = data_list[res_idx] # type: ignore
    fr = float(res_data["f"])
    omega_r = float(res_data["omega"])
    theta_r = float(res_data["theta"])
    print(f"\n✅ 共振点：fr={fr:.0f}Hz, θr={theta_r:.2f}°")

    # 4. 生成表格
    table_rows = []
    for d in data_list: # type: ignore
        f = float(d["f"])
        omega = float(d["omega"])
        theta = float(d["theta"])
        phi = float(d["phi"])
        
        T0 = fitter.get_T0(theta)
        omega0 = float(2 * np.pi / T0)
        ratio_omega = float(omega / omega_r)
        ratio_amp_sq = float((theta / theta_r) ** 2)
        
        table_rows.append({
            "f/Hz": round(f, 0),
            "ω/rad/s": round(omega, 2),
            "θ/°": round(theta, 2),
            "T0/s": round(T0, 4),
            "ω0/rad/s": round(omega0, 2),
            "ω/ωr": round(ratio_omega, 4),
            "(θ/θr)²": round(ratio_amp_sq, 4),
            "-φ/°": round(phi, 0)
        })

    df_table = pd.DataFrame(table_rows) # type: ignore
    print("\n" + "="*150)
    print("📋 表4.5.3（可直接抄写）")
    print("-"*150)
    print(df_table.to_string(index=False)) # type: ignore
    print("="*150 + "\n")

    # 5. 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10)) # type: ignore
    
    ax1.plot(df_table["ω/ωr"], df_table["(θ/θr)²"], 'o-', color='#1f77b4', markersize=7) # type: ignore
    ax1.axvline(1.0, color='red', linestyle=':', label='共振') # type: ignore
    ax1.set_title('幅频特性曲线') # type: ignore
    ax1.legend() # type: ignore
    ax1.grid(alpha=0.3) # type: ignore
    
    ax2.plot(df_table["ω/ωr"], df_table["-φ/°"], 's-', color='#ff7f0e', markersize=7) # type: ignore
    ax2.axvline(1.0, color='red', linestyle=':', label='共振') # type: ignore
    ax2.axhline(90, color='gray', linestyle='--') # type: ignore
    ax2.set_title('相频特性曲线') # type: ignore
    ax2.legend() # type: ignore
    ax2.grid(alpha=0.3) # type: ignore
    ax2.set_ylim(0, 180) # type: ignore
    
    plt.tight_layout() # type: ignore
    plt.show() # type: ignore

if __name__ == '__main__':
    main()