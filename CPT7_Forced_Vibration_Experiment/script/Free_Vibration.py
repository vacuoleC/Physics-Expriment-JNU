import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.signal as signal

import os

class ExperimentConfig:
    """
    受迫振动实验配置类
    说明：
    1. 文件读取：类里面给出了具体的如何读取文件，请按照说明进行操作
    2. 单位：以教材上给出的单位为准，脚本会自动进行单位转换
    3. 

    """
    """
    数据文件配置说明

    目录结构要求：

    CPT7_Forced_Vibration_Experiment/
    ├── data/
    │   └── origin_data/           # 原始数据根目录
    │       ├── Free_Vibration/    # 自由振动数据
    |       │   ├── 滤波数据.csv
    │       │   └── 原始数据.csv
    │       ├── Damped_Vibration/  # 阻尼振动数据
    │       │   ├── 600/           # 600mA数据
    │       │   │   ├── 滤波数据.csv
    │       │   │   └── 原始数据.csv
    │       │   ├── 800/           # 可选
    │       │   └── 1000/          # 可选
    │       └── Forced_Vibration/  # 受迫振动数据
    │           ├── 1800/          # 频率1800Hz
    │           │   ├── dphi.txt   # 相位差数据
    │           │   ├── 滤波数据.csv
    │           │   └── 原始数据.csv
    │           ├── 1840/          # 其他频率数据...
    │           └── ...            # (1880-2200)

    使用说明：
    --------
    1. 自由振动数据：将"原始数据.csv"和"滤波数据.csv"放入 Free_Vibration/ 目录
    2. 阻尼振动数据：
    - 必需：600mA数据放入 Damped_Vibration/600/ 目录
    - 可选：800mA和1000mA数据放入对应目录（如无则忽略）
    3. 受迫振动数据：
    - 每个频率目录(1800-2200Hz)需包含：
        * dphi.txt（相位差数据）
        * 原始数据.csv（振动数据）
        * 滤波数据.csv
    4. 路径配置：
    - 脚本会自动读取到三个主目录（Free_Vibration等）
    - 只需确保目录结构与上述要求一致即可

    注意事项：
    --------
    - 所有目录名称必须严格匹配（包括大小写）
    - 缺失的可选数据不会影响程序运行
    - 程序会自动检测并跳过不存在的数据文件
    """

    # --------------------根目录读取--------------------
    DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'oringin_data')
    
    # --------------------自由振动数据目录--------------------
    FREE_VIBRATION = os.path.join(DATA_ROOT, 'Free_Vibration')
    
    # --------------------阻尼振动数据目录--------------------
    DAMPED_VIBRATION = os.path.join(DATA_ROOT, 'Damped_Vibration')
    
    # --------------------受迫振动数据目录--------------------
    FORCED_VIBRATION = os.path.join(DATA_ROOT, 'Forced_Vibration')
    
    # 文件名常量
    RAW_DATA = '原始数据.csv'
    FILTERED_DATA = '滤波数据.csv'
    PHI_DATA = 'dphi.txt'


class FreeVibrationProcessor:
    """
    自由振动数据处理类
    """
    def __init__(self, config: ExperimentConfig):
        self.data_path = os.path.join(config.FREE_VIBRATION, config.FILTERED_DATA)  # 滤波数据路径
        self.fs = 1000  # 采样频率
        self.dt = 1 / self.fs  # 采样间隔

        self.angle = None  # 滤波后的角度数据（核心数据）
        self.time = None  # 时间数据
        
        self.all_peaks_idx = None  # 所有峰值索引
        self.valid_peaks_idx = None  # 有效峰值索引（修正拼写错误）
        self.valid_angle = None  # 有效峰值对应的角度值（修正拼写错误）
        self.df = None  # 原始数据框

        self.T0_arr = None       # 用于拟合的 T0 数组
        self.theta0_arr = None   # 用于拟合的 theta0 数组
        self.fit_slope = None    # 拟合斜率
        self.fit_intercept = None# 拟合截距
        self.fit_r_squared = None# 拟合优度 R²

        self.results: list = []  # 存储结果[(time, angle), ...]

    def load_data(self):
        """加载并预处理数据（去趋势+低通滤波）"""
        self.df = pd.read_csv(self.data_path, header=None)
        raw_angle = self.df.iloc[:, 0].values
        
        # 预处理：去趋势 + 低通滤波（核心正确角度数据）
        self.angle = signal.detrend(raw_angle)  # 去除线性趋势# type: ignore
        b, a = signal.butter(4, 10, 'low', fs=self.fs)  # type: ignore # 4阶巴特沃斯低通
        self.angle = signal.filtfilt(b, a, self.angle)  # 零相位滤波
        
        # 生成时间轴
        self.time = np.arange(len(self.angle)) * self.dt

    def find_peak(self):
        """优化寻峰：先粗找周期，再精准找主波峰"""
        # 粗找峰值，估算周期
        rough_peaks, _ = signal.find_peaks(
            self.angle, 
            distance=100, 
            height=np.max(self.angle) * 0.05 # type: ignore
        )
        
        # 动态计算最小峰距（避免固定值误差）
        if len(rough_peaks) >= 2:
            rough_period = np.mean(np.diff(rough_peaks)) * self.dt
            min_distance = int(rough_period * self.fs * 0.7)
        else:
            min_distance = 700

        # 精准找峰（主波峰）
        peaks, properties = signal.find_peaks(
            self.angle,
            distance=min_distance,
            height=np.max(self.angle) * 0.1, # type: ignore
            prominence=np.max(self.angle) * 0.05, # type: ignore
            width=50,
            wlen=min_distance * 2
        )

        self.all_peaks_idx = peaks
        print(f'✅ 优化寻峰完成：共找到{len(peaks)}个真实主波峰')
        print(f'   峰值时间范围：{self.time[peaks[0]]:.2f}s ~ {self.time[peaks[-1]]:.2f}s')# type: ignore
        print(f'   峰值角度范围：{self.angle[peaks[0]]:.2f}° ~ {self.angle[peaks[-1]]:.2f}°')# type: ignore

    def cut_below_30(self):
        """过滤掉角度小于30°的峰值（保留有效峰值）"""
        # 从滤波后的角度数组中取峰值对应的y值（核心修正）
        peak_angles = self.angle[self.all_peaks_idx]# type: ignore
        
        # 找到第一个小于30°的峰值索引
        below_30_idx = np.where(peak_angles < 30)[0]
        if len(below_30_idx) > 0:
            first_below = below_30_idx[0]
            self.valid_peaks_idx = self.all_peaks_idx[:first_below]# type: ignore
        else:
            self.valid_peaks_idx = self.all_peaks_idx

        # 有效峰值对应的角度值（滤波后）
        self.valid_angle = self.angle[self.valid_peaks_idx]# type: ignore
        print(f'有效峰值数量：{len(self.valid_peaks_idx)}')# type: ignore

    def compute_results(self):
        """计算周期(T0)和平均角度(theta0)：使用滤波后的正确角度值"""
        self.results = []  # 重置结果列表
        
        # 遍历有效峰值，计算每两个相邻峰的周期和平均角度
        for i in range(len(self.valid_peaks_idx) - 1):# type: ignore
            # 峰值索引
            peak_idx_1 = self.valid_peaks_idx[i]# type: ignore
            peak_idx_2 = self.valid_peaks_idx[i + 1]# type: ignore
            
            # 正确的角度值（滤波后）
            angle_1 = self.angle[peak_idx_1]# type: ignore
            angle_2 = self.angle[peak_idx_2]# type: ignore
            
            # 计算周期和平均角度
            T0 = (peak_idx_2 - peak_idx_1) * self.dt
            theta0 = (angle_1 + angle_2) / 2
            
            # 存储结果
            self.results.append((T0, theta0))

        # 打印结果（供实验报告使用）
        print('='*60)
        print("实验报告表格数据（使用滤波后正确角度值）")
        for i, (T0, theta0) in enumerate(self.results):
            print(f'第{i+1}组  T0: {T0:.4f}s, theta0: {theta0:.4f}°')
        print('='*60)

    def fit_theta0_T0(self):
        """对 theta0-T0 进行线性拟合，并计算 R²（基于正确的角度值）"""
        if len(self.results) < 2:
            print("⚠️ 有效峰值数量不足，无法进行线性拟合")
            return
        
        # 提取拟合用的数组（基于正确的计算结果）
        self.T0_arr = np.array([item[0] for item in self.results])
        self.theta0_arr = np.array([item[1] for item in self.results])

        # 线性拟合 (1次多项式)
        coeffs = np.polyfit(self.T0_arr, self.theta0_arr, 1)
        self.fit_slope, self.fit_intercept = coeffs

        # 计算拟合优度 R²
        y_pred = self.fit_slope * self.T0_arr + self.fit_intercept
        ss_res = np.sum((self.theta0_arr - y_pred) ** 2)
        ss_tot = np.sum((self.theta0_arr - np.mean(self.theta0_arr)) ** 2)
        self.fit_r_squared = 1 - (ss_res / ss_tot)

        print("==== theta0-T0 线性拟合结果（修正后） ====")
        print(f"拟合方程: theta0 = {self.fit_slope:.4f} * T0 + {self.fit_intercept:.4f}")
        print(f"拟合优度 R² = {self.fit_r_squared:.6f}")

    def plot_results(self):
        """可视化：原始信号 + 峰值 + 衰减包络线 + 拟合结果"""
        plt.figure(figsize=(14, 6))

        # 绘制原始滤波信号
        plt.plot(
            self.time, # type: ignore
            self.angle,# type: ignore
            color='#888888',
            label='Filtered Angle Signal'
        )
        
        # 绘制所有峰值
        plt.scatter(
            self.time[self.all_peaks_idx],# type: ignore
            self.angle[self.all_peaks_idx],# type: ignore
            color='red',
            s=12,
            alpha=0.6,
            label='All Peaks'
        )
        
        # 绘制有效峰值
        plt.scatter(
            self.time[self.valid_peaks_idx],# type: ignore
            self.angle[self.valid_peaks_idx],# type: ignore
            color='green',
            s=25,
            label='Valid Peaks (≥30°)',
            zorder=5
        )
        
        # 30°阈值线
        plt.axhline(
            y=30,
            color='orange',
            linestyle='--',
            linewidth=1.5,
            alpha=0.8,
            label='30° Cut-off'
        )

        # 绘制衰减包络线（指数拟合）
        if self.valid_peaks_idx is not None and len(self.valid_peaks_idx) > 2:
            t_peaks = self.time[self.valid_peaks_idx]# type: ignore
            ang_peaks = self.valid_angle# type: ignore
            
            # 指数拟合（θ = θ0 * exp(-βt)）：取对数转线性拟合
            log_ang = np.log(ang_peaks)# type: ignore
            coeffs_exp = np.polyfit(t_peaks, log_ang, 1)
            beta = -coeffs_exp[0]
            theta0_exp = np.exp(coeffs_exp[1])
            
            # 生成包络线
            t_env = np.linspace(0, np.max(t_peaks), 1000)
            ang_env = theta0_exp * np.exp(-beta * t_env)
            
            # 绘制包络线
            plt.plot(
                t_env, ang_env, color='#0066cc', linewidth=2.5, linestyle='-',
                label=f'Decay Envelope: θ = {theta0_exp:.1f}·exp(-{beta:.4f}t)',
                zorder=4
            )
            
            # 标注线性拟合结果
            if self.fit_slope is not None:
                fit_text = f'θ₀-T₀ Fit: θ₀ = {self.fit_slope:.2f}·T₀ + {self.fit_intercept:.1f}\n(R²={self.fit_r_squared:.4f})'
                plt.text(
                    0.02, 0.98,
                    fit_text,
                    transform=plt.gca().transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
                )

        plt.xlabel('Time (s)')
        plt.ylabel('Angle (°)')
        plt.title('Free Vibration Analysis (Corrected Y-Value)')
        plt.legend(loc='upper right')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def run(self):
        """执行完整处理流程"""
        self.load_data()
        self.find_peak()
        self.cut_below_30()
        self.compute_results()
        self.fit_theta0_T0()
        self.plot_results()


def main():
    config = ExperimentConfig()
    processor = FreeVibrationProcessor(config)
    processor.run()


if __name__ == '__main__':
    main()