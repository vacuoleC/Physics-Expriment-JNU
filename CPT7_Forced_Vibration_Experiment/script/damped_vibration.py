import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']  # 优先使用黑体/微软雅黑/宋体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

class ExperimentConfig:
    """受迫振动实验配置类，匹配你的目录结构"""
    # 根目录：严格匹配你的文件夹名 oringin_data
    DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'oringin_data')
    # 阻尼振动数据目录
    DAMPED_VIBRATION = os.path.join(DATA_ROOT, 'Damped_Vibration')
    # 文件名常量
    FILTERED_DATA = '滤波数据.csv'


class DampedVibrationProcessor:
    """
    阻尼振动数据处理类
    严格匹配教材表4.5.2格式，100Hz采样，仅正峰，逐差法n=5计算β
    """
    def __init__(self, config: ExperimentConfig, damping_current: str = "600"):
        self.config = config
        self.damping_current = damping_current
        # 数据路径匹配你的目录结构
        self.data_path = os.path.join(
            config.DAMPED_VIBRATION, 
            damping_current, 
            config.FILTERED_DATA
        )
        
        # 实验核心参数（严格按你的要求设置）
        self.fs = 100  # 采样率固定为100Hz，匹配实验实际设置
        self.dt = 1 / self.fs  # 采样间隔
        self.required_peak_num = 11  # 11个正峰=10个完整周期，匹配表格10T要求
        self.n = 5  # 教材逐差法n=5

        # 核心数据
        self.angle = None  # 预处理后的角度数据
        self.time = None   # 时间轴
        self.df = None     # 原始数据

        # 正峰寻峰结果
        self.peak_idx = None       # 正峰索引
        self.peak_time = None      # 正峰对应的时间
        self.peak_amplitude = None # 正峰对应的振幅（正值）

        # 表格计算用数据
        self.table_theta = None    # 表格1-10次的振幅θ1-θ10
        self.ln_ratio_list = None  # ln(θi/θi+5)列表
        self.ln_ratio_mean = None  # ln比值的平均值
        self.T_10 = None           # 10个周期的总时间10T
        self.T_bar = None          # 平均周期T̄
        self.beta = None           # 最终阻尼系数β

    def load_data(self):
        """加载并预处理数据，适配100Hz采样率"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据文件不存在：{self.data_path}")
        
        # 加载滤波数据
        self.df = pd.read_csv(self.data_path, header=None)
        raw_angle = self.df.iloc[:, 0].values
        
        # 预处理：去趋势+低通滤波，适配100Hz采样
        self.angle = signal.detrend(raw_angle)# type: ignore
        b, a = signal.butter(4, 10, 'low', fs=self.fs)  # type: ignore # 10Hz截止频率，适配100Hz采样
        self.angle = signal.filtfilt(b, a, self.angle)
        
        # 生成时间轴
        self.time = np.arange(len(self.angle)) * self.dt
        print(f"✅ [{self.damping_current}mA] 数据加载完成")
        print(f"   采样率：{self.fs}Hz，总时长：{len(self.angle)*self.dt:.2f}s，数据点数：{len(self.angle)}")

    def find_positive_peaks(self):
        """仅寻找正方向的峰（极大值），完全剔除负峰，适配100Hz采样"""
        # 寻峰参数适配100Hz采样率，放宽阈值保证找到足够的峰
        peaks, _ = signal.find_peaks(
            self.angle,
            distance=int(self.fs * 0.5),  # 最小峰间隔0.5s，过滤噪声峰
            height=np.max(self.angle) * 0.02,  # 放宽高度阈值，保留小振幅峰# type: ignore
            prominence=np.max(self.angle) * 0.01, # 放宽突出度阈值# type: ignore
            width=int(self.fs * 0.02)
        )

        self.peak_idx = peaks
        self.peak_time = self.time[peaks]# type: ignore
        self.peak_amplitude = self.angle[peaks]# type: ignore

        print(f"✅ [{self.damping_current}mA] 正峰寻峰完成，共找到{len(peaks)}个正峰")
        print(f"   正峰时间范围：{self.peak_time[0]:.2f}s ~ {self.peak_time[-1]:.2f}s")
        print(f"   正峰振幅范围：{self.peak_amplitude[-1]:.2f}° ~ {self.peak_amplitude[0]:.2f}°")

        # 检查峰数量是否满足要求（至少11个正峰）
        if len(peaks) < self.required_peak_num:
            raise ValueError(
                f"有效正峰数量不足，仅找到{len(peaks)}个，至少需要{self.required_peak_num}个。\n"
                f"⚠️  原因：当前数据总时长仅{len(self.angle)*self.dt:.2f}s，不足以采集10个完整振动周期，请补充更长时间的实验数据。"# type: ignore
            )

    def generate_table_data(self):
        """生成严格匹配教材表4.5.2的可填写数据"""
        # 取前11个连续正峰，保证10个完整周期
        valid_peaks_time = self.peak_time[:self.required_peak_num]# type: ignore
        valid_peaks_amp = self.peak_amplitude[:self.required_peak_num]# type: ignore

        # 表格1-10次的振幅（取前10个峰）
        self.table_theta = valid_peaks_amp[:10]
        # 10个周期的总时间10T = 第11个峰时间 - 第1个峰时间
        self.T_10 = valid_peaks_time[-1] - valid_peaks_time[0]
        # 平均周期T̄
        self.T_bar = self.T_10 / 10

        # 逐差法计算ln(θi/θi+5)，i=1~5
        self.ln_ratio_list = []
        for i in range(self.n):
            theta_i = self.table_theta[i]
            theta_i5 = self.table_theta[i + self.n]
            ln_ratio = np.log(theta_i / theta_i5)
            self.ln_ratio_list.append(ln_ratio)
        
        # 计算ln比值的平均值
        self.ln_ratio_mean = np.mean(self.ln_ratio_list)

        # 打印可直接填表的完整数据
        print('='*100)
        print(f"📊 [{self.damping_current}mA] 表4.5.2 阻尼系数测量数据（可直接填写）")
        print('-'*100)
        print(f"{'次数':<6} {'振幅(°)':<10} {'次数':<6} {'振幅(°)':<10} {'ln(θi/θi+5)':<15}")
        print('-'*100)
        for i in range(self.n):
            print(f"{i+1:<6} {self.table_theta[i]:<10.2f} {i+6:<6} {self.table_theta[i+5]:<10.2f} {self.ln_ratio_list[i]:<15.4f}")
        print('-'*100)
        print(f"{'':<22} ln(θi/θi+5)平均值：{self.ln_ratio_mean:.4f}")
        print(f"{'':<22} 10T = {self.T_10:.2f}s ； 平均周期 T̄ = {self.T_bar:.3f}s")
        print('='*100)

    def calculate_beta(self):
        """严格按教材公式(4.5.17)计算阻尼系数β"""
        self.beta = self.ln_ratio_mean / (self.n * self.T_bar)# type: ignore
        print(f"✅ [{self.damping_current}mA] 阻尼系数计算完成")
        print(f"   最终平均阻尼系数 β = {self.beta:.6f} s⁻¹")
        print('='*100)

    def plot_results(self):
        """可视化振动曲线、正峰标注、计算结果，可直接插入实验报告"""
        plt.figure(figsize=(14, 7))

        # 绘制滤波后的阻尼振动曲线
        plt.plot(self.time, self.angle, color='#888888', alpha=0.6, label='阻尼振动滤波信号')# type: ignore
        
        # 绘制所有正峰
        plt.scatter(self.peak_time, self.peak_amplitude,# type: ignore
                    color='red', s=30, alpha=0.7, label='所有正峰')
        
        # 绘制表格用的11个有效正峰（重点标注）
        valid_peak_time = self.peak_time[:self.required_peak_num]# type: ignore
        valid_peak_amp = self.peak_amplitude[:self.required_peak_num]# type: ignore
        plt.scatter(valid_peak_time, valid_peak_amp,
                    color='#9400d3', s=80, zorder=5, label='表格计算用有效正峰(11个)')

        # 标注核心计算结果
        result_text = (
            f'阻尼电流：{self.damping_current}mA\n'
            f'采样率：{self.fs}Hz\n'
            f'10个周期总时间 10T = {self.T_10:.2f}s\n'
            f'平均周期 T̄ = {self.T_bar:.3f}s\n'
            f'ln(θi/θi+5)平均值 = {self.ln_ratio_mean:.4f}\n'
            f'最终阻尼系数 β = {self.beta:.6f} s⁻¹'
        )
        plt.text(
            0.02, 0.98,
            result_text,
            transform=plt.gca().transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
        )

        plt.xlabel('绝对时间 t (s)', fontsize=12)
        plt.ylabel('角度 θ (°)', fontsize=12)
        plt.title(f'阻尼振动分析（{self.damping_current}mA）- 100Hz采样 仅正峰', fontsize=14)
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def run(self):
        """执行完整处理流程"""
        print(f"\n===== 开始处理 {self.damping_current}mA 阻尼振动数据 =====")
        self.load_data()
        self.find_positive_peaks()
        self.generate_table_data()
        self.calculate_beta()
        self.plot_results()
        print(f"===== {self.damping_current}mA 数据处理完成 =====\n")


def main():
    """主函数：自动处理600、800、1000mA三组数据"""
    config = ExperimentConfig()
    damping_current_list = ["600", "800", "1000"]

    for current in damping_current_list:
        # 检查文件夹是否存在
        current_dir = os.path.join(config.DAMPED_VIBRATION, current)
        if not os.path.exists(current_dir):
            print(f"⚠️  {current}mA 文件夹不存在，跳过处理")
            continue
        # 处理数据
        try:
            processor = DampedVibrationProcessor(config, damping_current=current)
            processor.run()
        except Exception as e:
            print(f"❌ {current}mA 数据处理失败：{str(e)}")


if __name__ == '__main__':
    main()