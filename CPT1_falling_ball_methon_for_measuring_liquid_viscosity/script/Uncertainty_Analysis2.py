import numpy as np


class ExperimentConfig:
    """
    落球法黏滞系数测量实验配置类
    说明：
    1. 所有参数标注实验常用单位，内部自动转换为国际单位制（m/kg/s）
    2. 可直接修改以下参数适配自己的实验数据，无需改动后续核心逻辑
    3. 多组测量数据在measure_groups中添加/删除即可
    """
    # ---------------------- 基础物理参数（实验测量/给定值）----------------------
    rhop = 0.945          # 油的密度 (g/cm³) → 自动转 kg/m³
    rho = 7.769           # 小球密度 (g/cm³) → 自动转 kg/m³
    L = 20.0              # 小球匀速下落距离 (cm) → 自动转 m
    H = 33.0              # 液柱高度 (cm) → 自动转 m
    D = 2.23              # 容器内径 (cm) → 自动转 m
    d0 = 0.008            # 千分尺零点误差 (cm) → 自动转 m
    T1 = 25.4             # 温度测量值1 (℃)（仅记录，不参与计算）
    T2 = 25.4             # 温度测量值2 (℃)（仅记录，不参与计算）
    g = 9.788             # 重力加速度 (m/s²)（固定值，无需修改）

    # ---------------------- 误差参数（仪器精度）----------------------
    delta_rhop = 0.003    # 油密度测量误差（比重计精度）(±g/cm³) → 自动转 kg/m³
    delta_t = 0.01        # 时间测量误差（秒表精度）(±s)
    delta_L_ruler = 0.15  # 普通尺精度 (±cm) → 自动转 m
    delta_L_microm = 0.004# 千分尺精度 (±cm) → 自动转 m

    # ---------------------- 多组测量数据（d:小球直径，t:下落时间）----------------------
    # 格式：{"组名": {"d_cm": [直径测量值(cm)], "t_s": [时间测量值(s)]}}
    # 可新增/删除组（如"中球组"），主函数会自动处理所有组
    measure_groups = {
        "小球组": {
            "d_cm": [0.1941, 0.1936, 0.2000, 0.1924, 0.1950],
            "t_s": [10.23, 10.43, 10.38, 10.41, 10.20]
        },
        "大球组": {
            "d_cm": [0.3002, 0.3009, 0.3015, 0.3008, 0.3001],
            "t_s": [5.02, 5.05, 5.09, 5.08, 5.10]
        }
    }

    # ---------------------- 单位转换方法（内部使用，无需修改）----------------------
    @classmethod
    def convert_to_si(cls):
        """将实验单位转换为国际单位制，返回转换后的参数字典"""
        return {
            # 密度转换：1 g/cm³ = 1000 kg/m³
            "rhop": cls.rhop * 1000,
            "rho": cls.rho * 1000,
            # 长度转换：1 cm = 0.01 m
            "L": cls.L * 0.01,
            "H": cls.H * 0.01,
            "D": cls.D * 0.01,
            "d0": cls.d0 * 0.01,
            # 误差参数转换
            "delta_rhop": cls.delta_rhop * 1000,
            "delta_L_ruler": cls.delta_L_ruler * 0.01,
            "delta_L_microm": cls.delta_L_microm * 0.01,
            # 无需转换的参数
            "g": cls.g,
            "delta_t": cls.delta_t,
            "T1": cls.T1,
            "T2": cls.T2
        }

    # ---------------------- 测量数据转换（内部使用，无需修改）----------------------
    @classmethod
    def get_converted_measure_groups(cls):
        """转换测量组数据为国际单位制：d_cm → d_m"""
        converted_groups = {}
        for group_name, data in cls.measure_groups.items():
            # 直径cm转m，时间保持s不变
            converted_groups[group_name] = {
                "d_m": [d * 0.01 for d in data["d_cm"]],
                "t_s": data["t_s"]
            }
        return converted_groups

# ============================ 核心计算类（无需修改）============================
class ViscosityUncertaintyCalculator:
    """
    落球法测量液体黏滞系数的不确定度计算器（解析偏导版）
    核心改进：灵敏系数采用严格解析偏导数，替代数值近似的中心差分法
    全程使用国际单位制（米/秒/千克）
    """

    def __init__(
            self, 
            d_measurements,  # 小球直径多次测量值（m，未修正零点误差）
            t_measurements,  # 下落时间多次测量值（s）
            L,               # 下落距离（m）
            D,               # 容器内径（m）
            H,               # 液柱高度（m）
            rho,             # 小球密度（kg/m³）
            rhop,            # 油的密度（kg/m³，比重计直接测量）
            g=9.788,         # 重力加速度（m/s²）
            # 误差参数（均为国际单位制）
            micrometer_half_div=0.00004,  # 千分尺半宽（m）：±0.004cm
            ruler_half_div=0.0015,        # 普通尺半宽（m）：±0.15cm
            micrometer_zero_error=0.0005, # 千分尺零点误差（m）：0.050cm
            t_half_div=0.01,              # 秒表半宽（s）：±0.01s
            rhop_half_div=3,              # 比重计半宽（kg/m³）：±0.003g/cm³
    ):
        """
        参数说明（全为国际单位制）：
        ----------
        d_measurements : list of float
            小球直径的多次测量值 (m)（未修正零点误差）
        t_measurements : list of float
            下落时间的多次测量值 (s)
        L : float
            小球匀速下落距离 (m)
        D : float
            盛放液体的容器内径 (m)
        H : float
            液柱高度 (m)
        rho : float
            小球密度 (kg/m³)
        rhop : float
            油的密度 (kg/m³)（比重计直接测量，无温度修正）
        g : float, optional
            重力加速度 (m/s²)，默认9.788
        micrometer_half_div : float, optional
            千分尺最小分度的一半 (m)（小球直径B类误差）
        ruler_half_div : float, optional
            普通尺最小分度的一半 (m)（L/D/H的B类误差）
        micrometer_zero_error : float, optional
            千分尺零点误差 (m)（修正小球直径测量值）
        t_half_div : float, optional
            时间测量仪器半宽 (s)（时间B类误差）
        rhop_half_div : float, optional
            比重计测量半宽 (kg/m³)（油密度B类误差）
        """
        # 原始数据（国际单位制）
        self.d_meas_raw = np.array(d_measurements)
        self.t_meas = np.array(t_measurements)
        self.L = L
        self.D = D
        self.H = H
        self.rho = rho
        self.rhop = rhop
        self.g = g
        
        # 误差参数（细分）
        self.microm_half = micrometer_half_div  # 千分尺半宽
        self.ruler_half = ruler_half_div        # 普通尺半宽
        self.d0 = micrometer_zero_error         # 千分尺零点误差
        self.t_half = t_half_div                # 秒表半宽
        self.rhop_half = rhop_half_div          # 比重计半宽

        # 置信系数（依据实验规范）
        self.C_normal = 3          # 正态分布（千分尺/米尺/秒表）
        self.C_uniform = np.sqrt(3)# 均匀分布（比重计）

        # 异常值校验（避免后续计算崩溃）
        self._validate_inputs()

        # 自动执行计算
        self.calculate()

    def _validate_inputs(self):
        """输入参数校验，避免分母为0等异常"""
        # 核心参数不能为0
        zero_check = [self.L, self.D, self.H, self.g]
        if any(v == 0 for v in zero_check):
            raise ValueError("L/D/H/g 不能为0！")
        
        # 小球密度需大于油密度（否则公式无物理意义）
        if self.rho - self.rhop <= 0:
            raise ValueError("小球密度ρ必须大于油密度ρ'！当前ρ-ρ' = {:.2f} kg/m³".format(self.rho - self.rhop))

        # 直径/时间测量值不能为0
        if np.min(self.d_meas_raw) <= 0 or np.min(self.t_meas) <= 0:
            raise ValueError("小球直径和下落时间测量值必须大于0！")

    @staticmethod
    def _eta_formula(d, t, L, D, H, rho, rhop, g=9.788):
        """落球法黏滞系数核心公式（Pa·s）"""
        A = 1 + 2.4 * d / D  # 容器内径修正项
        B = 1 + 1.6 * d / H  # 液柱高度修正项
        numerator = (rho - rhop) * g * (d ** 2) * t
        denominator = 18 * L * A * B
        return numerator / denominator

    def _compute_A_uncertainty(self, data):
        """计算多次测量平均值的A类标准不确定度（样本标准差 / √n）"""
        n = len(data)
        if n < 2:
            return 0.0
        std = np.std(data, ddof=1)  # 样本标准差（自由度n-1）
        return std / np.sqrt(n)

    def _compute_B_micrometer(self):
        """千分尺B类标准不确定度（正态分布，C=3）"""
        return self.microm_half / self.C_normal

    def _compute_B_ruler(self):
        """米尺B类标准不确定度（正态分布，C=3）"""
        return self.ruler_half / self.C_normal

    def _compute_B_time(self):
        """秒表B类标准不确定度（正态分布，C=3）"""
        return self.t_half / self.C_normal

    def _compute_B_rhop(self):
        """油密度的B类不确定度（比重计，均匀分布C=√3）"""
        return self.rhop_half / self.C_uniform

    def calculate(self):
        """执行所有计算，结果存入实例属性（核心：解析偏导计算灵敏系数）"""
        # 1. 千分尺零点误差修正
        self.d_meas_corrected = self.d_meas_raw - self.d0
        self.t_mean = np.mean(self.t_meas)
        self.d_mean = np.mean(self.d_meas_corrected)  # 修正后直径均值

        # 2. 黏滞系数均值计算
        self.eta_mean = self._eta_formula(
            self.d_mean, self.t_mean, self.L, self.D, self.H,
            self.rho, self.rhop, self.g
        )

        # 3. 辅助项计算（用于偏导数）
        A = 1 + 2.4 * self.d_mean / self.D  # 容器内径修正项
        B = 1 + 1.6 * self.d_mean / self.H  # 液柱高度修正项
        delta_rho = self.rho - self.rhop    # 密度差

        # 4. 解析偏导数（灵敏系数）计算（严格无近似）
        # 参数顺序：[d, t, L, D, H, rho, rhop]
        self.c = [
            # 1. 对d的偏导：η*(2/d - 2.4/(D*A) - 1.6/(H*B))
            self.eta_mean * (2 / self.d_mean - 2.4/(self.D*A) - 1.6/(self.H*B)),
            # 2. 对t的偏导：η/t
            self.eta_mean / self.t_mean,
            # 3. 对L的偏导：-η/L
            -self.eta_mean / self.L,
            # 4. 对D的偏导：η*(2.4d)/(D²*A)
            self.eta_mean * (2.4 * self.d_mean) / (self.D**2 * A),
            # 5. 对H的偏导：η*(1.6d)/(H²*B)
            self.eta_mean * (1.6 * self.d_mean) / (self.H**2 * B),
            # 6. 对rho的偏导：η/(ρ-ρ')
            self.eta_mean / delta_rho,
            # 7. 对rhop的偏导：-η/(ρ-ρ')
            -self.eta_mean / delta_rho
        ]

        # 5. 不确定度分量计算
        # A类（统计误差）
        self.u_d_A = self._compute_A_uncertainty(self.d_meas_corrected)  # 直径A类
        self.u_t_A = self._compute_A_uncertainty(self.t_meas)            # 时间A类

        # B类（仪器误差）
        self.u_d_B = self._compute_B_micrometer()  # 直径B类（千分尺）
        self.u_L_B = self._compute_B_ruler()       # L的B类（米尺）
        self.u_D_B = self._compute_B_ruler()       # D的B类（米尺）
        self.u_H_B = self._compute_B_ruler()       # H的B类（米尺）
        self.u_t_B = self._compute_B_time()        # 时间B类（秒表）
        self.u_rhop = self._compute_B_rhop()       # 油密度B类（比重计）

        # 合成：直径/时间的总不确定度（A类+B类方和根）
        self.u_d = np.sqrt(self.u_d_A**2 + self.u_d_B**2)
        self.u_t = np.sqrt(self.u_t_A**2 + self.u_t_B**2)

        # 小球密度：假设相对不确定度0.1%（正态分布）
        self.u_rho = self.rho * 0.001

        # 6. 黏滞系数合成标准不确定度（方和根）
        u_vals = [self.u_d, self.u_t, self.u_L_B, self.u_D_B,
                  self.u_H_B, self.u_rho, self.u_rhop]
        self.u_eta = np.sqrt(sum((self.c[i] * u_vals[i])**2 for i in range(7)))

        # 相对不确定度
        self.rel_uncertainty = self.u_eta / self.eta_mean

    def print_results(self, group_name):
        """打印计算结果（含灵敏系数公式说明）"""
        print(f"\n==================== {group_name} 计算结果 ====================")
        print(f"=== 实验参数（原始单位）===")
        print(f"油密度ρ' = {ExperimentConfig.rhop} g/cm³ | 小球密度ρ = {ExperimentConfig.rho} g/cm³")
        print(f"下落距离L = {ExperimentConfig.L} cm | 液柱高度H = {ExperimentConfig.H} cm | 容器内径D = {ExperimentConfig.D} cm")
        print(f"千分尺零点误差d0 = {ExperimentConfig.d0} cm | 温度T1={ExperimentConfig.T1}℃, T2={ExperimentConfig.T2}℃")
        
        print(f"\n=== 数据修正 ===")
        print(f"千分尺零点误差修正：d_修正 = d_测量 - {self.d0:.6f} m")
        print(f"修正前直径均值：{np.mean(self.d_meas_raw):.6e} m")
        print(f"修正后直径均值：{self.d_mean:.6e} m")
        
        print(f"\n=== 黏滞系数计算结果 ===")
        print(f"黏滞系数 η = {self.eta_mean:.3e} Pa·s")
        print(f"合成标准不确定度 u(η) = {self.u_eta:.3e} Pa·s")
        print(f"相对不确定度 = {self.rel_uncertainty*100:.2f}%")
        
        print("\n=== 各参数灵敏系数（解析偏导）===")
        names = ['d', 't', 'L', 'D', 'H', 'rho', 'rhop']
        formulas = [
            "η·(2/d - 2.4/(D·A) - 1.6/(H·B))",
            "η/t",
            "-η/L",
            "η·(2.4d)/(D²·A)",
            "η·(1.6d)/(H²·B)",
            "η/(ρ-ρ')",
            "-η/(ρ-ρ')"
        ]
        for i, (name, formula) in enumerate(zip(names, formulas)):
            print(f"  ∂η/∂{name} = {formula} = {self.c[i]:.2e} Pa·s/单位")
        
        print("\n=== 各不确定度分量（绝对值）===")
        contrib_names = ['d(总)', 't(总)', 'L', 'D', 'H', 'rho', 'rhop(比重计)']
        u_vals = [self.u_d, self.u_t, self.u_L_B, self.u_D_B,
                  self.u_H_B, self.u_rho, self.u_rhop]
        for i, name in enumerate(contrib_names):
            contrib = self.c[i] * u_vals[i]
            print(f"  {name}: {contrib:.3e} Pa·s")
        print("="*60)

# ============================ 主函数（无需修改）============================
def main():
    # 1. 读取配置并转换为国际单位制
    si_params = ExperimentConfig.convert_to_si()
    converted_groups = ExperimentConfig.get_converted_measure_groups()

    # 2. 循环处理所有测量组
    for group_name, data in converted_groups.items():
        try:
            # 初始化计算器（传入转换后的国际单位参数）
            calculator = ViscosityUncertaintyCalculator(
                d_measurements=data["d_m"],
                t_measurements=data["t_s"],
                L=si_params["L"],
                D=si_params["D"],
                H=si_params["H"],
                rho=si_params["rho"],
                rhop=si_params["rhop"],
                g=si_params["g"],
                micrometer_half_div=si_params["delta_L_microm"],
                ruler_half_div=si_params["delta_L_ruler"],
                micrometer_zero_error=si_params["d0"],
                t_half_div=si_params["delta_t"],
                rhop_half_div=si_params["delta_rhop"] # pyright: ignore[reportArgumentType]
            )
            # 打印该组结果
            calculator.print_results(group_name)
        except ValueError as e:
            print(f"\n【错误】{group_name} 计算失败：{e}")

if __name__ == "__main__":
    main()