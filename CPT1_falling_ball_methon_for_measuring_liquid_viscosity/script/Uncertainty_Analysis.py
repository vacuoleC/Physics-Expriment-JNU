import numpy as np

class ViscosityUncertaintyCalculator:
    """
    落球法测量液体黏滞系数的不确定度计算器
    """
    def __init__(
            self, 
            d_measurements, 
            t_measurements, 
            L, 
            D, 
            H, 
            rho, 
            T1, 
            T2,
            rhop_ref, 
            T_ref, 
            alpha=-0.2, 
            g=9.8,
            instrument_half_div=0.0005, 
            temp_uncertainty_half=0.1):
        """
        参数
        ----------
        d_measurements : list of float
            小球直径的多次测量值 (m)
        t_measurements : list of float
            下落时间的多次测量值 (s)
        L : float
            下落距离 (m)
        D : float
            容器内径 (m)
        H : float
            液柱高度 (m)
        rho : float
            小球密度 (kg/m³)
        T1, T2 : float
            两次温度测量值 (°C)
        rhop_ref : float, optional
            参考温度下的液体密度 (kg/m³)
        T_ref : float, optional
            参考温度 (°C)
        alpha : float, optional
            液体密度的温度系数 (kg/(m³·°C))
        g : float, optional
            重力加速度 (m/s²)
        instrument_half_div : float, optional
            长度测量仪器最小分度的一半 (m)，用于B类不确定度
        temp_uncertainty_half : float, optional
            温度计精度半宽 (°C)
        """
        # 原始数据
        self.d_meas = np.array(d_measurements)
        self.t_meas = np.array(t_measurements)
        self.L = L
        self.D = D
        self.H = H
        self.rho = rho
        self.T1 = T1
        self.T2 = T2
        self.rhop_ref = rhop_ref
        self.T_ref = T_ref
        self.alpha = alpha
        self.g = g
        self.instr_half = instrument_half_div
        self.temp_half = temp_uncertainty_half

        # 计算结果（初始化时自动计算）
        self.calculate()

    # ------------------------------------------------------------------
    # 核心计算公式
    # ------------------------------------------------------------------
    @staticmethod
    def _eta_formula(d, t, L, D, H, rho, rhop, g=9.8):
        """给定各参数，计算一次黏滞系数 η (Pa·s)"""
        A = (rho - rhop) * g / (18 * L)
        B = d**2 * t
        C = 1 / ((1 + 2.4 * d / D) * (1 + 1.6 * d / H))
        return A * B * C

    def _compute_rhop(self, T_avg):
        """根据平均温度计算液体密度"""
        return self.rhop_ref + self.alpha * (T_avg - self.T_ref)

    # ------------------------------------------------------------------
    # 不确定度评定（内部方法）
    # ------------------------------------------------------------------
    def _compute_A_uncertainty(self, data):
        """计算多次测量平均值的A类标准不确定度（样本标准差 / sqrt(n)）"""
        n = len(data)
        if n < 2:
            return 0.0
        std = np.std(data, ddof=1)
        return std / np.sqrt(n)

    def _compute_B_length(self):
        """长度单次测量的B类标准不确定度（均匀分布）"""
        return self.instr_half / np.sqrt(3)

    def _compute_B_temperature(self):
        """平均温度的B类标准不确定度（两次平均，均匀分布）"""
        u_T_single = self.temp_half / np.sqrt(3)
        return u_T_single / np.sqrt(2)   # 两次平均

    def _sensitivity(self, params, idx, delta):
        """用中心差分法求灵敏系数"""
        params_plus = params.copy()
        params_plus[idx] += delta
        params_minus = params.copy()
        params_minus[idx] -= delta
        eta_plus = self._eta_formula(*params_plus, g=self.g)
        eta_minus = self._eta_formula(*params_minus, g=self.g)
        return (eta_plus - eta_minus) / (2 * delta)

    # ------------------------------------------------------------------
    # 主计算流程
    # ------------------------------------------------------------------
    def calculate(self):
        """执行所有计算，并将结果存储在实例属性中"""
        # --- 平均值 ---
        self.d_mean = np.mean(self.d_meas)
        self.t_mean = np.mean(self.t_meas)

        # 平均温度及液体密度
        T_avg = (self.T1 + self.T2) / 2
        self.T_avg = T_avg
        self.rhop = self._compute_rhop(T_avg)

        # 计算 η 的平均值
        self.eta_mean = self._eta_formula(self.d_mean, self.t_mean,
                                          self.L, self.D, self.H,
                                          self.rho, self.rhop, self.g)

        # --- 不确定度分量（标准不确定度）---
        # A类
        self.u_d = self._compute_A_uncertainty(self.d_meas)
        self.u_t = self._compute_A_uncertainty(self.t_meas)

        # B类（长度）
        self.u_L = self._compute_B_length()
        self.u_D = self._compute_B_length()
        self.u_H = self._compute_B_length()

        # 小球密度（假设相对不确定度 0.1%）
        self.u_rho = self.rho * 0.001

        # 温度及液体密度
        self.u_T = self._compute_B_temperature()
        self.u_rhop = abs(self.alpha) * self.u_T

        # --- 灵敏系数（数值差分）---
        # 参数顺序: [d, t, L, D, H, rho, rhop]
        params_nominal = [self.d_mean, self.t_mean, self.L, self.D,
                          self.H, self.rho, self.rhop]
        # 扰动步长（根据各量数量级选取）
        steps = [1e-6, 1e-4, 1e-4, 1e-4, 1e-4, 1.0, 1.0]

        self.c = []
        for i in range(len(params_nominal)):
            ci = self._sensitivity(params_nominal, i, steps[i])
            self.c.append(ci)

        # 合成标准不确定度
        u_vals = [self.u_d, self.u_t, self.u_L, self.u_D,
                  self.u_H, self.u_rho, self.u_rhop]
        self.u_eta = np.sqrt(sum((self.c[i] * u_vals[i])**2
                                 for i in range(len(u_vals))))

        # 相对不确定度
        self.rel_uncertainty = self.u_eta / self.eta_mean

    # ------------------------------------------------------------------
    # 结果输出
    # ------------------------------------------------------------------
    def print_results(self):
        """打印计算结果"""
        print(f"黏滞系数 η = {self.eta_mean:.6e} Pa·s")
        print(f"合成标准不确定度 u(η) = {self.u_eta:.6e} Pa·s")
        print(f"相对不确定度 = {self.rel_uncertainty*100:.2f}%")
        print("\n各不确定度分量（绝对值）：")
        names = ['d', 't', 'L', 'D', 'H', 'rho', 'rhop']
        u_vals = [self.u_d, self.u_t, self.u_L, self.u_D,
                  self.u_H, self.u_rho, self.u_rhop]
        for i, name in enumerate(names):
            contrib = self.c[i] * u_vals[i]
            print(f"  {name}: {contrib:.2e} Pa·s")

    def get_results(self):
        """返回包含所有结果的字典"""
        return {
            'eta_mean': self.eta_mean,
            'u_eta': self.u_eta,
            'rel_uncertainty': self.rel_uncertainty,
            'd_mean': self.d_mean,
            't_mean': self.t_mean,
            'rhop': self.rhop,
            'T_avg': self.T_avg,
            'u_d': self.u_d,
            'u_t': self.u_t,
            'u_L': self.u_L,
            'u_D': self.u_D,
            'u_H': self.u_H,
            'u_rho': self.u_rho,
            'u_rhop': self.u_rhop,
            'c': self.c,
        }


# ============================ 使用示例 ============================
def main():
    # 实验组1
    calc1 = ViscosityUncertaintyCalculator(
        d_measurements=[0.001002, 0.001001, 0.001003, 0.001002, 0.001001],
        t_measurements=[12.5, 12.6, 12.4, 12.5, 12.5],
        L=0.2,
        D=0.05,
        H=0.3,
        rho=7800,
        T1=25.0,
        T2=25.1,
        rhop_ref=998.0,
        T_ref=25.0,
    )
    print("=== 实验组1 ===")
    calc1.print_results()
    print()



if __name__ == "__main__":
    main()