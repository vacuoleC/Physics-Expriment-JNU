import numpy as np

class ViscosityUncertaintyCalculator:
    """
    落球法测量液体黏滞系数的不确定度计算器（适配自定义误差参数）
    """

    """
    请注意：一定要使用国际单位制    
    请注意：一定要使用国际单位制
    请注意：一定要使用国际单位制
    请注意：一定要使用国际单位制
    请注意：一定要使用国际单位制
    请注意：一定要使用国际单位制
    请注意：一定要使用国际单位制
    请注意：一定要使用国际单位制
    请注意：一定要使用国际单位制
    请注意：一定要使用国际单位制
    请注意：一定要使用国际单位制
    请注意：一定要使用国际单位制
    请注意：一定要使用国际单位制
    请注意：一定要使用国际单位制
    请注意：一定要使用国际单位制
    请注意：一定要使用国际单位制
    请注意：一定要使用国际单位制
    请注意：一定要使用国际单位制
    请注意：一定要使用国际单位制
    请注意：一定要使用国际单位制
    请注意：一定要使用国际单位制
    请注意：一定要使用国际单位制
    请注意：一定要使用国际单位制
    请注意：一定要使用国际单位制
    请注意：一定要使用国际单位制
    请注意：一定要使用国际单位制
    """

    def __init__(
            self, 
            d_measurements, 
            t_measurements, 
            L, 
            D, 
            H, 
            rho, 
            rhop,
            T1, 
            T2,
            g=9.788,
            # 新增/修改的误差参数
            micrometer_half_div=0.00004,  # 千分尺半宽（m）
            ruler_half_div=0.0015,        # 普通尺半宽（m）
            micrometer_zero_error=0.0005, # 千分尺零点误差（m）
            t_half_div=0.01,              # 时间测量半宽（s）
            rhop_half_div=3,              # 比重计半宽（kg/m³）
            temp_uncertainty_half=0.1):   # 温度计精度半宽（°C）
        """
        参数
        ----------
        d_measurements : list of float
            小球直径的多次测量值 (m)（未修正零点误差）
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
        g : float, optional
            重力加速度 (m/s²)
        micrometer_half_div : float, optional
            千分尺最小分度的一半 (m)（小球直径B类误差）
        ruler_half_div : float, optional
            普通尺最小分度的一半 (m)（L/D/H的B类误差）
        micrometer_zero_error : float, optional
            千分尺零点误差 (m)（修正小球直径测量值）
        t_half_div : float, optional
            时间测量仪器半宽 (s)（时间B类误差）
        rhop_half_div : float, optional
            比重计测量半宽 (kg/m³)（液体密度B类误差）
        temp_uncertainty_half : float, optional
            温度计精度半宽 (°C)
        """
        # 原始数据
        self.d_meas_raw = np.array(d_measurements)
        self.t_meas = np.array(t_measurements)
        self.L = L
        self.D = D
        self.H = H
        self.rho = rho
        self.rhop = rhop
        self.T1 = T1
        self.T2 = T2
        self.g = g
        
        # 误差参数（细分）
        self.microm_half = micrometer_half_div  # 千分尺半宽
        self.ruler_half = ruler_half_div        # 普通尺半宽
        self.d0 = micrometer_zero_error         # 零点误差
        self.t_half = t_half_div                # 时间半宽
        self.rhop_half = rhop_half_div          # 比重计半宽
        self.temp_half = temp_uncertainty_half  # 温度半宽

        # 计算结果（初始化时自动计算）
        self.calculate()

    @staticmethod
    def _eta_formula(d, t, L, D, H, rho, rhop, g=9.8):
        """给定各参数，计算一次黏滞系数 η (Pa·s)"""
        A = (rho - rhop) * g / (18 * L)
        B = d**2 * t
        C = 1 / ((1 + 2.4 * d / D) * (1 + 1.6 * d / H))
        return A * B * C


    def _compute_A_uncertainty(self, data):
        """计算多次测量平均值的A类标准不确定度（样本标准差 / sqrt(n)）"""
        n = len(data)
        if n < 2:
            return 0.0
        std = np.std(data, ddof=1)
        return std / np.sqrt(n)

    def _compute_B_micrometer(self):
        """千分尺单次测量的B类标准不确定度（均匀分布）"""
        return self.microm_half / np.sqrt(3)

    def _compute_B_ruler(self):
        """普通尺单次测量的B类标准不确定度（均匀分布）"""
        return self.ruler_half / np.sqrt(3)

    def _compute_B_time(self):
        """时间单次测量的B类标准不确定度（均匀分布）"""
        return self.t_half / np.sqrt(3)

    def _compute_B_rhop(self):
        """液体密度的B类不确定度（比重计）"""
        # 比重计引起的rhop不确定度
        u_rhop_balance = self.rhop_half / np.sqrt(3)
        # 3. 合成
        return u_rhop_balance

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


    def calculate(self):
        """执行所有计算，并将结果存储在实例属性中"""
        # --- 1. 原始数据修正（千分尺零点误差）---
        self.d_meas_corrected = self.d_meas_raw + self.d0  # 修正零点误差
        self.t_mean = np.mean(self.t_meas)

        # --- 2. 平均值计算 ---
        self.d_mean = np.mean(self.d_meas_corrected)  # 修正后的直径均值
        T_avg = (self.T1 + self.T2) / 2
        self.T_avg = T_avg

        # --- 3. 黏滞系数均值 ---
        self.eta_mean = self._eta_formula(self.d_mean, self.t_mean,
                                          self.L, self.D, self.H,
                                          self.rho, self.rhop, self.g)

        # --- 4. 不确定度分量（标准不确定度）---
        # A类
        self.u_d_A = self._compute_A_uncertainty(self.d_meas_corrected)  # 直径A类
        self.u_t_A = self._compute_A_uncertainty(self.t_meas)            # 时间A类

        # B类（长度）
        self.u_d_B = self._compute_B_micrometer()  # 直径B类（千分尺）
        self.u_L_B = self._compute_B_ruler()       # L的B类（普通尺）
        self.u_D_B = self._compute_B_ruler()       # D的B类（普通尺）
        self.u_H_B = self._compute_B_ruler()       # H的B类（普通尺）
        
        # B类（时间：A类+B类合成）
        self.u_t_B = self._compute_B_time()
        self.u_t = np.sqrt(self.u_t_A**2 + self.u_t_B**2)  # 时间总不确定度

        # B类（直径：A类+B类合成）
        self.u_d = np.sqrt(self.u_d_A**2 + self.u_d_B**2)  # 直径总不确定度

        # 小球密度（假设相对不确定度 0.1%）
        self.u_rho = self.rho * 0.001

        # 液体密度（温度+比重计合成）
        self.u_rhop = self._compute_B_rhop()

        # --- 5. 灵敏系数（数值差分）---
        # 参数顺序: [d, t, L, D, H, rho, rhop]
        params_nominal = [self.d_mean, self.t_mean, self.L, self.D,
                          self.H, self.rho, self.rhop]
        # 扰动步长（根据各量数量级选取）
        steps = [1e-6, 1e-4, 1e-4, 1e-4, 1e-4, 1.0, 1.0]

        self.c = []
        for i in range(len(params_nominal)):
            ci = self._sensitivity(params_nominal, i, steps[i])
            self.c.append(ci)

        # --- 6. 合成标准不确定度 ---
        u_vals = [self.u_d, self.u_t, self.u_L_B, self.u_D_B,
                  self.u_H_B, self.u_rho, self.u_rhop]
        self.u_eta = np.sqrt(sum((self.c[i] * u_vals[i])**2
                                 for i in range(len(u_vals))))

        # 相对不确定度
        self.rel_uncertainty = self.u_eta / self.eta_mean


    def print_results(self):
        """打印计算结果"""
        print(f"=== 数据修正 ===")
        print(f"千分尺零点误差修正：d_修正 = d_测量 + {self.d0:.6f} m")
        print(f"修正前直径均值：{np.mean(self.d_meas_raw):.6e} m")
        print(f"修正后直径均值：{self.d_mean:.6e} m")
        print(f"\n=== 黏滞系数计算结果 ===")
        print(f"黏滞系数 η = {self.eta_mean:.6e} Pa·s")
        print(f"合成标准不确定度 u(η) = {self.u_eta:.6e} Pa·s")
        print(f"相对不确定度 = {self.rel_uncertainty*100:.2f}%")
        print("\n=== 各不确定度分量（绝对值）===")
        names = ['d(总)', 't(总)', 'L', 'D', 'H', 'rho', 'rhop(比重计)']
        u_vals = [self.u_d, self.u_t, self.u_L_B, self.u_D_B,
                  self.u_H_B, self.u_rho, self.u_rhop]
        for i, name in enumerate(names):
            contrib = self.c[i] * u_vals[i]
            print(f"  {name}: {contrib:.6e} Pa·s (灵敏系数：{self.c[i]:.2e})")


# ============================ 使用示例 ============================
def main():

    #小球组
    d_small = [
        0.1952,
        0.1947,
        0.1910,
        0.1910,
        0.1948,
    ]

    t_small = [
        9.03,
        8.91,
        8.87,
        8.81,
        8.75,
    ]

    #大球组
    d_large = [
        0.2941,
        0.2943,
        0.2952,
        0.2952,
        0.2948,
    ]

    t_large = [
        4.15,
        4.19,
        4.28,
        4.21,
        4.22,
    ]

    print("=== 小球组 ===")
    calc1 = ViscosityUncertaintyCalculator(
        d_measurements=d_small,
        t_measurements=t_small,
        L=0.15,          
        D=0.0223,         
        H=0.341,          
        rho=7.76 * 10e3,       
        T1=23.7,
        T2=23.8,
        rhop=0.956 * 10e3, 
        micrometer_half_div=0.00004,    
        ruler_half_div=0.0015,          
        micrometer_zero_error=0.0005,   
        t_half_div=0.01,                
        rhop_half_div=3,               
    )
    calc1.print_results()

    print("\n=== 大球组 ===")
    calc2 = ViscosityUncertaintyCalculator(
        d_measurements=d_large,
        t_measurements=t_large,
        L=0.15,
        D=0.0223,
        H=0.341,
        rho=7.76 * 10e3,
        T1=23.7,
        T2=23.8,
        rhop=0.956 * 10e3,
        micrometer_half_div=0.00004,
        ruler_half_div=0.0015,
        micrometer_zero_error=0.0005,
        t_half_div=0.01,
        rhop_half_div=3,
    )
    calc2.print_results()

if __name__ == "__main__":
    main()