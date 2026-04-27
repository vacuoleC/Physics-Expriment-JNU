"""
惠斯通电桥测电阻实验 - 数据处理脚本
功能：计算待测电阻、A类/B类不确定度、合成不确定度、扩展不确定度、电桥灵敏度
"""

import math
import sys
import io

# 设置标准输出编码为UTF-8，解决Windows控制台中文乱码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============================================================
# 1. 实验原始数据
# ============================================================

# 1.1 自组电桥（交换法测量）
self_assembled_data = [
    {"Rx_rough": 1.1213e3, "R1": 1000.0, "R2": 1000.0, "Rs": 1118.7, "Rs_prime": 1118.3},
    {"Rx_rough": 1.1213e3, "R1": 1000.0, "R2": 500.0, "Rs": 559.4, "Rs_prime": 2236.0},
]

# 1.2 箱式电桥（交换法测量）
box_type_data = [
    {"Rx_rough": 1.0013e3, "K": 1, "Rs": 998.0, "Rs_prime": 998.2, "delta_n": 5, "Rs_after": 999.62},
    {"Rx_rough": 1.0013e3, "K": 10, "Rs": 99.80, "Rs_prime": 9982.0, "delta_n": 5, "Rs_after": 1002.70},
]

# 1.3 仪器参数
a_percent = 0.1  # 准确度等级 0.1%
b_values = {
    "x10k": 0.1,
    "x1k": 0.1,
    "x100": 0.1,
    "x10": 0.1,
    "x1": 0.5,
    "x0.1": 2.0,
}
k_uniform = math.sqrt(3)  # 均匀分布置信因子
k_expansion = 2  # 扩展不确定度置信因子 (P=95%)


# ============================================================
# 2. 辅助函数
# ============================================================

def calculate_Rx(Rs, Rs_prime):
    """交换法计算待测电阻 Rx = sqrt(Rs * Rs')"""
    return math.sqrt(Rs * Rs_prime)


def count_digits(R):
    """统计电阻箱读数使用的转盘个数m（按有效数字位数统计）"""
    # 将电阻值转换为字符串，统计有效数字位数
    # 电阻箱读数通常为整数或一位小数
    if R == 0:
        return 1
    s = f"{R:g}"  # 去除末尾多余的0
    # 统计数字个数（不包括小数点）
    m = 0
    for c in s:
        if c.isdigit():
            m += 1
    return m


def get_b_value(R):
    """根据电阻值大小确定b值（对应不同倍率档）"""
    # 根据电阻值范围判断使用的倍率档
    if R >= 10000:
        return b_values["x10k"]
    elif R >= 1000:
        return b_values["x1k"]
    elif R >= 100:
        return b_values["x100"]
    elif R >= 10:
        return b_values["x10"]
    elif R >= 1:
        return b_values["x1"]
    else:
        return b_values["x0.1"]


def calculate_delta_R(R):
    """计算电阻箱仪器绝对误差 ΔR = a% * R + b * m"""
    m = count_digits(R)
    b = get_b_value(R)
    delta_R = a_percent / 100 * R + b * m
    return delta_R


def calculate_u_R(R):
    """计算单个电阻箱读数的标准不确定度 u(R) = ΔR / sqrt(3)"""
    delta_R = calculate_delta_R(R)
    return delta_R / k_uniform


def round_uncertainty(u):
    """不确定度修约：只进不舍，保留1位有效数字；若首位为1或2，保留2位"""
    if u == 0:
        return 0
    # 计算数量级
    order = math.floor(math.log10(u))
    # 归一化到1-10之间
    normalized = u / (10 ** order)
    
    # 判断首位数字
    first_digit = int(normalized)
    if first_digit == 1 or first_digit == 2:
        # 保留2位有效数字，只进不舍
        # 例如：1.3364 -> 1.4 (进位)
        shifted = normalized * 10  # 1.3364 -> 13.364
        ceil_val = math.ceil(shifted)  # 14
        rounded = ceil_val / 10  # 1.4
        return rounded * (10 ** order)
    else:
        # 保留1位有效数字，只进不舍
        # 例如：3.52 -> 4
        rounded = math.ceil(normalized)
        return rounded * (10 ** order)


def get_decimal_places(uncertainty):
    """根据不确定度获取需要保留的小数位数"""
    if uncertainty == 0:
        return 0
    order = math.floor(math.log10(uncertainty))
    # 如果order >= 0，说明不确定度 >= 1，小数位数为0
    # 如果order < 0，说明不确定度 < 1，需要保留小数
    # 但需要考虑2位有效数字的情况
    # 例如：1.4 -> 0位小数, 0.14 -> 2位小数, 0.014 -> 3位小数
    first_digit = int(uncertainty / (10 ** order))
    if first_digit == 1 or first_digit == 2:
        # 保留2位有效数字
        return max(0, 1 - order)
    else:
        # 保留1位有效数字
        return max(0, -order)


def round_value_with_uncertainty(value, uncertainty):
    """测量值修约：与不确定度的末位对齐"""
    if uncertainty == 0:
        return value
    decimals = get_decimal_places(uncertainty)
    factor = 10 ** decimals
    rounded_value = round(value * factor) / factor
    return rounded_value


def format_result(value, uncertainty, unit="Ω"):
    """格式化输出结果"""
    if uncertainty == 0:
        return f"{value:.4g} {unit}"
    
    # 确定小数位数
    decimals = get_decimal_places(uncertainty)
    
    # 格式化输出
    return f"({value:.{decimals}f} ± {uncertainty:.{decimals}f}) {unit}"


# ============================================================
# 3. 主计算流程
# ============================================================

def main():
    print("=" * 70)
    print("惠斯通电桥测电阻实验 - 数据处理结果")
    print("=" * 70)
    
    # --------------------------------------------------------
    # 3.1 自组电桥数据处理
    # --------------------------------------------------------
    print("\n" + "-" * 50)
    print("【一】自组电桥（交换法）")
    print("-" * 50)
    
    Rx_values_self = []
    for i, data in enumerate(self_assembled_data, 1):
        Rx = calculate_Rx(data["Rs"], data["Rs_prime"])
        Rx_values_self.append(Rx)
        print(f"第{i}组: Rs = {data['Rs']} Ω, Rs' = {data['Rs_prime']} Ω")
        print(f"        Rx{i} = √({data['Rs']} × {data['Rs_prime']}) = {Rx:.4f} Ω")
    
    # 计算平均值
    n_self = len(Rx_values_self)
    Rx_mean_self = sum(Rx_values_self) / n_self
    print(f"\n自组电桥平均值: Rx_mean = {Rx_mean_self:.4f} Ω")
    
    # A类不确定度
    if n_self > 1:
        sum_sq_diff = sum((Rx - Rx_mean_self) ** 2 for Rx in Rx_values_self)
        uA_self = math.sqrt(sum_sq_diff / (n_self * (n_self - 1)))
    else:
        uA_self = 0
    print(f"A类不确定度: uA(Rx_mean) = {uA_self:.6f} Ω")
    
    # B类不确定度（取各组B类不确定度的平均值）
    uB_values_self = []
    for i, data in enumerate(self_assembled_data, 1):
        u_Rs = calculate_u_R(data["Rs"])
        u_Rs_prime = calculate_u_R(data["Rs_prime"])
        Rx_i = Rx_values_self[i - 1]
        uB_i = Rx_i / 2 * math.sqrt((u_Rs / data["Rs"]) ** 2 + (u_Rs_prime / data["Rs_prime"]) ** 2)
        uB_values_self.append(uB_i)
        print(f"第{i}组B类: u(Rs) = {u_Rs:.4f} Ω, u(Rs') = {u_Rs_prime:.4f} Ω, uB(Rx{i}) = {uB_i:.4f} Ω")
    
    uB_self = sum(uB_values_self) / len(uB_values_self)
    print(f"平均B类不确定度: uB(Rx_mean) = {uB_self:.4f} Ω")
    
    # 合成不确定度
    uc_self = math.sqrt(uA_self ** 2 + uB_self ** 2)
    print(f"合成不确定度: uc(Rx_mean) = {uc_self:.4f} Ω")
    
    # 扩展不确定度
    U_self = k_expansion * uc_self
    print(f"扩展不确定度: U(Rx_mean) = {U_self:.4f} Ω (P=95%)")
    
    # 修约
    U_self_rounded = round_uncertainty(U_self)
    Rx_self_rounded = round_value_with_uncertainty(Rx_mean_self, U_self_rounded)
    print(f"\n【自组电桥最终结果】")
    print(f"  Rx = {format_result(Rx_self_rounded, U_self_rounded)} (P=95%)")
    
    # --------------------------------------------------------
    # 3.2 箱式电桥数据处理
    # --------------------------------------------------------
    print("\n" + "-" * 50)
    print("【二】箱式电桥（交换法）")
    print("-" * 50)
    
    Rx_values_box = []
    for i, data in enumerate(box_type_data, 1):
        Rx = calculate_Rx(data["Rs"], data["Rs_prime"])
        Rx_values_box.append(Rx)
        print(f"第{i}组 (K={data['K']}): Rs = {data['Rs']} Ω, Rs' = {data['Rs_prime']} Ω")
        print(f"        Rx{i} = √({data['Rs']} × {data['Rs_prime']}) = {Rx:.4f} Ω")
    
    # 计算平均值
    n_box = len(Rx_values_box)
    Rx_mean_box = sum(Rx_values_box) / n_box
    print(f"\n箱式电桥平均值: Rx_mean = {Rx_mean_box:.4f} Ω")
    
    # A类不确定度
    if n_box > 1:
        sum_sq_diff_box = sum((Rx - Rx_mean_box) ** 2 for Rx in Rx_values_box)
        uA_box = math.sqrt(sum_sq_diff_box / (n_box * (n_box - 1)))
    else:
        uA_box = 0
    print(f"A类不确定度: uA(Rx_mean) = {uA_box:.6f} Ω")
    
    # B类不确定度
    uB_values_box = []
    for i, data in enumerate(box_type_data, 1):
        u_Rs = calculate_u_R(data["Rs"])
        u_Rs_prime = calculate_u_R(data["Rs_prime"])
        Rx_i = Rx_values_box[i - 1]
        uB_i = Rx_i / 2 * math.sqrt((u_Rs / data["Rs"]) ** 2 + (u_Rs_prime / data["Rs_prime"]) ** 2)
        uB_values_box.append(uB_i)
        print(f"第{i}组B类: u(Rs) = {u_Rs:.4f} Ω, u(Rs') = {u_Rs_prime:.4f} Ω, uB(Rx{i}) = {uB_i:.4f} Ω")
    
    uB_box = sum(uB_values_box) / len(uB_values_box)
    print(f"平均B类不确定度: uB(Rx_mean) = {uB_box:.4f} Ω")
    
    # 合成不确定度
    uc_box = math.sqrt(uA_box ** 2 + uB_box ** 2)
    print(f"合成不确定度: uc(Rx_mean) = {uc_box:.4f} Ω")
    
    # 扩展不确定度
    U_box = k_expansion * uc_box
    print(f"扩展不确定度: U(Rx_mean) = {U_box:.4f} Ω (P=95%)")
    
    # 修约
    U_box_rounded = round_uncertainty(U_box)
    Rx_box_rounded = round_value_with_uncertainty(Rx_mean_box, U_box_rounded)
    print(f"\n【箱式电桥最终结果】")
    print(f"  Rx = {format_result(Rx_box_rounded, U_box_rounded)} (P=95%)")
    
    # --------------------------------------------------------
    # 3.3 电桥灵敏度计算
    # --------------------------------------------------------
    print("\n" + "-" * 50)
    print("【三】电桥灵敏度计算")
    print("-" * 50)
    
    for i, data in enumerate(box_type_data, 1):
        Rx_i = Rx_values_box[i - 1]
        delta_n = data["delta_n"]
        Rs_after = data["Rs_after"]
        Rs = data["Rs"]
        delta_R = abs(Rs_after - Rs)
        
        # 灵敏度 S = delta_n * Rx / |Rs' - Rs|
        # 注意：这里的Rs'是偏转后的电阻值，不是交换法的Rs'
        S = delta_n * Rx_i / delta_R
        print(f"第{i}组 (K={data['K']}):")
        print(f"  Δn = {delta_n} 格, Rs = {Rs} Ω, 偏转后Rs' = {Rs_after} Ω")
        print(f"  ΔR = |{Rs_after} - {Rs}| = {delta_R:.2f} Ω")
        print(f"  S = {delta_n} × {Rx_i:.2f} / {delta_R:.2f} = {S:.2f} 格")
        
        # 修约：保留2位有效数字
        S_rounded = round(S, -int(math.floor(math.log10(S))) + 2) if S > 0 else 0
        print(f"  灵敏度 S ≈ {S_rounded:.2g} 格")
    
    # --------------------------------------------------------
    # 3.4 总结合并结果
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("【最终实验结果汇总】")
    print("=" * 70)
    print(f"自组电桥: Rx = {format_result(Rx_self_rounded, U_self_rounded)} (P=95%)")
    print(f"箱式电桥: Rx = {format_result(Rx_box_rounded, U_box_rounded)} (P=95%)")
    print("=" * 70)


if __name__ == "__main__":
    main()