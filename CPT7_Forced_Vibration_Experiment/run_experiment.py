import os
import sys
import subprocess

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

# ===================== 路径配置（自动适配你的项目结构）=====================
# 获取当前脚本所在的根目录（项目根目录）
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# 脚本文件夹路径
SCRIPT_DIR = os.path.join(ROOT_DIR, 'script')
# 实验脚本列表（按实验逻辑顺序排列，严格匹配你的文件名）
EXPERIMENT_SCRIPTS = [
    "Free_Vibration.py",          # 1. 自由振动：基础拟合（后续实验的依赖）
    "damped_vibration.py",        # 2. 阻尼振动：阻尼系数计算
    "forced_vibration_table.py"   # 3. 受迫振动：表格生成+幅频相频曲线
]
# 获取当前使用的Python解释器，保证环境一致
PYTHON_EXEC = sys.executable

# ===================== 运行单个实验脚本的函数 =====================
def run_single_script(script_name: str) -> bool:
    """运行单个脚本，返回是否运行成功"""
    script_full_path = os.path.join(SCRIPT_DIR, script_name)
    
    # 检查脚本文件是否存在
    if not os.path.exists(script_full_path):
        print(f"❌ 脚本文件不存在：{script_full_path}")
        return False
    
    # 打印分隔线和提示
    print("\n" + "="*100)
    print(f"▶️  开始运行：{script_name}")
    print("="*100)
    
    # 运行脚本，输出直接打印到控制台
    result = subprocess.run(
        [PYTHON_EXEC, script_full_path],
        cwd=ROOT_DIR,  # 工作目录设为项目根目录，保证脚本内的相对路径正常
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    
    # 判断运行结果
    if result.returncode == 0:
        print(f"\n✅ {script_name} 运行完成！")
        return True
    else:
        print(f"\n❌ {script_name} 运行失败，返回码：{result.returncode}")
        return False

# ===================== 主函数 =====================
def main():
    print("🚀 受迫振动实验全流程自动运行脚本")
    print(f"项目根目录：{ROOT_DIR}")
    print(f"脚本目录：{SCRIPT_DIR}")
    print(f"Python解释器：{PYTHON_EXEC}")

    # 检查script文件夹是否存在
    if not os.path.exists(SCRIPT_DIR):
        print(f"❌ script文件夹不存在：{SCRIPT_DIR}")
        sys.exit(1)

    # 按顺序运行所有实验脚本
    success_count = 0
    total_count = len(EXPERIMENT_SCRIPTS)

    for idx, script_name in enumerate(EXPERIMENT_SCRIPTS):
        is_success = run_single_script(script_name)
        
        if is_success:
            success_count += 1
        else:
            # 第一个脚本（自由振动）是后续所有实验的依赖，失败则直接终止
            if idx == 0:
                print("\n❌ 自由振动脚本运行失败，后续实验依赖其结果，程序终止")
                sys.exit(1)
            # 后续脚本失败，询问是否继续
            user_choice = input("\n⚠️  当前脚本运行失败，是否继续运行下一个？(y/n)：").strip().lower()
            if user_choice not in ['y', 'yes', '是']:
                print("程序终止")
                sys.exit(1)

    # 全部运行完成的总结
    print("\n" + "="*100)
    print(f"📋 全流程运行完成：成功 {success_count}/{total_count} 个脚本")
    if success_count == total_count:
        print("🎉 所有实验脚本全部运行成功！")
    else:
        print("⚠️  部分脚本运行失败，请检查报错信息")
    print("="*100)

if __name__ == '__main__':
    main()