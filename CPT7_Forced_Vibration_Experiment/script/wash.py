import pandas as pd
import numpy as np
import math

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
    DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'origin_data')
    
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