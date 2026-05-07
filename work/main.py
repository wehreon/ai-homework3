"""
main.py - NYC 出行数据智能问答系统
一键运行入口：数据处理 → 分析可视化 → 预测模型 → 问答接口
"""

import os
import sys

# 确保工作目录在项目根目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 添加 src 目录到系统路径
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

from M1 import DataProcessor
from M2 import Analyzer
from M3 import Predictor
from M4 import QAInterface

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  NYC Yellow Taxi 出行数据智能问答系统")
    print("  一键启动：数据处理 → 分析 → 预测 → 问答")
    print("=" * 60)

    # ================================================================
    # 模块一：数据处理与特征工程
    # ================================================================
    print("\n" + "▶" * 30)
    print("▶ 阶段 1/4：数据处理与特征工程")
    print("▶" * 30)

    processor = DataProcessor(data_dir='data', output_dir='outputs')
    df_clean = processor.run()

    # ================================================================
    # 模块二：数据分析与可视化
    # ================================================================
    print("\n" + "▶" * 30)
    print("▶ 阶段 2/4：数据分析与可视化")
    print("▶" * 30)

    analyzer = Analyzer(output_dir='outputs')
    analyzer.load_data(df=df_clean)
    analyzer.run()

    # ================================================================
    # 模块三：预测模型
    # ================================================================
    print("\n" + "▶" * 30)
    print("▶ 阶段 3/4：出行需求预测模型")
    print("▶" * 30)

    predictor = Predictor(output_dir='outputs')
    predictor.load_data(df=df_clean)
    predictor.run()

    # ================================================================
    # 模块四：问答接口
    # ================================================================
    print("\n" + "▶" * 30)
    print("▶ 阶段 4/4：智能问答接口")
    print("▶" * 30)

    qa = QAInterface(output_dir='outputs')
    qa.load_data(df=df_clean, demand_df=predictor.demand_df)
    qa.run()

    print("\n" + "=" * 60)
    print("  系统运行完毕，感谢使用！")
    print("=" * 60)