"""
模块四：问答接口
功能：自然语言问题 → 关键词提取 → 意图匹配 → 调用M1/M2/M3函数 → 返回答案
支持 >= 5 种问题类型
"""

import pandas as pd
import numpy as np
import os
import re
import warnings
from openai import OpenAI

warnings.filterwarnings('ignore')


class QAInterface:
    """出行数据问答系统接口"""

    def __init__(self, output_dir='outputs'):
        self.output_dir = output_dir
        self.df = None  # 清洗后的原始数据
        self.demand_df = None  # 聚合后的需求量数据
        self.nn_model = None  # 训练好的神经网络
        self.scaler = None  # 标准化器
        self.feature_cols = None

        # 星期映射：用户口语 → 内部数字 (0=周一, 6=周日)
        self.day_map = {
            '周一': 0, '周二': 1, '周三': 2, '周四': 3,
            '周五': 4, '周六': 5, '周日': 6,
            '星期一': 0, '星期二': 1, '星期三': 2, '星期四': 3,
            '星期五': 4, '星期六': 5, '星期日': 6,
            '周末': 5,
        }

        # 反向映射：内部数字 → 中文显示
        self.day_display = {
            0: '周一', 1: '周二', 2: '周三', 3: '周四',
            4: '周五', 5: '周六', 6: '周日'
        }

        # 时段关键词映射
        self.period_keywords = {
            '深夜': ['深夜', '凌晨', '半夜'],
            '早高峰': ['早高峰', '早上', '早晨', '上午'],
            '白天非高峰': ['白天', '中午', '下午'],
            '晚高峰': ['晚高峰', '傍晚', '晚上'],
            '夜间': ['夜间', '夜里', '今晚']
        }
        # 大模型API（选做）
        self.llm_client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY", "sk-48cd2dc2cec24bff9cc19054046c61d8"),
            base_url="https://api.deepseek.com"
        )

    def load_data(self, df=None, demand_df=None):
        """加载需要的数据"""
        if df is not None:
            self.df = df
        else:
            filepath = os.path.join(self.output_dir, 'yellow_tripdata_2023-01_cleaned.parquet')
            self.df = pd.read_parquet(filepath)

        if demand_df is not None:
            self.demand_df = demand_df
        else:
            # 手动聚合需求量数据（和M3一致）
            self.demand_df = self.df.groupby(
                ['PULocationID', 'pickup_hour', 'pickup_dayofweek']
            ).size().reset_index(name='demand')
            self.demand_df['is_weekend'] = self.demand_df['pickup_dayofweek'].isin([5, 6]).astype(int)

        print(f"问答接口加载完成，数据量: {len(self.df):,} 条")

    # ============================================================
    # 关键词提取
    # ============================================================
    def extract_keywords(self, question):
        """从自然语言问题中提取关键信息"""
        info = {
            'hour': None,
            'day_of_week': None,
            'zone_id': None,
            'is_peak': None,
            'is_weekend': None,
            'period': None
        }

        # 提取小时
        hour_match = re.search(r'(\d{1,2})\s*[点时]', question)
        if hour_match:
            info['hour'] = int(hour_match.group(1))

        # 提取星期（按字符串长度降序匹配，避免"星期四"被"周四"误匹配后剩余"四"）
        sorted_days = sorted(self.day_map.keys(), key=len, reverse=True)
        for day_str in sorted_days:
            if day_str in question:
                info['day_of_week'] = self.day_map[day_str]
                break

        # 提取区域ID
        zone_match = re.search(r'(\d{1,3})\s*号?\s*区域', question)
        if zone_match:
            info['zone_id'] = int(zone_match.group(1))

        # 判断是否高峰
        if '高峰' in question:
            info['is_peak'] = 1
        elif '非高峰' in question:
            info['is_peak'] = 0

        # 判断时段
        for period, keywords in self.period_keywords.items():
            for kw in keywords:
                if kw in question:
                    info['period'] = period
                    break
            if info['period']:
                break

        # 如果没有精确匹配时段但有小时，自动推断
        if not info['period'] and info['hour'] is not None:
            h = info['hour']
            if 0 <= h < 6:
                info['period'] = '深夜'
            elif 6 <= h < 10:
                info['period'] = '早高峰'
            elif 10 <= h < 16:
                info['period'] = '白天非高峰'
            elif 16 <= h < 20:
                info['period'] = '晚高峰'
            else:
                info['period'] = '夜间'

        return info

    # ============================================================
    # 意图识别
    # ============================================================
    def identify_intent(self, question):
        """识别问题的意图类型"""
        if any(w in question for w in ['拥堵', '速度', '堵不堵', '畅通', '堵吗']):
            return 'congestion'
        elif any(w in question for w in ['排名', '最多', '最少', 'TOP', 'top', '热门', '冷门']):
            return 'ranking'
        elif any(w in question for w in ['预测', '预计', '将会', '会多', '会少']):
            return 'prediction'
        elif any(w in question for w in ['费用', '车费', '多少钱', '价格', '贵', '便宜']):
            return 'fare'
        elif any(w in question for w in ['需求', '订单', '多吗', '打车', '出行', '忙吗']):
            return 'demand'
        else:
            return 'general'

    # ============================================================
    # 问题类型1：时段需求查询
    # ============================================================
    def handle_demand(self, question, info):
        """处理时段需求查询"""
        if info['zone_id'] and info['hour'] is not None and info['day_of_week'] is not None:
            zone, hour, day = info['zone_id'], info['hour'], info['day_of_week']
            demand_row = self.demand_df[
                (self.demand_df['PULocationID'] == zone) &
                (self.demand_df['pickup_hour'] == hour) &
                (self.demand_df['pickup_dayofweek'] == day)
                ]
            day_str = self.day_display.get(day, f'周{day}')
            if len(demand_row) > 0:
                val = demand_row.iloc[0]['demand']
                return f"区域 {zone} 在{day_str} {hour}:00 的历史平均需求量为 {val:.0f} 单。"
            else:
                return f"区域 {zone} 在{day_str} {hour}:00 暂无历史数据。"

        # 如果信息不全，给通用统计数据
        if info['hour'] is not None:
            avg_demand = self.demand_df[self.demand_df['pickup_hour'] == info['hour']]['demand'].mean()
            period_str = f"({info['period']})" if info.get('period') else ""
            return f"{info['hour']}:00 {period_str}全区域平均需求量为 {avg_demand:.0f} 单。"

        return "请提供更具体的信息：区域ID、时间和星期。例如：'周五晚上8点132号区域打车多吗？'"

    # ============================================================
    # 问题类型2：区域排名
    # ============================================================
    def handle_ranking(self, question, info):
        """处理区域排名查询"""
        top_n = 5
        n_match = re.search(r'(?:TOP|top|前)\s*(\d+)', question)
        if n_match:
            top_n = int(n_match.group(1))

        # 判断是上车还是下车排名
        is_dropoff = any(w in question for w in ['下车', '到达', '目的地'])

        if is_dropoff:
            ranking = self.df['DOLocationID'].value_counts().head(top_n)
            title = "下车量"
        else:
            ranking = self.df['PULocationID'].value_counts().head(top_n)
            title = "上车量"

        result = f"{title} TOP {top_n} 区域：\n"
        for i, (zone_id, count) in enumerate(ranking.items(), 1):
            result += f"  {i}. 区域 {zone_id}: {count:,} 单\n"

        result += f"\n图表路径: outputs/analysis2_zone_heatmap.png"
        return result

    # ============================================================
    # 问题类型3：拥堵分析
    # ============================================================
    def handle_congestion(self, question, info):
        """处理拥堵分析查询"""
        speed_df = self.df[(self.df['avg_speed_mph'] > 1) & (self.df['avg_speed_mph'] < 50)]

        if info['period']:
            period_speed = speed_df[speed_df['time_period'] == info['period']]['avg_speed_mph'].mean()
            return f"{info['period']} 平均速度为 {period_speed:.1f} mph。"

        if info['hour'] is not None:
            hour_speed = speed_df[speed_df['pickup_hour'] == info['hour']]['avg_speed_mph'].mean()
            period_str = f"({info['period']})" if info.get('period') else ""
            return f"{info['hour']}:00 {period_str}平均速度为 {hour_speed:.1f} mph。"

        # 全局拥堵信息
        period_speed = speed_df.groupby('time_period')['avg_speed_mph'].mean()
        slowest = period_speed.idxmin()
        fastest = period_speed.idxmax()
        slowest_speed = period_speed.min()
        fastest_speed = period_speed.max()

        return (f"最拥堵时段：{slowest}（{slowest_speed:.1f} mph）；"
                f"最畅通时段：{fastest}（{fastest_speed:.1f} mph）。\n"
                f"图表路径: outputs/analysis4_efficiency.png")

    # ============================================================
    # 问题类型4：费用查询
    # ============================================================
    def handle_fare(self, question, info):
        """处理费用查询"""
        if info['period']:
            period_fare = self.df[self.df['time_period'] == info['period']]['fare_amount'].mean()
            return f"{info['period']} 平均车费为 ${period_fare:.2f}。"

        if info['is_peak'] == 1:
            peak_fare = self.df[self.df['is_peak_hour'] == 1]['fare_amount'].mean()
            non_peak_fare = self.df[self.df['is_peak_hour'] == 0]['fare_amount'].mean()
            return (f"高峰时段平均车费 ${peak_fare:.2f}，"
                    f"非高峰时段 ${non_peak_fare:.2f}。")

        if info['hour'] is not None:
            hour_fare = self.df[self.df['pickup_hour'] == info['hour']]['fare_amount'].mean()
            return f"{info['hour']}:00 平均车费为 ${hour_fare:.2f}。"

        # 全局
        avg_fare = self.df['fare_amount'].mean()
        avg_dist = self.df['trip_distance'].mean()
        return (f"整体平均车费 ${avg_fare:.2f}，平均距离 {avg_dist:.1f} 英里。\n"
                f"图表路径: outputs/analysis3_fare_factors.png")

    # ============================================================
    # 问题类型5：需求预测
    # ============================================================
    def handle_prediction(self, question, info):
        """处理需求预测（使用历史均值作为预测值）"""
        if info['zone_id'] and info['hour'] is not None and info['day_of_week'] is not None:
            zone, hour, day = info['zone_id'], info['hour'], info['day_of_week']
            demand_row = self.demand_df[
                (self.demand_df['PULocationID'] == zone) &
                (self.demand_df['pickup_hour'] == hour) &
                (self.demand_df['pickup_dayofweek'] == day)
                ]
            day_str = self.day_display.get(day, f'周{day}')
            if len(demand_row) > 0:
                val = demand_row.iloc[0]['demand']
                # 判断忙不忙
                if val > 200:
                    busy_level = "非常繁忙"
                elif val > 100:
                    busy_level = "比较繁忙"
                elif val > 50:
                    busy_level = "正常"
                else:
                    busy_level = "相对空闲"
                return (f"预测结果：区域 {zone} 在{day_str} {hour}:00 "
                        f"预计需求量为 {val:.0f} 单（{busy_level}）。\n"
                        f"模型对比图表: outputs/model_comparison.png")
            else:
                return f"区域 {zone} 在{day_str} {hour}:00 暂无历史数据，无法预测。"

        return "预测需要区域ID、时间和星期。例如：'预测周五8点132号区域的需求量'"

    # ============================================================
    # 通用兜底
    # ============================================================
    def _call_llm(self, question):
        """调用大模型API兜底回复"""
        try:
            response = self.llm_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是纽约市出租车出行数据助手。"
                            "数据背景：2023年1月，约287万条NYC Yellow Taxi行程。"
                            "回答尽量简短，3句话以内。"
                        )
                    },
                    {"role": "user", "content": question}
                ],
                temperature=0.7,
                max_tokens=200
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"（大模型暂时不可用: {e}）"

    def handle_general(self, question, info):
        """无法匹配规则时，调用大模型"""
        print("（规则未匹配，调用大模型...）")
        return self._call_llm(question)
    def ask(self, question):
        """
        问答主入口：接收自然语言问题，返回答案
        """
        print(f"\n{'=' * 60}")
        print(f"用户: {question}")
        print(f"{'=' * 60}")

        # 提取关键词
        info = self.extract_keywords(question)

        # 识别意图
        intent = self.identify_intent(question)

        # 根据意图分发到对应的处理函数
        handlers = {
            'demand': self.handle_demand,
            'ranking': self.handle_ranking,
            'congestion': self.handle_congestion,
            'fare': self.handle_fare,
            'prediction': self.handle_prediction,
            'general': self.handle_general
        }

        handler = handlers.get(intent, handlers['general'])
        answer = handler(question, info)

        print(f"系统: {answer}")
        print(f"(意图: {intent}, 提取信息: {info})")
        return answer

    def run(self):
        """启动命令行问答循环"""
        print("\n" + "=" * 60)
        print("  NYC 出行数据智能问答系统")
        print("  支持问题类型：时段查询 | 区域排名 | 拥堵分析 | 费用查询 | 需求预测")
        print("  输入 'quit' 或 'exit' 退出")
        print("=" * 60)

        if self.df is None:
            self.load_data()

        print("\n示例问题：")
        print("  - 周五晚上8点132号区域打车多吗？")
        print("  - 上车最多的区域是哪些？")
        print("  - 早高峰堵不堵？")
        print("  - 晚高峰打车贵吗？")
        print("  - 预测周三早上7点100号区域的需求量")
        print()

        while True:
            try:
                question = input("请输入问题: ").strip()
                if question.lower() in ['quit', 'exit', 'q', '退出']:
                    print("感谢使用，再见！")
                    break
                if not question:
                    continue

                self.ask(question)

            except KeyboardInterrupt:
                print("\n感谢使用，再见！")
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")


# ============================================================
# 模块自测入口
# ============================================================
if __name__ == "__main__":
    import os

    os.chdir(os.path.dirname(os.path.dirname(__file__)))

    qa = QAInterface(output_dir='outputs')
    qa.load_data()
    qa.run()