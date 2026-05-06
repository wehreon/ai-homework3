"""
模块二：数据分析与可视化
功能：出行需求分析、区域热度分析、车费影响因素分析、车速与效率分析
所有图表自动保存至 outputs/ 目录
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

# 设置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class Analyzer:
    """数据分析与可视化类"""

    def __init__(self, output_dir='outputs'):
        """
        初始化分析器

        参数:
            output_dir: 图表输出目录（相对路径）
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.df = None

    def load_data(self, filepath=None, df=None):
        """
        加载数据

        参数:
            filepath: 数据文件路径
            df: 直接传入DataFrame
        """
        if df is not None:
            self.df = df
        elif filepath is not None:
            self.df = pd.read_parquet(filepath)
        else:
            # 默认读取清洗后的数据
            self.df = pd.read_parquet(os.path.join(self.output_dir, 'yellow_tripdata_2023-01_cleaned.parquet'))

        print(f"分析模块加载数据: {len(self.df):,} 条")
        return self.df

    # ================================================================
    # 分析一：出行需求时间规律
    # ================================================================
    def analysis1_demand_pattern(self):
        """
        分析一：出行需求时间规律
        图表1：分小时平均订单量折线图（工作日 vs 周末对比）
        """
        print("\n" + "=" * 60)
        print("分析一：出行需求时间规律")
        print("=" * 60)

        df = self.df

        # 按小时和是否周末聚合订单量
        hourly_demand = df.groupby(['pickup_hour', 'is_weekend']).size().unstack(fill_value=0)
        hourly_demand.columns = ['工作日', '周末']

        # 打印关键数据
        peak_weekday = hourly_demand['工作日'].idxmax()
        peak_weekend = hourly_demand['周末'].idxmax()
        print(f"  工作日高峰小时: {peak_weekday}点 ({hourly_demand.loc[peak_weekday, '工作日']:,.0f} 单)")
        print(f"  周末高峰小时: {peak_weekend}点 ({hourly_demand.loc[peak_weekend, '周末']:,.0f} 单)")

        # 绘制图表
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 子图1：分小时订单量（工作日 vs 周末）
        ax1 = axes[0]
        ax1.plot(hourly_demand.index, hourly_demand['工作日'],
                 marker='o', linewidth=2, color='steelblue', label='工作日', markersize=4)
        ax1.plot(hourly_demand.index, hourly_demand['周末'],
                 marker='s', linewidth=2, color='coral', label='周末', markersize=4)
        ax1.set_xlabel('小时')
        ax1.set_ylabel('订单量（条）')
        ax1.set_title('分小时订单量：工作日 vs 周末')
        ax1.set_xticks(range(0, 24, 2))
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 子图2：分时段的订单占比
        ax2 = axes[1]
        period_order = ['深夜', '早高峰', '白天非高峰', '晚高峰', '夜间']
        period_demand = df.groupby(['time_period', 'is_weekend']).size().unstack(fill_value=0)
        period_demand.columns = ['工作日', '周末']
        period_demand = period_demand.reindex(period_order)

        x = np.arange(len(period_demand))
        width = 0.35
        ax2.bar(x - width / 2, period_demand['工作日'], width, label='工作日', color='steelblue')
        ax2.bar(x + width / 2, period_demand['周末'], width, label='周末', color='coral')
        ax2.set_xlabel('时段')
        ax2.set_ylabel('订单量（条）')
        ax2.set_title('分时段订单量对比')
        ax2.set_xticks(x)
        ax2.set_xticklabels(period_order, rotation=30)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'analysis1_demand_pattern.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 图表已保存: {save_path}")

        return hourly_demand

    # ================================================================
    # 分析二：区域热度分析
    # ================================================================
    def analysis2_zone_heatmap(self):
        """
        分析二：区域热度分析
        图表2：上下客量最高的 TOP 10 区域柱状图
        图表3：高峰时段 TOP 5 区域订单量
        """
        print("\n" + "=" * 60)
        print("分析二：区域热度分析")
        print("=" * 60)

        df = self.df

        # TOP 10 上车区域
        pu_top10 = df['PULocationID'].value_counts().head(10)
        # TOP 10 下车区域
        do_top10 = df['DOLocationID'].value_counts().head(10)

        print(f"  上车量最高区域: ID {pu_top10.index[0]} ({pu_top10.values[0]:,} 单)")
        print(f"  下车量最高区域: ID {do_top10.index[0]} ({do_top10.values[0]:,} 单)")

        # 绘制图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 子图1：上车 TOP 10
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(10), pu_top10.values, color='steelblue')
        ax1.set_xticks(range(10))
        ax1.set_xticklabels(pu_top10.index, rotation=45)
        ax1.set_xlabel('区域ID')
        ax1.set_ylabel('订单量（条）')
        ax1.set_title('上车量 TOP 10 区域')
        # 在柱子上标注数值
        for bar, val in zip(bars1, pu_top10.values):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                     f'{val:,}', ha='center', va='bottom', fontsize=8)

        # 子图2：下车 TOP 10
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(10), do_top10.values, color='coral')
        ax2.set_xticks(range(10))
        ax2.set_xticklabels(do_top10.index, rotation=45)
        ax2.set_xlabel('区域ID')
        ax2.set_ylabel('订单量（条）')
        ax2.set_title('下车量 TOP 10 区域')
        for bar, val in zip(bars2, do_top10.values):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                     f'{val:,}', ha='center', va='bottom', fontsize=8)

        # 子图3：高峰时段上车 TOP 5 区域
        ax3 = axes[1, 0]
        peak_df = df[df['is_peak_hour'] == 1]
        peak_pu = peak_df['PULocationID'].value_counts().head(5)
        colors_peak = ['#FF6B6B', '#FF8E53', '#FFA94D', '#FFC078', '#FFD8A8']
        ax3.pie(peak_pu.values, labels=peak_pu.index, autopct='%1.1f%%',
                colors=colors_peak, startangle=90)
        ax3.set_title('高峰时段上车 TOP 5 区域占比')

        # 子图4：非高峰 vs 高峰 TOP 5 区域对比
        ax4 = axes[1, 1]
        non_peak_df = df[df['is_peak_hour'] == 0]
        top5_zones = pu_top10.head(5).index

        peak_counts = [len(peak_df[peak_df['PULocationID'] == z]) for z in top5_zones]
        non_peak_counts = [len(non_peak_df[non_peak_df['PULocationID'] == z]) for z in top5_zones]

        x = np.arange(len(top5_zones))
        width = 0.35
        ax4.bar(x - width / 2, non_peak_counts, width, label='非高峰', color='steelblue')
        ax4.bar(x + width / 2, peak_counts, width, label='高峰', color='#FF6B6B')
        ax4.set_xticks(x)
        ax4.set_xticklabels(top5_zones)
        ax4.set_xlabel('区域ID')
        ax4.set_ylabel('订单量（条）')
        ax4.set_title('TOP 5 区域高峰/非高峰订单对比')
        ax4.legend()

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'analysis2_zone_heatmap.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 图表已保存: {save_path}")

        return pu_top10, do_top10

    # ================================================================
    # 分析三：车费影响因素分析
    # ================================================================
    def analysis3_fare_factors(self):
        """
        分析三：车费影响因素分析
        图表4：行程距离-车费散点图（按时段着色）
        图表5：不同乘客数对应的平均车费
        """
        print("\n" + "=" * 60)
        print("分析三：车费影响因素分析")
        print("=" * 60)

        df = self.df

        # 采样以提高绘图效率（数据量大时散点图会卡）
        sample_size = min(50000, len(df))
        df_sample = df.sample(n=sample_size, random_state=42)

        # 打印关键统计
        correlation = df['trip_distance'].corr(df['fare_amount'])
        print(f"  距离与车费的相关系数: {correlation:.3f}")
        print(f"  平均车费: ${df['fare_amount'].mean():.2f}")
        print(f"  高峰时段平均车费: ${df[df['is_peak_hour'] == 1]['fare_amount'].mean():.2f}")
        print(f"  非高峰时段平均车费: ${df[df['is_peak_hour'] == 0]['fare_amount'].mean():.2f}")

        # 绘制图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 子图1：距离-车费散点图（按时段着色）
        ax1 = axes[0, 0]
        # 过滤极端值使图更清晰
        plot_df = df_sample[(df_sample['trip_distance'] < 20) & (df_sample['fare_amount'] < 80)]

        # 按时段着色
        period_colors = {'深夜': '#6C5B7B', '早高峰': '#C06C84',
                         '白天非高峰': '#355C7D', '晚高峰': '#F67280', '夜间': '#8B9DC3'}
        for period in period_colors:
            subset = plot_df[plot_df['time_period'] == period]
            ax1.scatter(subset['trip_distance'], subset['fare_amount'],
                        c=period_colors[period], alpha=0.4, s=3, label=period)
        ax1.set_xlabel('行程距离（英里）')
        ax1.set_ylabel('车费（美元）')
        ax1.set_title(f'行程距离 vs 车费 (相关系数={correlation:.3f})')
        ax1.legend(markerscale=3, fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 子图2：不同乘客数的平均车费
        ax2 = axes[0, 1]
        passenger_fare = df.groupby('passenger_count').agg(
            avg_fare=('fare_amount', 'mean'),
            trip_count=('fare_amount', 'count')
        ).reset_index()
        passenger_fare = passenger_fare[passenger_fare['passenger_count'].between(1, 6)]

        bars2 = ax2.bar(passenger_fare['passenger_count'], passenger_fare['avg_fare'],
                        color='steelblue')
        ax2.set_xlabel('乘客数（人）')
        ax2.set_ylabel('平均车费（美元）')
        ax2.set_title('不同乘客数的平均车费')
        ax2.set_xticks(passenger_fare['passenger_count'])
        for bar, val in zip(bars2, passenger_fare['avg_fare']):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f'${val:.1f}', ha='center', va='bottom', fontsize=9)

        # 子图3：不同时段的平均车费对比
        ax3 = axes[1, 0]
        period_order = ['深夜', '早高峰', '白天非高峰', '晚高峰', '夜间']
        period_fare = df.groupby('time_period')['fare_amount'].mean().reindex(period_order)
        bars3 = ax3.bar(range(len(period_fare)), period_fare.values,
                        color=['#6C5B7B', '#C06C84', '#355C7D', '#F67280', '#8B9DC3'])
        ax3.set_xticks(range(len(period_fare)))
        ax3.set_xticklabels(period_order, rotation=30)
        ax3.set_xlabel('时段')
        ax3.set_ylabel('平均车费（美元）')
        ax3.set_title('不同时段的平均车费')
        for bar, val in zip(bars3, period_fare.values):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                     f'${val:.2f}', ha='center', va='bottom', fontsize=9)

        # 子图4：距离-速度-车费气泡图
        ax4 = axes[1, 1]
        bubble_df = df_sample[(df_sample['trip_distance'] < 15) &
                              (df_sample['avg_speed_mph'] < 50) &
                              (df_sample['fare_amount'] < 60)]
        scatter = ax4.scatter(bubble_df['trip_distance'], bubble_df['avg_speed_mph'],
                              c=bubble_df['fare_amount'], cmap='YlOrRd',
                              alpha=0.5, s=10)
        ax4.set_xlabel('行程距离（英里）')
        ax4.set_ylabel('平均速度（mph）')
        ax4.set_title('距离-速度-车费关系（颜色=车费）')
        plt.colorbar(scatter, ax=ax4, label='车费($)')

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'analysis3_fare_factors.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 图表已保存: {save_path}")

        return correlation

    # ================================================================
    # 分析四（自选）：出行效率分析——速度与拥堵模式
    # ================================================================
    def analysis4_efficiency(self):
        """
        分析四（自选）：出行效率分析
        洞察价值：速度是衡量交通拥堵的最直接指标，分析速度的时间分布
        可以发现拥堵规律，为出行时间预测和路径规划提供依据。

        图表5：各时段平均速度箱线图
        图表6：工作日/周末分小时平均速度变化
        """
        print("\n" + "=" * 60)
        print("分析四（自选）：出行效率——速度与拥堵模式")
        print("=" * 60)

        df = self.df
        # 过滤掉速度异常值，使用正常范围内的数据
        speed_df = df[(df['avg_speed_mph'] > 1) & (df['avg_speed_mph'] < 50)]

        # 打印关键发现
        period_speed = speed_df.groupby('time_period')['avg_speed_mph'].mean()
        slowest_period = period_speed.idxmin()
        fastest_period = period_speed.idxmax()
        print(f"  最拥堵时段: {slowest_period} (平均速度 {period_speed[slowest_period]:.1f} mph)")
        print(f"  最畅通时段: {fastest_period} (平均速度 {period_speed[fastest_period]:.1f} mph)")
        print(f"  工作日平均速度: {speed_df[speed_df['is_weekend'] == 0]['avg_speed_mph'].mean():.1f} mph")
        print(f"  周末平均速度: {speed_df[speed_df['is_weekend'] == 1]['avg_speed_mph'].mean():.1f} mph")

        # 绘制图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 子图1：各时段速度箱线图
        ax1 = axes[0, 0]
        period_order = ['深夜', '早高峰', '白天非高峰', '晚高峰', '夜间']
        period_data = [speed_df[speed_df['time_period'] == p]['avg_speed_mph'].values
                       for p in period_order]
        bp = ax1.boxplot(period_data, labels=period_order, patch_artist=True,
                         showfliers=False)
        colors = ['#6C5B7B', '#C06C84', '#355C7D', '#F67280', '#8B9DC3']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax1.set_xlabel('时段')
        ax1.set_ylabel('平均速度（mph）')
        ax1.set_title('各时段行程速度分布')
        ax1.grid(True, alpha=0.3, axis='y')

        # 子图2：工作日/周末分小时平均速度
        ax2 = axes[0, 1]
        hourly_speed = speed_df.groupby(['pickup_hour', 'is_weekend'])['avg_speed_mph'].mean().unstack()
        hourly_speed.columns = ['工作日', '周末']

        ax2.plot(hourly_speed.index, hourly_speed['工作日'],
                 marker='o', linewidth=2, color='steelblue', label='工作日', markersize=4)
        ax2.plot(hourly_speed.index, hourly_speed['周末'],
                 marker='s', linewidth=2, color='coral', label='周末', markersize=4)
        ax2.set_xlabel('小时')
        ax2.set_ylabel('平均速度（mph）')
        ax2.set_title('分小时平均速度：工作日 vs 周末')
        ax2.set_xticks(range(0, 24, 2))
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 子图3：速度分类占比
        ax3 = axes[1, 0]
        speed_cat = df['speed_category'].value_counts()
        explode = [0.05 if i == speed_cat.index.tolist().index('拥堵') else 0
                   for i in range(len(speed_cat))]
        cat_colors = {'极度拥堵': '#D32F2F', '拥堵': '#F44336', '正常': '#4CAF50',
                      '畅通': '#2196F3', '疑似异常': '#FF9800'}
        pie_colors = [cat_colors.get(cat, '#9E9E9E') for cat in speed_cat.index]
        ax3.pie(speed_cat.values, labels=speed_cat.index, autopct='%1.1f%%',
                colors=pie_colors, explode=explode, startangle=90)
        ax3.set_title('速度分类占比')

        # 子图4：人均距离与速度的关系
        ax4 = axes[1, 1]
        eff_df = df[(df['avg_speed_mph'] > 1) & (df['avg_speed_mph'] < 50) &
                    (df['avg_distance_per_passenger'] < 20)]

        # 按乘客数分组计算均值
        passenger_eff = eff_df.groupby('passenger_count').agg(
            avg_speed=('avg_speed_mph', 'mean'),
            avg_dist_per_person=('avg_distance_per_passenger', 'mean')
        ).reset_index()
        passenger_eff = passenger_eff[passenger_eff['passenger_count'].between(1, 5)]

        ax4_twin = ax4.twinx()
        bar_width = 0.25
        x = passenger_eff['passenger_count']
        bars_speed = ax4.bar(x - bar_width / 2, passenger_eff['avg_speed'],
                             bar_width, color='steelblue', label='平均速度(mph)')
        bars_dist = ax4_twin.bar(x + bar_width / 2, passenger_eff['avg_dist_per_person'],
                                 bar_width, color='coral', label='人均距离(英里)')
        ax4.set_xlabel('乘客数')
        ax4.set_ylabel('平均速度（mph）', color='steelblue')
        ax4_twin.set_ylabel('人均距离（英里）', color='coral')
        ax4.set_title('不同乘客数的出行效率对比')
        ax4.set_xticks(x)

        # 合并图例
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=8)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'analysis4_efficiency.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ 图表已保存: {save_path}")

        return period_speed

    # ================================================================
    # 运行全部分析
    # ================================================================
    def run(self):
        """运行全部四项分析"""
        print("\n" + "=" * 60)
        print("模块二：数据分析与可视化")
        print("=" * 60)

        if self.df is None:
            self.load_data()

        self.analysis1_demand_pattern()
        self.analysis2_zone_heatmap()
        self.analysis3_fare_factors()
        self.analysis4_efficiency()

        print("\n" + "=" * 60)
        print("模块二完成！所有图表已保存至 outputs/ 目录")
        print("=" * 60)


# ================================================================
# 模块自测入口
# ================================================================
if __name__ == "__main__":
    import os

    # 切换到项目根目录
    os.chdir(os.path.dirname(os.path.dirname(__file__)))

    analyzer = Analyzer(output_dir='outputs')
    analyzer.load_data()
    analyzer.run()