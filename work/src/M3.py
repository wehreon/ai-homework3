"""
模块三：预测模型
任务：预测某区域某时段的出行需求量
方法：PyTorch 神经网络 vs 随机森林对比
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

# 深度学习
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 机器学习
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 神经网络定义
# ============================================================
class DemandPredictNN(nn.Module):
    """用于出行需求预测的简单全连接神经网络"""

    def __init__(self, input_dim):
        super(DemandPredictNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


# ============================================================
# 预测器主类
# ============================================================
class Predictor:
    """出行需求预测模型类"""

    def __init__(self, output_dir='outputs'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.df = None
        self.demand_df = None

        # 数据相关
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.feature_cols = None

        # 结果存储
        self.nn_train_losses = []
        self.results = {}

    def load_data(self, df=None):
        """加载清洗后的数据"""
        if df is not None:
            self.df = df
        else:
            filepath = os.path.join(self.output_dir, 'yellow_tripdata_2023-01_cleaned.parquet')
            self.df = pd.read_parquet(filepath)
        print(f"预测模块加载数据: {len(self.df):,} 条")
        return self.df

    def build_demand_dataset(self):
        """
        第一步：聚合数据，构建需求量数据集
        """
        print("\n" + "=" * 60)
        print("第一步：构建需求量数据集")
        print("=" * 60)

        df = self.df

        # 聚合：区域 + 小时 + 星期 → 订单量
        demand_df = df.groupby(
            ['PULocationID', 'pickup_hour', 'pickup_dayofweek']
        ).size().reset_index(name='demand')

        # 添加特征：是否周末
        demand_df['is_weekend'] = demand_df['pickup_dayofweek'].isin([5, 6]).astype(int)

        print(f"聚合后数据量: {len(demand_df):,} 条")
        print(f"区域数: {demand_df['PULocationID'].nunique()}")
        print(f"需求量范围: {demand_df['demand'].min()} ~ {demand_df['demand'].max()}")
        print(f"平均需求量: {demand_df['demand'].mean():.1f}")

        self.demand_df = demand_df
        return demand_df

    def prepare_data(self):
        """
        第二步：划分训练/测试集 (8:2)，对区域ID做 One-Hot 编码，并标准化
        """
        print("\n" + "=" * 60)
        print("第二步：数据准备（含 One-Hot 编码 + 标准化）")
        print("=" * 60)

        if self.demand_df is None:
            self.build_demand_dataset()

        df = self.demand_df

        # 对区域ID做 One-Hot 编码
        location_dummies = pd.get_dummies(df['PULocationID'], prefix='loc')

        # 数值特征
        numeric_features = df[['pickup_hour', 'pickup_dayofweek', 'is_weekend']]

        # 合并
        X = pd.concat([location_dummies, numeric_features], axis=1)
        self.feature_cols = list(X.columns)
        y = df['demand'].values.reshape(-1, 1)

        # 划分训练/测试集 (8:2)
        X_train_raw, X_test_raw, self.y_train, self.y_test = train_test_split(
            X.values, y, test_size=0.2, random_state=42
        )

        # 标准化
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_raw)
        X_test_scaled = self.scaler.transform(X_test_raw)

        # 转换为浮点型张量
        self.X_train = torch.FloatTensor(X_train_scaled)
        self.X_test = torch.FloatTensor(X_test_scaled)
        self.y_train = torch.FloatTensor(self.y_train)
        self.y_test = torch.FloatTensor(self.y_test)

        print(f"One-Hot 编码后特征维度: {self.X_train.shape[1]}")
        print(f"训练集大小: {len(self.X_train):,} 条")
        print(f"测试集大小: {len(self.X_test):,} 条")

    def train_neural_network(self, epochs=300, batch_size=512, lr=0.0005):
        """
        第三步：训练优化后的神经网络并绘制 Loss 曲线
        - 更多 epoch (300)
        - 更低学习率 (0.0005)
        - 更大的 batch size (512)
        """
        print("\n" + "=" * 60)
        print("第三步：训练神经网络 (PyTorch, 优化版)")
        print("=" * 60)

        input_dim = self.X_train.shape[1]
        model = DemandPredictNN(input_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 使用学习率调度器，在平台期降低学习率
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)

        train_dataset = TensorDataset(self.X_train, self.y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        train_losses = []
        print("开始训练（300 epochs，预计约1-2分钟）...")
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            scheduler.step(avg_loss)

            if (epoch + 1) % 30 == 0:
                print(f"  Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

        # 评估
        model.eval()
        with torch.no_grad():
            y_pred_nn = model(self.X_test).numpy().flatten()

        mae_nn = mean_absolute_error(self.y_test.numpy(), y_pred_nn)
        rmse_nn = np.sqrt(mean_squared_error(self.y_test.numpy(), y_pred_nn))

        print(f"\n神经网络测试结果（优化后）:")
        print(f"  MAE:  {mae_nn:.2f}")
        print(f"  RMSE: {rmse_nn:.2f}")

        # 绘制 Loss 曲线
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, epochs + 1), train_losses, color='steelblue', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('神经网络训练 Loss 曲线（优化后）')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        loss_path = os.path.join(self.output_dir, 'nn_loss_curve.png')
        plt.savefig(loss_path, dpi=150)
        plt.close()
        print(f"  ✓ Loss 曲线已保存: {loss_path}")

        self.nn_train_losses = train_losses
        self.results['Neural Network'] = {'MAE': mae_nn, 'RMSE': rmse_nn, 'Predictions': y_pred_nn}
        self.nn_model = model

    def train_random_forest(self):
        """
        第四步：训练随机森林
        """
        print("\n" + "=" * 60)
        print("第四步：训练随机森林 (Scikit-learn)")
        print("=" * 60)

        # 注意：随机森林接受 NumPy 数组
        X_train_np = self.X_train.numpy()
        y_train_np = self.y_train.numpy().ravel()
        X_test_np = self.X_test.numpy()

        print("开始训练随机森林...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_np, y_train_np)

        y_pred_rf = rf_model.predict(X_test_np)

        mae_rf = mean_absolute_error(self.y_test.numpy(), y_pred_rf)
        rmse_rf = np.sqrt(mean_squared_error(self.y_test.numpy(), y_pred_rf))

        print(f"\n随机森林测试结果:")
        print(f"  MAE:  {mae_rf:.2f}")
        print(f"  RMSE: {rmse_rf:.2f}")

        self.results['Random Forest'] = {'MAE': mae_rf, 'RMSE': rmse_rf, 'Predictions': y_pred_rf}
        self.rf_model = rf_model

    def compare_models(self):
        """
        第五步：模型对比
        """
        print("\n" + "=" * 60)
        print("第五步：模型对比分析")
        print("=" * 60)

        print(f"\n{'模型':<20} {'MAE':<10} {'RMSE':<10}")
        print("-" * 40)
        for model_name, metrics in self.results.items():
            print(f"{model_name:<20} {metrics['MAE']:<10.2f} {metrics['RMSE']:<10.2f}")

        # 绘制对比柱状图
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        model_names = list(self.results.keys())
        mae_values = [self.results[m]['MAE'] for m in model_names]
        rmse_values = [self.results[m]['RMSE'] for m in model_names]

        colors = ['steelblue', 'coral']

        # MAE 对比
        axes[0].bar(model_names, mae_values, color=colors)
        axes[0].set_ylabel('MAE')
        axes[0].set_title('MAE 对比')
        for bar, val in zip(axes[0].patches, mae_values):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f'{val:.2f}', ha='center', fontsize=10)

        # RMSE 对比
        axes[1].bar(model_names, rmse_values, color=colors)
        axes[1].set_ylabel('RMSE')
        axes[1].set_title('RMSE 对比')
        for bar, val in zip(axes[1].patches, rmse_values):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f'{val:.2f}', ha='center', fontsize=10)

        plt.suptitle('神经网络 vs 随机森林：预测误差对比', fontsize=14)
        plt.tight_layout()
        compare_path = os.path.join(self.output_dir, 'model_comparison.png')
        plt.savefig(compare_path, dpi=150)
        plt.close()
        print(f"  ✓ 对比图表已保存: {compare_path}")

    def run(self):
        """运行完整的预测模型流程"""
        print("\n" + "=" * 60)
        print("模块三：预测模型")
        print("=" * 60)

        if self.df is None:
            self.load_data()

        self.build_demand_dataset()
        self.prepare_data()
        self.train_neural_network()
        self.train_random_forest()
        self.compare_models()

        print("\n" + "=" * 60)
        print("模块三完成！")
        print("=" * 60)
        return self.results


# ============================================================
# 模块自测入口
# ============================================================
if __name__ == "__main__":
    import os

    os.chdir(os.path.dirname(os.path.dirname(__file__)))

    predictor = Predictor(output_dir='outputs')
    predictor.load_data()
    predictor.run()