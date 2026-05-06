import akshare as ak
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. 获取股票数据（以贵州茅台为例）
df = ak.stock_zh_a_hist(symbol="600519", period="daily", start_date="20200101", end_date="20231231")

# 2. 数据预处理
df = df[['日期', '收盘']]
df.columns = ['date', 'close']
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# 3. 构造特征
df['return'] = df['close'].pct_change()
df['MA5'] = df['close'].rolling(5).mean()
df['MA10'] = df['close'].rolling(10).mean()

# 标签：明天涨=1，跌=0
df['label'] = (df['close'].shift(-1) > df['close']).astype(int)

df = df.dropna()

# 4. 构建数据集
features = ['return', 'MA5', 'MA10']
X = df[features]
y = df['label']

# 按时间划分（非常重要）
split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 5. 模型训练
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. 预测
y_pred = model.predict(X_test)

# 7. 评估
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# 8. 可视化
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="True")
plt.plot(y_pred, label="Pred")
plt.legend()
plt.title("Prediction vs True")
plt.show()