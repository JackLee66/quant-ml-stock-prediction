import akshare as ak
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# 强制禁用代理（解决 VPN 冲突）
os.environ['NO_PROXY'] = 'eastmoney.com,sina.com.cn'
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''

#1.retry获取茅台股票数据
for i in range(5):
    try:
        df = ak.stock_zh_a_hist(
            symbol='600519',
            period='daily',
            start_date='20200101',
            end_date='20230101'
        )

        print("数据获取成功")
        #保存数据
        df.to_csv('maotai.csv', index=False)
        break

    except Exception as e:
        print("获取失败，正在重试...", e)

        time.sleep(3)

#查看基本信息
df = pd.read_csv('maotai.csv')
print(df.head())

#选择需要的列
df = df[['日期','收盘','成交量']]
#对列重命名
df.columns = ['date','close','volume']

#日期格式转换
df['date'] = pd.to_datetime(df['date'])
#按时间排序
df = df.sort_values('date')


#2.特征工程
#收益率
df['return'] = df['close'].pct_change()

#5日、10日平均线
df['MA5'] = df['close'].rolling(5).mean()
df['MA10'] = df['close'].rolling(10).mean()

#波动率
df['volatility'] = df['return'].rolling(5).std()

#动量因子
df['momentum'] = df['close'] - df['close'].shift(5)

#均线差值
df['ma_diff'] = df['MA5'] - df['MA10']

#成交量变化率
df['volume_change'] = df['volume'].pct_change()


#3.构造标签
#明天上涨为1，否则为0
df['label'] = (
    df['close'].shift(-1) > df['close']
).astype(int)

#删除空值
df = df.dropna()

#4.构建数据集
features =  [
    'return',
    'MA5',
    'MA10',
    'volatility',
    'momentum',
    'ma_diff',
    'volume_change'
]

x = df[features]
y = df['label']

#按时间划分
split = int(len(df) * 0.8)

x_train = x[:split]
x_test = x[split:]

y_train = y[:split]
y_test = y[split:]


#5.模型训练
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(x_train, y_train)


#6.模型预测
y_pred = model.predict(x_test)

#模型准确率
acc = accuracy_score(y_test, y_pred)
print("Accuracy:",acc)

#7.策略回测
df_test = df.iloc[split:].copy()
#预测信号
df_test['pred'] = y_pred
#股票真实收益率
df_test['real_return'] = df_test['return']
#策略收益率
df_test['strategy_return'] = (
    df_test['pred'] * df_test['real_return']
)
#市场累计收益
df_test['cum_market'] = (
    1 + df_test['real_return']
).cumprod()
#策略累计收益
df_test['cum_strategy'] = (
    1 + df_test['strategy_return']
).cumprod()


#8.可视化
plt.figure(figsize=(12, 6))

plt.plot(
    df_test['date'],
    df_test['cum_market'],
    label='Market Return'
)

plt.plot(
    df_test['date'],
    df_test['cum_strategy'],
    label='Strategy Return'
)

plt.xlabel('Date')
plt.ylabel('Cumulative Return')

plt.title('Quantitative Strategy Backtest')

plt.legend()

plt.grid(True)

plt.show()
