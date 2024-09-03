import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# 生成示例数据：时间序列
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=100)
data = np.sin(np.linspace(0, 10, 100)) + np.random.randn(100) * 0.1

# 创建DataFrame
df = pd.DataFrame({'Date': dates, 'Value': data})

# 预处理数据
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['Value']])
X, y = [], []
for i in range(len(scaled_data) - 10):
    X.append(scaled_data[i:i+10])
    y.append(scaled_data[i+10])
X, y = np.array(X), np.array(y)

# 构建LSTM模型
model = Sequential()
model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(X, y, epochs=20, verbose=1)

# 预测
predicted = model.predict(X)
predicted = scaler.inverse_transform(predicted)
actual = scaler.inverse_transform(y.reshape(-1, 1))

# 可视化
plt.figure(figsize=(12, 6))
plt.plot(df['Date'][10:], actual, label='Actual', color='blue')
plt.plot(df['Date'][10:], predicted, label='Predicted', color='red', linestyle='--')
plt.title('LSTM Model Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()