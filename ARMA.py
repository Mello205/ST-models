import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# 生成示例数据：股票价格的时间序列
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=100)
data = np.cumsum(np.random.randn(100)) + 100  # 随机漫步序列

# 创建DataFrame
df = pd.DataFrame(data, index=dates, columns=['Stock Price'])

# 拟合ARMA模型 (p=2, q=2)
model = ARIMA(df['Stock Price'], order=(2, 0, 2))
arma_result = model.fit()

# 预测未来20个时间点
forecast = arma_result.get_forecast(steps=20)
forecast_index = pd.date_range(df.index[-1], periods=21, freq='D')[1:]
forecast_values = forecast.predicted_mean

# 可视化
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Stock Price'], label='Observed', color='blue')
plt.plot(forecast_index, forecast_values, label='Forecast', color='red', linestyle='--')
plt.fill_between(forecast_index, 
                 forecast.conf_int().iloc[:, 0], 
                 forecast.conf_int().iloc[:, 1], 
                 color='pink', alpha=0.3)
plt.title('ARMA Model Forecast of Stock Price')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()