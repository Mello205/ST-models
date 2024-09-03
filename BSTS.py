import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpyro
from bsts import BSTS
import jax
import jax.numpy as jnp

# 确认可用设备数量
print(f"Number of available devices: {jax.local_device_count()}")

# 设置主机设备数量（根据实际情况调整）
numpyro.set_host_device_count(1)  # 设置为实际可用的设备数量

# 生成示例数据
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=365)
data = np.linspace(10, 50, 365) + 10 * np.sin(np.linspace(0, 2 * np.pi, 365)) + np.random.randn(365) * 5
df = pd.DataFrame({'Date': dates, 'Value': data})

# 确保数据格式正确
values = np.asarray(df['Value'], dtype=np.float32)

# 初始化 BSTS 模型
model = BSTS(values)

# 拟合模型
model.fit(values)

# 预测未来30天
forecast = model.predict(steps=30)

# 生成未来日期
forecast_index = pd.date_range(dates[-1] + pd.DateOffset(days=1), periods=30)
forecast_values = forecast['mean']  # 根据实际返回值的结构调整

# 可视化
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], df['Value'], label='Observed', color='blue')
plt.plot(forecast_index, forecast_values, label='Forecast', color='red', linestyle='--')
plt.title('BSTS Model Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()