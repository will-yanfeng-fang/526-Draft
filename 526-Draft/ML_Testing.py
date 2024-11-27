import tensorflow as tf
import pandas as pd
import numpy as py
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df4 = pd.read_excel('Dataset_labeled.xlsx')
data = pd.DataFrame(df4)
data_filtered = data[data['haveNearCity']==1]
data_filtered['dis_to_bor'] = data_filtered['dis_to_bor'] * 1000

# 选择自变量和因变量
features = ['dis_to_bor', 'OCP_pref', 'mrgecpl_rstor', 'hs_price', 'incm_tnpc'] # 这里选择还需要调整
target = 'avg_totpop'

# 分离特征和目标变量
X = data_filtered[features]
y = data_filtered[target]

print(X)

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) # 参数微调，初始值 0.2，42

# 定义模型结构
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)  # 输出层，用于回归
])

# 编译模型
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 训练模型
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# 可视化训练过程
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# 评估模型性能
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test MAE: {mae}")

# 预测
y_pred = model.predict(X_test)

# 可视化预测结果
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Birth Rate')
plt.show()

"""
模型能正常运行，但机器学习学不不到东西（？）
loss & mae 都没有输出值
"""