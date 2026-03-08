import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

print("🧠 正在加载 T-SFE 深度卷积神经网络 (1D-CNN)...")

# ==========================================
# 1. 加载与预处理数据 (专为 CNN 打造的三维张量)
# ==========================================
df = pd.read_csv("tsfe_training_data_5000.csv")

def sequence_to_cnn_matrix(seq):
    # CNN 需要的形状是 (通道数, 序列长度)，即 4行(A,C,G,U) x 43列
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'U': [0,0,0,1]}
    mat = np.array([mapping.get(base, [0,0,0,0]) for base in seq])
    return mat.T # 转置为 (4, 43)

print("[1/4] 正在将 RNA 翻译成卷积特征图谱...")
X = np.stack(df['Sequence'].apply(sequence_to_cnn_matrix).values).astype(np.float32)
y = df['Fidelity_Score'].values.astype(np.float32).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为 PyTorch 的张量 (Tensor)
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# ==========================================
# 2. 定义神经网络架构 (Architecture)
# ==========================================
class RNACNN(nn.Module):
    def __init__(self):
        super(RNACNN, self).__init__()
        # 第一层卷积：用 16 个大小为 3 的滑动窗口，捕捉局部三核苷酸特征
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # 第二层卷积：用 32 个窗口，组合底层特征，捕捉更长的茎环结构
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # 全连接层：把提取到的拓扑特征压缩，最终输出 1 个保真度打分
        self.fc1 = nn.Linear(32 * 43, 64) 
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # 展平操作 (Flatten)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = RNACNN()
print("\n[2/4] 网络搭建完毕！架构如下：")
print("   输入层 (4x43) -> Conv1d(16核) -> Conv1d(32核) -> Dense(64) -> 输出打分")

# ==========================================
# 3. 配置引擎：优化器与损失函数
# ==========================================
criterion = nn.MSELoss() # 均方误差
optimizer = optim.Adam(model.parameters(), lr=0.001) # 经典的 Adam 优化器

# ==========================================
# 4. 训练循环 (Epochs)
# ==========================================
epochs = 80 # 让 AI 反复阅读这批数据 80 遍
print(f"\n[3/4] 轰鸣吧！开始 {epochs} 轮的深度学习训练 (Forward & Backward)...")

start_time = time.time()
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()    # 清空梯度
        outputs = model(inputs)  # 正向传播 (猜分数)
        loss = criterion(outputs, labels) # 计算误差
        loss.backward()          # 反向传播 (找原因)
        optimizer.step()         # 更新权重 (自我进化)
        running_loss += loss.item()
        
    # 每 20 轮汇报一次进度
    if (epoch + 1) % 20 == 0:
        print(f"   Epoch [{epoch+1}/{epochs}] | 训练误差 (Loss): {running_loss/len(train_loader):.4f}")

elapsed = time.time() - start_time
print(f"✅ 深度学习训练完毕！耗时: {elapsed:.2f} 秒")

# ==========================================
# 5. 闭卷考试与终极对决
# ==========================================
print("\n[4/4] 神经网路正在进行闭卷考试...")
model.eval() # 开启考试模式
with torch.no_grad():
    y_pred_tensor = model(torch.tensor(X_test))
    y_pred = y_pred_tensor.numpy()

r2 = r2_score(y_test, y_pred)
print(f"🏆 1D-CNN 的最终考试成绩 (R² Score): {r2:.4f}")

if r2 > 0.1596:
    print(f"🚀 胜利！卷积神经网络成功超越了随机森林的 0.1596，算法升维有效！")
else:
    print("🤔 看来 5000 条数据依然不足以“喂饱”这个深度的网络，需要叠加路线 A (十万级数据) 才能产生质变。")
