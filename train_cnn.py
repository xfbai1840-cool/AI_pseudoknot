import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import time

print("🧠 正在加载【抗过拟合版】T-SFE 深度卷积神经网络...")

# ==========================================
# 1. 加载与预处理数据 (保持不变)
# ==========================================
df = pd.read_csv("tsfe_training_data_5000.csv")

def sequence_to_cnn_matrix(seq):
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'U': [0,0,0,1]}
    mat = np.array([mapping.get(base, [0,0,0,0]) for base in seq])
    return mat.T 

X = np.stack(df['Sequence'].apply(sequence_to_cnn_matrix).values).astype(np.float32)
y = df['Fidelity_Score'].values.astype(np.float32).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# ==========================================
# 2. 架构升级：加入 Dropout 与 BatchNorm
# ==========================================
class RNACNN_Robust(nn.Module):
    def __init__(self):
        super(RNACNN_Robust, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16) # 新增：批归一化，防止数据剧烈震荡
        self.relu = nn.ReLU()
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.dropout = nn.Dropout(p=0.5) # 核心魔法：50% 随机失活神经元
        
        self.fc1 = nn.Linear(32 * 43, 64) 
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1) 
        
        x = self.dropout(x) # 第一次打断死记硬背
        x = self.relu(self.fc1(x))
        x = self.dropout(x) # 第二次打断
        
        x = self.fc2(x)
        return x

model = RNACNN_Robust()
print("\n[1/3] 网络搭建完毕！已部署 Dropout 与 BatchNorm 护盾。")

# ==========================================
# 3. 优化器升级：加入权重惩罚 (Weight Decay)
# ==========================================
criterion = nn.MSELoss() 
# 新增 weight_decay=1e-3：逼迫 AI 寻找最简单的结构规律
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)

# ==========================================
# 4. 训练循环
# ==========================================
epochs = 80 
print(f"\n[2/3] 轰鸣吧！开始 {epochs} 轮的硬核训练...")

start_time = time.time()
for epoch in range(epochs):
    model.train() # 开启训练模式 (Dropout 激活)
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()    
        outputs = model(inputs)  
        loss = criterion(outputs, labels) 
        loss.backward()          
        optimizer.step()         
        running_loss += loss.item()
        
    if (epoch + 1) % 20 == 0:
        print(f"   Epoch [{epoch+1}/{epochs}] | 训练误差 (Loss): {running_loss/len(train_loader):.4f}")

print(f"✅ 深度学习训练完毕！耗时: {time.time() - start_time:.2f} 秒")

# ==========================================
# 5. 闭卷考试与终极对决
# ==========================================
print("\n[3/3] AI 正在进行闭卷考试...")
model.eval() # ⚠️ 极其关键：考试时必须关闭 Dropout，让所有神经元全力应战！
with torch.no_grad():
    y_pred_tensor = model(torch.tensor(X_test))
    y_pred = y_pred_tensor.numpy()

r2 = r2_score(y_test, y_pred)
print(f"🏆 抗过拟合版 1D-CNN 的最终考试成绩 (R² Score): {r2:.4f}")

if r2 > 0:
    print(f"🚀 胜利！成功打破死记硬背！分数从负数拉回到了正数！")
else:
    print("🤔 过拟合依然严重，或者 5000 条数据对卷积网络来说真的太少了。建议开启十万级数据工厂模式。")
