import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import time

print("🧠 正在唤醒 T-SFE 专属随机森林 AI 模型...")

# ==========================================
# 1. 喂养数据
# ==========================================
csv_file = "tsfe_training_data_5000.csv"
try:
    df = pd.read_csv(csv_file)
    print(f"✅ 成功加载黄金数据集: {csv_file} (共 {len(df)} 条记录)")
except FileNotFoundError:
    print(f"❌ 找不到 {csv_file}，请确认它在这个文件夹里！")
    exit()

# ==========================================
# 2. 特征工程 (Feature Engineering)
# 将 ACGU 字母转换成 AI 能计算的 0 和 1 (One-Hot Encoding)
# ==========================================
def sequence_to_onehot(seq):
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'U': [0,0,0,1]}
    # 每条 43nt 的序列，会被展开成 43 * 4 = 172 个数字特征
    return np.array([mapping.get(base, [0,0,0,0]) for base in seq]).flatten()

print("\n[1/3] 正在将 RNA 序列翻译成 AI 矩阵...")
X = np.stack(df['Sequence'].apply(sequence_to_onehot).values)
y = df['Fidelity_Score'].values # 让 AI 学习如何预测“拓扑保真度”

# ==========================================
# 3. 划分考场与训练营
# ==========================================
# 拿出 80% (4000条) 给 AI 学习，留下 20% (1000条) 作为绝密考卷测试它的真实能力
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"📚 训练集: {len(X_train)} 条 | 📝 测试集 (闭卷考试): {len(X_test)} 条")

# ==========================================
# 4. 训练模型与闭卷考试
# ==========================================
print("\n[2/3] 轰鸣吧！随机森林正在疯狂生长，寻找碱基之间的非线性物理规律...")
start_time = time.time()

# 召唤拥有 100 棵决策树的随机森林，动用所有 CPU 核心 (n_jobs=-1)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

elapsed = time.time() - start_time
print(f"✅ 训练完毕！仅耗时: {elapsed:.2f} 秒")

print("\n[3/3] AI 正在进行闭卷考试...")
y_pred = rf_model.predict(X_test)

# 计算 R-squared (R平方) 得分，越接近 1.0 说明 AI 越聪明
r2 = r2_score(y_test, y_pred)
print(f"🏆 AI 的最终考试成绩 (R² Score): {r2:.4f}")

if r2 > 0.7:
    print("🌟 评价：不可思议！AI 已经敏锐地抓住了假结三维折叠的核心密码！")
elif r2 > 0.5:
    print("👍 评价：表现及格！AI 初步理解了物理规律。如果把数据量加到 50,000 条，它将化身神明！")
else:
    print("🤔 评价：差强人意。看来 5000 条数据对它来说还有点少。")

# ==========================================
# 5. 生物学洞察：AI 发现的“阿喀琉斯之踵”
# ==========================================
# AI 可以告诉我们，在这 43 个位置里，哪几个位置一旦突变，假结立刻崩溃
importances = rf_model.feature_importances_
position_importance = importances.reshape(-1, 4).sum(axis=1)

print("\n🧬 顶级机密：AI 破解的【关键结构位点 Top 5】")
top_indices = position_importance.argsort()[-5:][::-1]
for i, idx in enumerate(top_indices):
    print(f"  NO.{i+1} -> 第 {idx + 1} 位碱基 (重要性权重: {position_importance[idx]:.4f})")
