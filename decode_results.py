import os
import pandas as pd
from rdkit import Chem

print("🚀 正在启动 T-SFE 战报解码与身份溯源引擎...")

# ==========================================
# 1. 战术配置
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "HTVS_Results_fipv_target.csv")  # 你的对接战报
SDF_FILE = os.path.join(BASE_DIR, "SDF(HY-LD-000006870)-Mar 11, 2025.sdf")  # 厂家的原始集装箱
OUTPUT_CSV = os.path.join(BASE_DIR, "Final_Top_Hits_fipv.csv")  # 最终生成的完美战报

if not os.path.exists(CSV_FILE) or not os.path.exists(SDF_FILE):
    print("❌ 找不到 CSV 战报或原始 SDF 集装箱，请检查路径！")
    exit()

# ==========================================
# 2. 读取战报并提取密码坐标
# ==========================================
df = pd.read_csv(CSV_FILE)
# 只要那些超越了恩曲替尼的强者 (-10.56 以下的)
df_top = df[df["Binding_Energy(kcal/mol)"] <= -10.56].copy()
print(f"✅ 从 2915 个弹药中，成功过滤出 {len(df_top)} 个超越恩曲替尼的怪物级先导物！")

# ==========================================
# 3. 扫描原始集装箱，提取身份信息
# ==========================================
print("📦 正在扫描 MCE 原始军火库提取化学身份信息...")
supplier = Chem.SDMolSupplier(SDF_FILE)

# 准备用来装真实名字和 CAS 号的列表
real_names = []
cas_numbers = []

for index, row in df_top.iterrows():
    compound_id = row['Compound_Name']  # 例如 "HY_0191_Molecule"

    try:
        mol_idx = int(compound_id.split('_')[1]) - 1
    except:
        real_names.append("解析失败")
        cas_numbers.append("解析失败")
        continue

    mol = supplier[mol_idx]
    if mol is not None:
        name = "未知化合物"
        cas = "无记录"

        # 【精准提取护盾】只拿我们要的，无视其他带有乱码的字段！
        try:
            if mol.HasProp("Product Name"):
                name = mol.GetProp("Product Name")
            elif mol.HasProp("Name"):
                name = mol.GetProp("Name")
        except Exception:
            pass  # 如果名字本身带乱码，直接跳过报错

        try:
            if mol.HasProp("CAS No."):
                cas = mol.GetProp("CAS No.")
            elif mol.HasProp("CAS"):
                cas = mol.GetProp("CAS")
        except Exception:
            pass

        real_names.append(name)
        cas_numbers.append(cas)
    else:
        real_names.append("结构损坏")
        cas_numbers.append("未知")

# ==========================================
# 4. 生成终极战报并导出
# ==========================================
df_top['Real_Compound_Name'] = real_names
df_top['CAS_Number'] = cas_numbers

# 重新排个版，把最重要的信息放前面
df_final = df_top[['Real_Compound_Name', 'CAS_Number', 'Binding_Energy(kcal/mol)', 'Compound_Name']]

df_final.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

print("\n==========================================")
print("🎉 解码完成！")
print(f"📁 终极王者战报已生成: {OUTPUT_CSV}")
print("==========================================")

# 提前打印 Top 3 让你爽一下
print("\n🥇 揭开面纱的 Top 3 终极杀手:")
for i, row in df_final.head(3).iterrows():
    print(f"  💊 {row['Real_Compound_Name']} (CAS: {row['CAS_Number']}) => {row['Binding_Energy(kcal/mol)']} kcal/mol")
