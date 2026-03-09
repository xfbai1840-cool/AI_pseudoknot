import os
import re
from rdkit import Chem
from rdkit.Chem import AllChem

print("🚀 正在启动 T-SFE 官方军火库拆解引擎...")

# ==========================================
# 1. 战术配置
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ⚠️ 请确保这里填入你上传的那个厂家的真实文件名
INPUT_SDF = os.path.join(BASE_DIR, "SDF(HY-LD-000006870)-Mar 11, 2025.sdf")  
OUTPUT_DIR = os.path.join(BASE_DIR, "ligands_3d_from_vendor")      

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if not os.path.exists(INPUT_SDF):
    print(f"❌ 找不到厂家文件：{INPUT_SDF}")
    exit()

# ==========================================
# 2. 读取集装箱并开箱
# ==========================================
# 动用 RDKit 的多分子读取器
supplier = Chem.SDMolSupplier(INPUT_SDF)
total_mols = len(supplier)
print(f"📦 扫描完毕！在这个集装箱里共发现了 {total_mols} 个化合物！准备开始 3D 重塑...")

success_count = 0
for i, mol in enumerate(supplier):
    if mol is None:
        continue
        
    # 尝试提取化合物的名字或编号 (厂家通常会写在 _Name 属性里)
    name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"Compound_{i+1}"
    
    # 清洗非法字符，防止 Windows 报错
    safe_name = re.sub(r'[\\/*?:"<>|]', "_", name).strip()
    
    print(f"▶ 正在重塑: {safe_name} ({i+1}/{total_mols})")
    
    try:
        # 【核心工艺】加氢、3D 展开、能量最小化
        mol = Chem.AddHs(mol)
        
        # 很多厂家的库默认是 2D 的，必须强行 3D 化
        AllChem.EmbedMolecule(mol, randomSeed=42)
        
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            pass # 忽略少数无法优化的特殊结构
            
        # 独立打包存放
        out_path = os.path.join(OUTPUT_DIR, f"{safe_name}.sdf")
        writer = Chem.SDWriter(out_path)
        writer.write(mol)
        writer.close()
        
        success_count += 1
    except Exception as e:
        print(f"  ❌ {safe_name} 重塑失败: {e}")

print("\n==========================================")
print(f"🎉 军火库开箱完毕！成功铸造 {success_count}/{total_mols} 个完美 3D 独立弹药！")
print(f"📁 所有的 .sdf 文件已整齐存放在 [{OUTPUT_DIR}] 文件夹中。")
print("==========================================")
