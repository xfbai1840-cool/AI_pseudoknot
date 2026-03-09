import os
import re
import time
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import AllChem

print("🚀 正在启动 T-SFE 高通量弹药库获取与 3D 建模引擎 (终极防弹版)...")

# ==========================================
# 1. 战术配置
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "my_compounds.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "ligands_3d")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

try:
    try:
        df = pd.read_csv(CSV_FILE, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(CSV_FILE, encoding='gbk')
        except UnicodeDecodeError:
            df = pd.read_csv(CSV_FILE, encoding='utf-8-sig')

    print(f"✅ 成功加载清单，共发现 {len(df)} 个分子！")
    print("⏳ 为防止触发反爬虫封禁，程序将温和抓取。2915 个分子预计需要 15-20 分钟，请挂机等待。")
except Exception as e:
    print(f"❌ 致命错误：读取表格崩溃。真实报错是: {e}")
    exit()


# ==========================================
# 2. 核心引擎：双库逆向获取与断点续传
# ==========================================
def process_compound(name, cas):
    # 【极其关键】清洗 Windows/Linux 非法字符，防止系统级崩溃！
    safe_name = re.sub(r'[\\/*?:"<>|]', "", name).strip()
    if not safe_name:
        safe_name = f"Compound_{cas}"

    print(f"\n▶ 正在追踪: {safe_name} (CAS: {cas})")
    output_file = os.path.join(OUTPUT_DIR, f"{safe_name}.sdf")

    # 🌟 断点续传逻辑：如果文件已存在，直接无视并跳过
    if os.path.exists(output_file):
        print(f"  ⏭️ 本地弹药库已存在该 3D 模型，直接跳过！")
        return True

    smiles = None

    # 引擎 A: 强闯 PubChem (同时请求立体和平面构象)
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas}/property/IsomericSMILES,CanonicalSMILES/JSON"
        res = requests.get(url, timeout=10)
        time.sleep(0.3)  # 遵守服务器协议，强行休息 0.3 秒，否则会被立刻封锁 IP

        if res.status_code == 200:
            data = res.json()
            if 'PropertyTable' in data and len(data['PropertyTable']['Properties']) > 0:
                props = data['PropertyTable']['Properties'][0]
                # 优先获取立体结构，退而求其次获取平面结构
                smiles = props.get('IsomericSMILES', props.get('CanonicalSMILES'))
    except Exception:
        pass

    # 引擎 B: 备用降落伞 - NIH Cactus
    if not smiles:
        try:
            cactus_url = f"https://cactus.nci.nih.gov/chemical/structure/{cas}/smiles"
            res_cactus = requests.get(cactus_url, timeout=10)
            if res_cactus.status_code == 200:
                smiles = res_cactus.text.strip()
                print("  💡 PubChem 掩护失败，已切换至 NIH 备用服务器截获密码！")
        except:
            pass

    if not smiles:
        print(f"  ❌ 两大全球数据库均未能破译该 CAS 号 ({cas})，抓取失败。")
        return False

    print(f"  ✅ 成功截获 1D 密码: {smiles[:25]}...")

    # 3D 空间建模与优化
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print("  ❌ RDKit 拒绝解析该 SMILES，结构可能非法。")
            return False

        mol = Chem.AddHs(mol)

        # 将 1D 拓扑图强行在 3D 坐标系中折叠展开
        embed_res = AllChem.EmbedMolecule(mol, randomSeed=42)
        if embed_res != 0:
            print("  ❌ 3D 空间折叠失败 (分子太大、几何受限或为盐类混合物)。")
            return False

        # MMFF94 物理力场能量最小化
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            pass  # 有些金属原子力场不支持，忽略并继续保存

        writer = Chem.SDWriter(output_file)
        writer.write(mol)
        writer.close()
        print(f"  🏆 3D 物理模型已完美铸造 -> {output_file}")
        return True
    except Exception as e:
        print(f"  ❌ 3D 建模过程发生意外碰撞: {e}")
        return False


# ==========================================
# 3. 启动高通量流水线
# ==========================================
success_count = 0
for index, row in df.iterrows():
    if 'Compound_Name' not in row or 'CAS_Number' not in row:
        print("\n❌ 表格格式错误：找不到 'Compound_Name' 或 'CAS_Number' 这两列！")
        exit()

    name = str(row['Compound_Name']).strip()
    cas = str(row['CAS_Number']).strip()

    if pd.isna(name) or name == 'nan' or name == '': continue

    if process_compound(name, cas):
        success_count += 1

print("\n==========================================")
print(f"🎉 高通量弹药库获取完毕！成功铸造 {success_count}/{len(df)} 个 3D 结构！")
print(f"📁 所有的 .sdf 文件已存放在 [{OUTPUT_DIR}] 文件夹中。")
print("==========================================")
