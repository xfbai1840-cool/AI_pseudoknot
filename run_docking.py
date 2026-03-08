import os
import numpy as np
from rdkit import Chem
from meeko import MoleculePreparation
from vina import Vina

print("🚀 正在启动 T-SFE 分子对接引擎 (AutoDock Vina)...")

# ==========================================
# 1. 极客魔改：将靶点 PDB 转换为 Vina 专用的 PDBQT 格式
# ==========================================
def convert_rna_pdb_to_pdbqt(pdb_file, pdbqt_file):
    """为 RNA 靶点强行注入 Vina 兼容的部分电荷和原子类型"""
    coords = []
    with open(pdb_file, 'r') as f_in, open(pdbqt_file, 'w') as f_out:
        for line in f_in:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # 提取 X, Y, Z 三维坐标计算中心点
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                coords.append([x, y, z])
                
                # 提取元素并推断 AutoDock 识别的物理化学类型
                element = line[76:78].strip().upper()
                ad_type = element
                if element == 'C': ad_type = 'C'  # 碳骨架
                elif element == 'N': ad_type = 'NA' # 氢键受体/供体
                elif element == 'O': ad_type = 'OA' # 氧原子
                elif element == 'P': ad_type = 'P'  # 磷酸骨架
                elif element == 'H': ad_type = 'HD' # 质子
                
                # 严格按照 AutoDock 格式规范拼装字符串
                new_line = line[:66].ljust(66) + "   0.000 " + ad_type.ljust(2) + "\n"
                f_out.write(new_line)
    return np.mean(coords, axis=0) if coords else [0, 0, 0]

print("[1/3] 正在构建靶点三维空间竞技场...")
center_coords = convert_rna_pdb_to_pdbqt("fipv_target.pdb", "fipv_target.pdbqt")
print(f"🎯 竞技场中心已锁定: X={center_coords[0]:.2f}, Y={center_coords[1]:.2f}, Z={center_coords[2]:.2f}")

# ==========================================
# 2. 弹药装填：准备小分子配体
# ==========================================
def prepare_ligand(sdf_file, pdbqt_name):
    # 用 RDKit 读入小分子的 3D 构象
    supplier = Chem.SDMolSupplier(sdf_file, removeHs=False, sanitize=False)
    mol = supplier[0]
    if mol is None:
        raise ValueError(f"读取 {sdf_file} 失败，请检查文件。")
    Chem.SanitizeMol(mol)
    
    # 用 Meeko 为小分子计算电荷并识别可旋转的“柔性键”
    prep = MoleculePreparation()
    prep.prepare(mol)
    prep.write_pdbqt_file(pdbqt_name)
    return pdbqt_name

ligands = {
    "恩曲替尼 (Entrectinib)": "Entrectinib.sdf",
    "缬草三酯 (Valepotriate)": "valepotriate.sdf"
}

print("\n[2/3] 正在为小分子装填电荷并计算柔性扭转键...")
for name, file in ligands.items():
    prepare_ligand(file, file.replace('.sdf', '.pdbqt'))
    print(f"✅ {name} 弹药装填完毕！")

# ==========================================
# 3. 终极碰撞：执行 Vina 分子对接
# ==========================================
print("\n[3/3] 轰鸣吧！开始计算靶点与小分子的物理结合能...")
v = Vina(sf_name='vina')
v.set_receptor("fipv_target.pdbqt")

# 设置一个 40x40x40 埃的隐形盒子，足够包裹住 AlphaFold 生成的整个假结
v.compute_vina_maps(center=center_coords, box_size=[40.0, 40.0, 40.0])

results = {}
for name, file in ligands.items():
    pdbqt_file = file.replace('.sdf', '.pdbqt')
    print(f"\n▶ 正在将 {name} 射入假结口袋...")
    
    v.set_ligand_from_file(pdbqt_file)
    # exhaustiveness=8：搜索深度，数字越大算得越精细
    v.dock(exhaustiveness=8, n_poses=1) 
    
    # 提取结合能 (Binding Affinity)，单位是 kcal/mol
    energy = v.energies(n_poses=1)[0][0] 
    results[name] = energy
    
    # 将小分子成功卡入靶点后的 3D 构象保存下来，稍后可以用 PyMOL 观赏！
    out_file = file.replace('.sdf', '_docked.pdbqt')
    v.write_poses(out_file, n_poses=1, overwrite=True)

# ==========================================
# 4. 打印战报
# ==========================================
print("\n==========================================")
print("🏆 终极对接战报 (结合能越低，说明卡得越死！)")
print("==========================================")
for name, energy in sorted(results.items(), key=lambda x: x[1]):
    print(f"💊 {name}: {energy:.2f} kcal/mol")
print("==========================================")
print("💡 评价标准: ")
print("  -5.0 kcal/mol -> 勉强碰上")
print("  -7.0 kcal/mol -> 强效结合 (具备成药潜力)")
print("  -9.0 kcal/mol -> 完美神药 (死死锁住！)")
