import random
import subprocess
import pandas as pd
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil
import sys

# ==========================================
# 1. 全局配置与靶点参数 (FIPV 假结基线)
# ==========================================
BASE_SEQ = "GCGGUUGCAUCUUGCAAAAAUGGUAUCGAAGGUACGAACAAUA"
IDEAL_DB = "(((((((.....[[[[[[......)))))))......]]]]]]"

TOTAL_SAMPLES = 5000
MUTATION_RATE = 0.15
NUM_WORKERS = 8


# ==========================================
# 2. 核心算法组件
# ==========================================
def mutate_sequence(seq, rate):
    bases = ['A', 'C', 'G', 'U']
    mutated = list(seq)
    for i in range(len(mutated)):
        if random.random() < rate:
            mutated[i] = random.choice([b for b in bases if b != seq[i]])
    return "".join(mutated)


def calculate_fidelity(pred_db, ideal_db):
    if not pred_db or len(pred_db) != len(ideal_db):
        return 0.0
    match_count = sum(1 for p, i in zip(pred_db, ideal_db) if p == i)
    base_score = match_count / len(ideal_db)
    if '[' not in pred_db or ']' not in pred_db:
        base_score *= 0.5
    return round(base_score, 4)


def run_rnapkplex(sequence):
    """调用底层的物理引擎，带上防弹级能量提取解析器"""
    try:
        process = subprocess.Popen(
            ['RNAPKplex'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        out, err = process.communicate(input=sequence + "\n")

        for line in out.strip().split('\n'):
            # 找到包含结构和能量的那一行
            if line.startswith('.') or line.startswith('(') or line.startswith('['):
                parts = line.split()
                db = parts[0]  # 第一部分永远是点括号拓扑图

                # 【防弹级解析】：把剩下的部分重新拼起来，精准提取括号里的数字
                rest_of_line = " ".join(parts[1:])
                energy = 0.0
                if '(' in rest_of_line and ')' in rest_of_line:
                    nrg_str = rest_of_line.split('(')[-1].split(')')[0].strip()
                    try:
                        energy = float(nrg_str)
                    except ValueError:
                        pass

                return db, energy

        return f"ERROR: {err.strip()}", 0.0
    except Exception as e:
        return f"CRASH: {str(e)}", 0.0


def process_single_task(seq):
    db, nrg = run_rnapkplex(seq)
    # 如果出错，直接抛弃这条数据
    if db is None or db.startswith("ERROR") or db.startswith("CRASH"):
        return None

    fid = calculate_fidelity(db, IDEAL_DB)
    return {
        'Sequence': seq,
        'Dot_Bracket': db,
        'MFE_kcal_mol': nrg,
        'Fidelity_Score': fid
    }


# ==========================================
# 3. 多核并发调度与主程序
# ==========================================
if __name__ == '__main__':
    print(f"🚀 启动 T-SFE 数据梦工厂...")

    # 【新增：发车前硬核引擎自检】
    print("\n[0/3] 正在进行 Linux 底层引擎自检...")
    engine_path = shutil.which('RNAPKplex')
    if not engine_path:
        print("❌ 致命错误：在 WSL 中找不到 'RNApkplex'！")
        sys.exit(1)

    # 抽查一条序列，看看引擎会不会报错
    test_db, _ = run_rnapkplex("GCGCGCGC")
    if test_db is None or test_db.startswith("ERROR") or test_db.startswith("CRASH"):
        print(f"❌ 引擎试运行失败，底层报错信息: {test_db}")
        sys.exit(1)
    print(f"✅ 自检通过！引擎绝对路径: {engine_path}")

    print("\n[1/3] 正在生成突变阵列...")
    sequences_to_test = set([BASE_SEQ])
    while len(sequences_to_test) < TOTAL_SAMPLES:
        sequences_to_test.add(mutate_sequence(BASE_SEQ, MUTATION_RATE))
    seq_list = list(sequences_to_test)

    print(f"\n[2/3] 轰鸣吧！物理引擎启动 (高 CPU 负载警告)...")
    start_time = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_single_task, seq): seq for seq in seq_list}
        for future in tqdm(as_completed(futures), total=len(futures), desc="折叠计算进度", unit="seq"):
            res = future.result()
            if res:  # 只收集非空（成功计算）的数据
                results.append(res)

    elapsed = time.time() - start_time

    print("\n[3/3] 正在打包黄金数据集...")

    # 【新增：防空壳崩溃保护】
    if not results:
        print("⚠️ 灾难性警报：5000 条序列全部计算失败，没有收集到任何有效数据！")
        sys.exit(1)

    df = pd.DataFrame(results)
    df = df.sort_values(by=['Fidelity_Score', 'MFE_kcal_mol'], ascending=[False, True])

    csv_filename = f"tsfe_training_data_{TOTAL_SAMPLES}.csv"
    df.to_csv(csv_filename, index=False)

    print("-" * 50)
    print(f"✅ 任务完成！耗时: {elapsed:.2f} 秒")
    print(f"📁 成功生成有效数据: {len(df)} 条。已导出至: {csv_filename}")
    print(f"🏆 来看一眼得分最高的完美突变体：")
    print(df.head(3))
