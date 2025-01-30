from collections import Counter
import matplotlib.pyplot as plt

def calculate_length_statistics_from_fasta(file_path):
    # 配列長を保持するリスト
    lengths = []
    # FASTAファイルを読み込む
    with open(file_path, "r") as fasta_file:
        for line in fasta_file:
            # ヘッダー行をスキップ（">"で始まる行）
            if line.startswith(">"):
                continue
            # 配列の長さを計算してリストに追加
            lengths.append(len(line.strip()))
    
    # 配列長の統計情報を計算
    if lengths:
        average_length = sum(lengths) / len(lengths)
        max_length = max(lengths)
        min_length = min(lengths)
        # 最頻値を計算（複数ある場合は最初のものを取得）
        most_common_length = Counter(lengths).most_common(1)[0][0]
    else:
        average_length = 0
        max_length = 0
        min_length = 0
        most_common_length = 0
    
    return lengths, average_length, max_length, min_length, most_common_length

def save_histogram(lengths, output_path):
    # ヒストグラムをプロットして保存
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=20, color="skyblue", edgecolor="black")
    plt.title("Distribution of Sequence Lengths")
    plt.xlabel("Sequence Length")
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()  # プロットを閉じてメモリを節約

# FASTAファイルのパス
fasta_file_path = "example.fa"
# ヒストグラム画像の保存パス
output_histogram_path = "sequence_length_histogram.png"

# 統計情報を計算
lengths, average_length, max_length, min_length, most_common_length = calculate_length_statistics_from_fasta(fasta_file_path)

# 結果を表示
print(f"配列の長さの平均: {average_length:.2f}")
print(f"配列の最大長: {max_length}")
print(f"配列の最小長: {min_length}")
print(f"配列の最頻長: {most_common_length}")

# ヒストグラムを保存
save_histogram(lengths, output_histogram_path)
print(f"ヒストグラムを保存しました: {output_histogram_path}")
