def convert_fasta_to_csv(input_file, output_file):
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        sequence = ""
        label = ""
        
        for line in infile:
            line = line.strip()
            if line.startswith(">"):
                # 新しいシーケンスの開始
                if sequence:
                    # 前のシーケンスとラベルを書き込み
                    outfile.write(f"{sequence},{label}\n")
                # 現在のラベルを取得
                label = line.split("|")[-1]
                sequence = ""
            else:
                # シーケンス行を取得
                sequence = line
        
        # 最後のシーケンスを書き込み
        if sequence:
            outfile.write(f"{sequence},{label}\n")


# 入力ファイルと出力ファイルのパスを指定
input_file_path = "/mnt/tsustu/Study1/data/test.txt"  # 例: "input_file.txt"
output_file_path = "/mnt/tsustu/Study1/data/AMP_1_test.txt"  # 例: "output_file.txt"

input_file_path2 = "/mnt/tsustu/Study1/data/train.txt"  # 例: "input_file.txt"
output_file_path2 = "/mnt/tsustu/Study1/data/AMP_1_train.txt"  # 例: "output_file.txt"

convert_fasta_to_csv(input_file_path, output_file_path)
convert_fasta_to_csv(input_file_path2, output_file_path2)
print(f"データが {output_file_path} に保存されました。")
