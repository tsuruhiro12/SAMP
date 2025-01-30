input_file = 'final_generated_seq.fasta'  # 入力ファイル名
output_file = 'formatted_output.txt'     # 出力ファイル名
# 重複を削除するためのセットを用意
unique_sequences = set()
with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    lines = infile.readlines()
    for i in range(1, len(lines), 2):  # 2行おきに処理（シーケンス部分を読み取る）
        sequence = lines[i].strip()  # 余分な空白を削除
        if sequence not in unique_sequences:  # 重複していない場合のみ処理
            unique_sequences.add(sequence)  # ユニークなシーケンスとして登録
            outfile.write(f"{sequence},1\n")  # 指定の形式で出力