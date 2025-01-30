#!/bin/bash
fastas_path=$(pwd)/example.fa
output_path=$(pwd)
batch_size=8
epoch=400
step=10
gen_interval=5
encode_method1="PC6"
echo ${encode_method1}
python train_1.py --f ${fastas_path} --o ${output_path} --b ${batch_size} --epoch ${epoch} --s ${step} --g ${gen_interval}