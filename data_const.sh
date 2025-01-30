#!/bin/sh
#main_path=`pwd`
main_path="/mnt/tsustu/Study1"
echo ${main_path}
# infile1=${main_path}/data/train.txt
# infile2=${main_path}/data/test.txt
outfile1=${main_path}/data/AMP_1_train.txt
outfile2=${main_path}/data/AMP_1_test.txt
test_fasta1=${main_path}/data/independent_test/AMPindependent_test.fa
test_csv1=${main_path}/data/independent_test/AMPindependent_test.csv
data_path=${main_path}/data
kfold=5
# python data_standard.py --infile1 ${infile1} --infile2 ${infile2} --outfile1 ${outfile1} --outfile2 ${outfile2}
python ${main_path}"/train_division_1.py" --infile2 ${outfile1} --datapath ${data_path} --kfold ${kfold}
# python ${main_path}"/test_fasta.py" --infile1 ${outfile2} --outfile1 ${test_csv1} --outfile2 ${test_fasta1} 