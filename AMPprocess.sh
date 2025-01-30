#!/bin/bash
kinds=AMP
species=md
oufsuffix=2
outsuffix=2
kfold=5
seqwin=183

w2v_path=../w2v_model
#w2v_bpe_model_file=/home/kurata/myproject/common/subword/w2v_bpe/w2v_bpe_400_64_4_50_1.pt
#bpe_model_file=/home/kurata/myproject/common/subword/model/bpe_model_400.model
# esm2_dict_file=/mnt/tsustu/Study1/esm2/noAMP_seq2esm2_dict/
esm2_dict_file=/mnt/tsustu/Study1/esm2/AMP_seq2esm2_dict/
prott5_file=/mnt/tsustu/Study1/ProtTrans/GAN_AMP_out.h5
# Users must build an esm2_dict_file for themselves, because it requires >60GB and it is hard to store here. See the esm2 directory.
# If users remove esm2, they must remove ESM2 from the encoding methods and set "esm2_dict_file=None".

space=" "
machine_method_1="LGBM XGB SVM RF NB KN LR"
encode_method_1="AAC DPC PAAC CTDC CTDT CTDD CKSAAP GAAC GDPC GTPC CTriad BE EAAC BLOSUM62 ZSCALE AAINDEX"
w2v_encode="W2V_1_128_100_40_1 W2V_2_128_100_40_1 W2V_3_128_100_40_1" 
# W2V_3_128_100_40_1
encode_method_1w=${encode_method_1}$space${w2v_encode}

machine_method_2="TX bLSTM CNN"
encode_method_2="prott5 BE NN"
# "BE NN"
encode_method_2w=${encode_method_2}$space${w2v_encode}

total_num=158
#total_num = 7*(17+3) + 3*(3+3) = 140+18 = 158  
#total_num = 7*(16+3) + 3*(3+3) = 133+18 = 151 when users remove EMS2.  

cd ..
main_path=`pwd`
echo ${main_path}

########## DATA SETTING ##########

test_fasta=${main_path}/data/independent_test/AMPindependent_test.fa
test_csv=${main_path}/data/independent_test/AMPindependent_test.csv

cd program
cd ml

########## MACHINE LEARNING ##########

train_path=${main_path}/data/AMPcross_val
result_path=${main_path}/data/AMPresult_${species}
esm2_dict=${esm2_dict_file}


# for machine_method in ${machine_method_1}
# do

#     for encode_method in ${encode_method_1}
#     do
#     kmer=1
#     w2v_model=None
#     size=-1
#     epochs=-1
#     window=-1
#     sg=-1
#     echo ${machine_method} ${encode_method}
#     python ml_train_test_45.py  --intrain ${train_path} --intest ${test_csv} --outpath ${result_path} --machine ${machine_method}  --encode ${encode_method} --kfold ${kfold} --seqwin ${seqwin} --kmer ${kmer} --w2vmodel ${w2v_model} --size ${size} --epochs ${epochs} --sg ${sg} --window ${window} --esm2 ${esm2_dict} --kinds ${kinds} --prott5 ${prott5_file}
#     done

#     encode_method=W2V
#     size=128
#     epochs=100
#     window=40
#     sg=1
#     for kmer in 1 2 3 #4
#     do
#     w2v_model=${w2v_path}/W2V_general_${kmer}_128_100_40_1.pt
#     echo ${machine_method} ${encode_method} ${kmer}
#     python ml_train_test_45.py  --intrain ${train_path} --intest ${test_csv} --outpath ${result_path} --machine ${machine_method}  --encode ${encode_method} --kfold ${kfold} --seqwin ${seqwin} --kmer ${kmer} --w2vmodel ${w2v_model} --size ${size} --epochs ${epochs} --sg ${sg} --window ${window} --esm2 ${esm2_dict}
#     done

# done
cd ..
cd network

######### DEEP LEARNING ##########

for deep_method in ${machine_method_2}
do

    # for encode_method in ${encode_method_2}
    # do
    # echo ${deep_method}: ${encode_method}

    # if [ $encode_method = BE ]; then
    # kmer=1
    # size=25
    # epochs=-1
    # window=-1
    # sg=-1
    # w2v_model=None
    # w2v_bpe_model=None
    # bpe_model=None
    # esm2_dict=None

    # elif [ $encode_method = NN ]; then
    # kmer=1
    # size=64
    # epochs=-1
    # window=-1
    # sg=-1
    # w2v_model=None
    # w2v_bpe_model=None
    # bpe_model=None
    # esm2_dict=None

    # elif [ $encode_method = W2V_BPE ]; then 
    # kmer=1
    # size=64
    # epochs=-1
    # window=-1
    # sg=-1
    # w2v_bpe_model=${w2v_bpe_model_file}
    # bpe_model=${bpe_model_file}
    # w2v_model=None
    # esm2_dict=None

    # elif [ $encode_method = prott5 ]; then
    #     kmer=1
    #     size=1024
    #     epochs=-1
    #     window=-1
    #     sg=-1
    #     w2v_model=None
    #     w2v_bpe_model=None
    #     bpe_model=None
    #     esm2_dict=None
    #     prott5_file=${prott5_file}


    # elif [ $encode_method = ESM2 ]; then
    #     kmer=1
    # if [ $deep_method = TX ]; then 
    #     size=128
    # else
    #     size=320
    #     # 1280
    # fi
    # epochs=-1
    # window=-1
    # sg=-1
    # w2v_model=None
    # w2v_bpe_model=None
    # bpe_model=None
    # esm2_dict=${esm2_dict_file}

    # else
    # echo no encode method in script
    # fi

    # python train_test_86.py  --intrain ${train_path} --intest ${test_csv} --outpath ${result_path} --losstype "balanced" --deeplearn ${deep_method}  --encode ${encode_method} --kfold ${kfold} --w2vmodel ${w2v_model} --seqwin ${seqwin} --kmer ${kmer} --size ${size} --epochs ${epochs} --sg ${sg} --window ${window} --w2v_bpe_model ${w2v_bpe_model} --bpe_model ${bpe_model} --esm2 ${esm2_dict} --prott5 ${prott5_file}
    # done

    encode_method=W2V
    for kmer in 1 2 3 #4
    do
    echo ${deep_method}: ${encode_method}: ${kmer}
    size=128
    epochs=100
    window=40
    sg=1
    w2v_model=${w2v_path}/W2V_general_${kmer}_128_100_40_1.pt
    w2v_bpe_model=None
    bpe_model=None
    esm2_dict=None
    python train_test_86.py  --intrain ${train_path} --intest ${test_csv} --outpath ${result_path} --losstype "balanced" --deeplearn ${deep_method}  --encode ${encode_method} --kfold ${kfold} --w2vmodel ${w2v_model} --seqwin ${seqwin} --kmer ${kmer} --size ${size} --epochs ${epochs} --sg ${sg} --window ${window} --w2v_bpe_model ${w2v_bpe_model} --bpe_model ${bpe_model} --esm2 ${esm2_dict} --prott5 ${prott5_file}
    done

done
cd ..

######### ENSEMBLE LEARNING ##########

echo evaluation
python analysis_622.py --machine_method_1 "${machine_method_1}" --encode_method_1 "${encode_method_1w}" --machine_method_2 "${machine_method_2}" --encode_method_2 "${encode_method_2w}" --species ${species}  --kinds ${kinds}

outfile=183_GAN0.85result_${oufsuffix}.xlsx
python csv_xlsx_34.py --machine_method_1 "${machine_method_1}" --encode_method_1 "${encode_method_1w}" --machine_method_2 "${machine_method_2}" --encode_method_2 "${encode_method_2w}" --species ${species} --outfile ${outfile} --kinds ${kinds}

echo ensemble
meta=LR
prefix=183_GAN_0.85seq_impdec23_combine
python ml_fusion_642.py --machine_method_1 "${machine_method_1}" --encode_method_1 "${encode_method_1w}" --machine_method_2 "${machine_method_2}" --encode_method_2 "${encode_method_2w}" --species ${species} --total_num ${total_num} --meta ${meta} --prefix ${prefix} --kinds ${kinds}

outfile=183_GAM_0.85result_stack_${outsuffix}.xlsx
python csv_xlsx_37.py --species ${species} --outfile ${outfile} --meta ${meta} --prefix ${prefix} --kinds ${kinds}




