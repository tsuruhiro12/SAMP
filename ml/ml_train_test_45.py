import sys
import os
import time
import argparse
import pandas as pd
import numpy as np
import torch
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import xgboost as xgb
import lightgbm as lgb
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from gensim.models import word2vec #追加しました
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import r2_score,  mean_squared_error, mean_absolute_error
from encodingAA_23 import AAC, DPC, TPC, PAAC, EAAC, CKSAAP, CKSAAGP, GAAC, GDPC, GTPC, CTDC, CTDT,CTDD, CTriad, BLOSUM62, ZSCALE,AAINDEX
import h5py,re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def data_pre(sequences, feature_file, theta=183, shuffle_seed=11, mode='train'):
    # fasta_file_path = f'{opt.input_path}/{fasta_file}'


    # with h5py.File(feature_file, "r") as f:
        # キー一覧（データセット名）
        # keys = list(f.keys())
        # print(keys)  # ['seq1', 'seq2']

        # # 'seq1' にアクセスして中身を読む
        # data_seq1 = f["ATAVDFGPHGLLPIRPIRIRPLCGKDKS"][:]  # 文字列なら np.bytes_、配列なら ndarray などになる
        # print(data_seq1)


    sequences_no = [re.sub('X', '', seq) for seq in sequences]
    pre_feas = get_pretrained_features(sequences_no, feature_file, theta=theta)
    return pre_feas

def get_pretrained_features(sequences, pre_dict_path, theta=183):
    features = []
    with h5py.File(pre_dict_path, "r") as h5fi:
        for seq in sequences:
            pre_features_ref = h5fi[seq][:]
            sequence_len = len(seq)
            feature = np.zeros((theta, pre_features_ref.shape[-1]), dtype=float)
            for i in range(min(sequence_len, theta)):
                feature[i, :] = pre_features_ref[i, :]
            features.append(feature)
    features = np.array(features)

    return features


def emb_seq_w2v(seq_mat, w2v_model, num, size):
    num_sample = len(seq_mat)
    seq_emb = []  
    for j in range(num_sample):
      seq=seq_mat[j]
      seq_w2v= []
      for i in range(len(seq) - num + 1):
        try:
            x = w2v_model.wv[seq[i:i+num]]
        except:
            x = np.zeros(size)
        seq_w2v.append(x)
      seq_w2v = np.array(seq_w2v, dtype=float)
      seq_emb.append(seq_w2v)
    seq_emb = np.array(seq_emb, dtype=float)
    seq_emb = seq_emb.reshape(num_sample,len(seq) - num + 1, -1) 
    #print(seq_emb.shape)
    seq_emb = seq_emb.reshape(num_sample,1,-1).squeeze()
    #print(seq_emb.shape)
    
    return seq_emb

def emb_seq_BE(seq_mat, aa_dict, num):
   num_sample = len(seq_mat) 
   for j in range(num_sample):
      seq = seq_mat[j]
      if j == 0:
         seq_emb = np.array([np.array([aa_dict[seq[i + k]] for k in range(num)]).reshape([21 * num]) for i in range(len(seq) - num + 1)])
      else : 
         seq_enc = np.array([np.array([aa_dict[seq[i + k]] for k in range(num)]).reshape([21 * num]) for i in range(len(seq) - num + 1)])
         seq_emb = np.append(seq_emb, seq_enc, axis=0)
   seq_emb = seq_emb.reshape(num_sample, len(seq) - num + 1, -1)  #1088 x 35 x 100
   seq_emb = seq_emb.reshape(num_sample, 1, -1).squeeze() #1088 X3500
   return seq_emb

def emb_seq_ESM2(seq_mat, seq2esm2_dict):
    esm2_mat =[]
    for seq in seq_mat:
        esm2_mat.append(seq2esm2_dict[seq.replace('X','')].squeeze(0).numpy().copy().reshape(1,-1).squeeze())
    seq_emb = np.array(esm2_mat)
    return seq_emb



def aa_dict_construction():
   #AA = 'ARNDCQEGHILKMFPSTWYV'
   AA = 'ARNDCQEGHILKMFPSTWYV'
   keys=[]
   vectors=[]
   for i, key in enumerate(AA) :
      base=np.zeros(21)
      keys.append(key)
      base[i]=1
      vectors.append(base)
   aa_dict = dict(zip(keys, vectors))
   aa_dict["X"] = np.zeros(21)
   return aa_dict


def pad_input_csv(filename, seqwin, index_col = None):
    df1 = pd.read_csv(filename, delimiter=',',index_col = index_col)
    seq = df1.loc[:,'seq'].tolist()
    #data triming and padding
    for i in range(len(seq)):
       if len(seq[i]) > seqwin:
         seq[i]=seq[i][0:seqwin]
       seq[i] = seq[i].ljust(seqwin, 'X')
       
    for i in range(len(seq)):
       df1.loc[i,'seq'] = seq[i]   
    return df1
      
def pickle_save(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)

def pickle_read(path):
    with open(path, "rb") as f:
        res = pickle.load(f)      
    return res
    
def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj,f)

def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data
if __name__=="__main__":
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--intrain', help='Path')
    parser.add_argument('--intest', help='Path')
    parser.add_argument('--outpath', help='Path')
    parser.add_argument('--losstype', help='Path', default = "balanced", choices=["balanced", "imbalanced"])
    parser.add_argument('--w2vmodel', help='Path')
    parser.add_argument('--machine', help='Path')
    parser.add_argument('--encode', help='Path')
    parser.add_argument('--kfold', type=int, help='Path')
    parser.add_argument('--seqwin', type=int, help='Path')
    parser.add_argument('--kmer', type=int, default=1, help='Path')
    parser.add_argument('--size', type=int, help='Path')
    parser.add_argument('--epochs', type=int, help='Path')
    parser.add_argument('--sg', type=int, help='Path')
    parser.add_argument('--window', type=int, help='Path')
    parser.add_argument('--esm2', help='Path')
    parser.add_argument('--prott5', help='Path')
    parser.add_argument('--kinds', type=str, help='term')
    
    path = parser.parse_args().intrain
    test_file = parser.parse_args().intest
    out_path_0 = parser.parse_args().outpath
    loss_type = parser.parse_args().losstype
    w2v_model = parser.parse_args().w2vmodel
    machine_method = parser.parse_args().machine
    encode_method = parser.parse_args().encode
    seq2esm2_dict_file = parser.parse_args().esm2
    prott5 = parser.parse_args().prott5
    
    kfold = parser.parse_args().kfold
    seqwin = parser.parse_args().seqwin
    kmer = parser.parse_args().kmer
    size = parser.parse_args().size
    epochs=parser.parse_args().epochs
    sg = parser.parse_args().sg
    window = parser.parse_args().window
    kinds = parser.parse_args().kinds
        
    
    
    if encode_method == 'W2V':
        #w2v_model = w2v_path + '/' + 'av_w2v_%s_100_4_20_1.pt' %kmer
        w2v_model = word2vec.Word2Vec.load(w2v_model)
        os.makedirs(out_path_0 + '/' + machine_method + '/' + encode_method + '_' + str(kmer) + '_' + str(size) + '_' + str(epochs) + '_' + str(window) + '_' + str(sg), exist_ok=True)
        out_path =  out_path_0 + '/' + machine_method + '/' + encode_method + '_' + str(kmer) + '_' + str(size) + '_' + str(epochs) + '_' + str(window) + '_' + str(sg)
    elif encode_method == 'BE':
        os.makedirs(out_path_0 + '/' + machine_method + '/' + encode_method, exist_ok=True)
        out_path =  out_path_0 + '/' + machine_method + '/' + encode_method
    elif encode_method == 'ESM2':
        w2v_model = []
        #esm2_model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        file_list = os.listdir(seq2esm2_dict_file)
        seq2esm2_dict = {}
        for file_esm in file_list:
            esm_emb_path = f"{seq2esm2_dict_file}/{file_esm}"
            is_file = os.path.isfile(esm_emb_path)
            if is_file:
                with open(esm_emb_path, 'rb') as f:
                    seq2esm2_dict_tmp = pickle.load(f)
            else:
                print("No seq2esm2_dict.pkl")
                exit()
            seq2esm2_dict.update(seq2esm2_dict_tmp)
        os.makedirs(out_path_0 + '/' + machine_method + '/' + encode_method, exist_ok=True)
        out_path =  out_path_0 + '/' + machine_method + '/' + encode_method    
    # elif encode_method == 'ESM2':
    #     w2v_model = []
    #     #esm2_model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_8M_UR50D")
    #     is_file = os.path.isfile(seq2esm2_dict_file)
    #     if is_file:
    #         with open(seq2esm2_dict_file, 'rb') as f:
    #             seq2esm2_dict = pickle.load(f)
    #     else:
    #         print("No seq2esm2_dict.pkl")
    #         exit()
    #     os.makedirs(out_path_0 + '/' + machine_method + '/' + encode_method, exist_ok=True)
    #     out_path =  out_path_0 + '/' + machine_method + '/' + encode_method

    elif encode_method == "prott5":
        
        os.makedirs(out_path_0 + '/' + machine_method + '/' + encode_method, exist_ok=True)
        out_path =  out_path_0 + '/' + machine_method + '/' + encode_method


    else:
        os.makedirs(out_path_0 + '/' + machine_method + '/' + encode_method, exist_ok=True)
        out_path =  out_path_0 + '/' + machine_method + '/' + encode_method

    for i in range(1, kfold+1):
        os.makedirs(out_path + "/" + str(i) + "/data_model", exist_ok=True)
        modelname= "machine_model.sav"

        train_dataset = pad_input_csv(path + "/" + str(i) + "/cv_train_" + str(i) + ".csv", seqwin, index_col = None)
        val_dataset = pad_input_csv(path + "/" + str(i) + "/cv_val_" + str(i) + ".csv", seqwin, index_col = None)
        test_dataset = pad_input_csv(test_file, seqwin, index_col = None)

        train_seq = train_dataset['seq'].tolist()
        val_seq = val_dataset['seq'].tolist()
        test_seq = test_dataset['seq'].tolist()
    
        
        if encode_method == 'W2V':
            train_X = emb_seq_w2v(train_seq, w2v_model, kmer, size)
            valid_X = emb_seq_w2v(val_seq, w2v_model, kmer, size)       
            test_X = emb_seq_w2v(test_seq, w2v_model, kmer, size)
         
        elif encode_method == 'BE':
            aa_dict = aa_dict_construction()
            train_X = emb_seq_BE(train_seq, aa_dict, kmer)
            valid_X = emb_seq_BE(val_seq, aa_dict, kmer)
            test_X = emb_seq_BE(test_seq, aa_dict, kmer)

        elif encode_method == 'ESM2':
            train_X = emb_seq_ESM2(train_seq, seq2esm2_dict)
            valid_X = emb_seq_ESM2(val_seq, seq2esm2_dict)
            test_X = emb_seq_ESM2(test_seq, seq2esm2_dict)  
        
        elif encode_method == 'prott5':
            
            train_X = data_pre(train_seq, prott5)
            valid_X = data_pre(val_seq, prott5)
            test_X = data_pre(test_seq, prott5)
            print(train_X.shape)  
            train_X = train_X.reshape(train_X.shape[0], -1)
            valid_X = valid_X.reshape(valid_X.shape[0], -1)
            test_X = test_X.reshape(test_X.shape[0], -1)

           
            # print(valid_X.shape)  
            # print(test_X.shape) 
      
        else:
            myOrder = 'ARNDCQEGHILKMFPSTWYV' #'ACDEFGHIKLMNPQRSTVWY'
            kw = {'order': myOrder, 'type': 'Protein'}
            train_seq=train_dataset.values.tolist()
            val_seq  =val_dataset.values.tolist()
            test_seq =test_dataset.values.tolist()
         
        
            if encode_method == 'AAC':
                train_X = np.array(AAC(train_seq, **kw), dtype=float)
                valid_X = np.array(AAC(val_seq, **kw), dtype=float)
                test_X = np.array(AAC(test_seq, **kw), dtype=float)

            elif encode_method == 'TPC':
                train_X = np.array(TPC(train_seq, **kw), dtype=float)
                valid_X = np.array(TPC(val_seq, **kw), dtype=float)
                test_X = np.array(TPC(test_seq, **kw), dtype=float)

            elif encode_method == 'DPC':
                train_X = np.array(DPC(train_seq, **kw), dtype=float)
                valid_X = np.array(DPC(val_seq, **kw), dtype=float)
                test_X = np.array(DPC(test_seq, **kw), dtype=float)

            elif encode_method == 'EAAC':
                train_X = np.array(EAAC(train_seq,  **kw), dtype=float)
                valid_X = np.array(EAAC(val_seq, **kw), dtype=float)
                test_X = np.array(EAAC(test_seq, **kw), dtype=float)
                
            elif encode_method == 'CTriad':
                train_X = np.array(CTriad(train_seq, **kw), dtype=float)
                valid_X = np.array(CTriad(val_seq, **kw), dtype=float)
                test_X = np.array(CTriad(test_seq, **kw), dtype=float)

            elif encode_method == 'GAAC':
                train_X = np.array(GAAC(train_seq, **kw), dtype=float)
                valid_X = np.array(GAAC(val_seq, **kw), dtype=float)
                test_X = np.array(GAAC(test_seq, **kw), dtype=float)

            elif encode_method == 'GDPC':
                train_X = np.array(GDPC(train_seq, **kw), dtype=float)
                valid_X = np.array(GDPC(val_seq, **kw), dtype=float)
                test_X = np.array(GDPC(test_seq, **kw), dtype=float)

            elif encode_method == 'GTPC':
                train_X = np.array(GTPC(train_seq, **kw), dtype=float)
                valid_X = np.array(GTPC(val_seq, **kw), dtype=float)
                test_X = np.array(GTPC(test_seq, **kw), dtype=float)

            elif encode_method == 'CTDC':
                train_X = np.array(CTDC(train_seq, **kw), dtype=float)
                valid_X = np.array(CTDC(val_seq, **kw), dtype=float)
                test_X = np.array(CTDC(test_seq, **kw), dtype=float)
                
            elif encode_method == 'CTDD':
                train_X = np.array(CTDD(train_seq, **kw), dtype=float)
                valid_X = np.array(CTDD(val_seq, **kw), dtype=float)
                test_X = np.array(CTDD(test_seq, **kw), dtype=float)

            elif encode_method == 'CTDT':
                train_X = np.array(CTDT(train_seq, **kw), dtype=float)
                valid_X = np.array(CTDT(val_seq, **kw), dtype=float)
                test_X = np.array(CTDT(test_seq, **kw), dtype=float)
                
            elif encode_method == 'PAAC':
                train_X = np.array(PAAC(train_seq, **kw), dtype=float)
                valid_X = np.array(PAAC(val_seq, **kw), dtype=float)
                test_X = np.array(PAAC(test_seq, **kw), dtype=float)      

            elif encode_method == 'CKSAAP':
                train_X = np.array(CKSAAP(train_seq, **kw), dtype=float)
                valid_X = np.array(CKSAAP(val_seq, **kw), dtype=float)
                test_X = np.array(CKSAAP(test_seq, **kw), dtype=float)

            elif encode_method == 'CKSAAGP':
                train_X = np.array(CKSAAGP(train_seq, **kw), dtype=float)
                valid_X = np.array(CKSAAGP(val_seq, **kw), dtype=float)
                test_X = np.array(CKSAAGP(test_seq, **kw), dtype=float)
                
            elif encode_method == 'AAINDEX':
                train_X = np.array(AAINDEX(train_seq, **kw), dtype=float)
                valid_X = np.array(AAINDEX(val_seq, **kw), dtype=float)
                test_X = np.array(AAINDEX(test_seq, **kw), dtype=float)

            elif encode_method == 'BLOSUM62':
                train_X = np.array(BLOSUM62(train_seq, **kw), dtype=float)
                valid_X = np.array(BLOSUM62(val_seq, **kw), dtype=float)
                test_X = np.array(BLOSUM62(test_seq, **kw), dtype=float)

            elif encode_method == 'ZSCALE':
                train_X = np.array(ZSCALE(train_seq, **kw), dtype=float)
                valid_X = np.array(ZSCALE(val_seq, **kw), dtype=float)
                test_X = np.array(ZSCALE(test_seq, **kw), dtype=float) 
            else :
                pass
                print('no encode method')
                exit()
        
        train_y = train_dataset['label'].to_numpy()    
        valid_y = val_dataset['label'].to_numpy() 
        test_y = test_dataset['label'].to_numpy()
        print(train_y.shape)

        train_result = np.zeros((len(train_y), 2))
        train_result[:, 1] = train_y            
        cv_result = np.zeros((len(valid_y), 2))
        cv_result[:, 1] = valid_y
        test_result = np.zeros((len(test_y), 2))
        test_result[:,1] = test_y   #score:one of two, label
     
        if machine_method == 'RF':
            model = RandomForestClassifier(max_depth=4, random_state=0, n_estimators=100)
            clf = model.fit(train_X, train_y)
            
        elif machine_method == 'NB':
            model = GaussianNB() # 正規分布を仮定したベイズ分類
            clf = model.fit(train_X, train_y)
            
        elif machine_method == 'KN':
            model = KNeighborsClassifier()
            clf = model.fit(train_X, train_y)

        elif machine_method == 'LR':
            model = LogisticRegression(random_state=0, max_iter=10000,n_jobs=4)
            clf = model.fit(train_X, train_y)                    
            
        elif machine_method == 'SVM':    
            model = svm.SVC(probability=True)
            clf = model.fit(train_X, train_y)

        elif machine_method == 'XGB':
            xgb_train = xgb.DMatrix(train_X, train_y)
            xgb_eval  = xgb.DMatrix(valid_X , valid_y)
            params= {
                "learning_rate": 0.01,
                "max_depth": 32,
                "tree_method": "hist",
                "device": "cuda",
                "objective": "binary:logistic",
                "max_bin": 16,
                "subsample": 0.6,
                "colsample_bytree": 0.6,
            }
                    # CPU用パラメータ
            # params_cpu = {
            #     "learning_rate": 0.01,
            #     "max_depth": 2,
            #     "tree_method": "hist",
            #     "objective": "binary:logistic",
            #     "max_bin": 16,
            #     "subsample": 0.6,
            #     "colsample_bytree": 0.6,
            # }
            # clf = xgb.train(params, 
            # xgb_train, evals=[(xgb_train, "train"), (xgb_eval, "validation")], 
            # num_boost_round=100, early_stopping_rounds=20)

        # CPU用パラメータ
            params_cpu = {
                "learning_rate": 0.01,
                "max_depth": 3,
                "tree_method": "hist",
                "objective": "binary:logistic",
                "max_bin": 16,
                "subsample": 0.6,
                "colsample_bytree": 0.6,
            }

            try:
                print("GPUでトレーニング中...")
                clf = xgb.train(params, xgb_train, evals=[(xgb_train, "train"), (xgb_eval, "validation")],
                                num_boost_round=100, early_stopping_rounds=20)
                print("GPUトレーニング成功")
            except xgb.core.XGBoostError as e:
                print("GPUトレーニング失敗、CPUで再実行")
                print(f"エラー内容: {e}")
                clf = xgb.train(params_cpu, xgb_train, evals=[(xgb_train, "train"), (xgb_eval, "validation")],
                                num_boost_round=100, early_stopping_rounds=20)
                print("CPUトレーニング成功")
            
        elif machine_method == 'LGBM':
           
            lgb_train = lgb.Dataset(train_X , train_y)
            lgb_eval = lgb.Dataset(valid_X , valid_y, reference=lgb_train)
            params = {         
            'objective': 'binary',# 二値分類問題         
            'metric': 'auc',# AUC の最大化を目指す          
            'verbosity': -1,# Fatal の場合出力
            'random_state': 123,
            }
            clf = lgb.train(params, lgb_train, valid_sets=lgb_eval,
                  #verbose_eval=50,  # 50イテレーション毎に学習結果出力
                  num_boost_round=1000,  # 最大イテレーション回数指定
                  #early_stopping_rounds=100
                  callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(50)]
                 ) 
        else:
            print('No learning method')
            exit()
        
        pickle.dump(clf, open(out_path + "/" + str(i) + "/data_model/machine_model",'wb'))
        #clf= pickle.load(open(out_path + "/" + str(i) + "/data_model/machine_model",'rb'))
        
        #CV
        if machine_method == 'LGBM':
            score = clf.predict(train_X, num_iteration=clf.best_iteration)
            train_result[:, 0] = score        
            score = clf.predict(valid_X, num_iteration=clf.best_iteration)
            cv_result[:, 0] = score
        elif machine_method == 'XGB':  
            score = clf.predict(xgb_train)
            train_result[:, 0] = score
            score = clf.predict(xgb_eval)
            cv_result[:, 0] = score
        else :
            score = clf.predict_proba(train_X)
            train_result[:, 0] = score[:,1]
            score = clf.predict_proba(valid_X)
            cv_result[:, 0] = score[:,1]
        
           #independent test
        if len(test_y) != 0:
            if machine_method == 'LGBM':
                test_result[:, 0] = clf.predict(test_X, num_iteration=clf.best_iteration)
            elif machine_method == 'XGB': 
                test_result[:, 0] = clf.predict(xgb.DMatrix(test_X))
            else:
                test_result[:, 0] = clf.predict_proba(test_X)[:,1]
                
        train_output = pd.DataFrame(train_result,  columns=['prob', 'label'] )
        train_output.to_csv(out_path  + "/" + str(i) + "/train_roc.csv")  #prob, label
        cv_output = pd.DataFrame(cv_result,  columns=['prob', 'label'] )
        cv_output.to_csv(out_path  + "/" + str(i) + "/val_roc.csv")  #prob, label

        #independent test
        test_output = pd.DataFrame(test_result,  columns=['prob', 'label'] )
        test_output.to_csv(out_path  + "/" + str(i) + "/test_roc.csv")  #prob, labely
        

print('elapsed time', time.time() - start)
