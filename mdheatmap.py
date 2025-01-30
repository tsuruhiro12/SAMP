#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import argparse
7
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--species', type=str, help='term')
    parser.add_argument('--kinds', type=str, help='term')
    args = parser.parse_args()
    #species='h_b'
    species = args.species
    kinds = args.kinds
    outfile = '%sheatmap_md.png' %( kinds )
    columns_measure= ['Threshold', 'Sensitivity', 'Specificity', 'Precision', 'Accuracy', 'MCC', 'F1', 'AUC', 'AUPRC']
    
    machine_method_item = ['LGBM','RF','SVM','XGB','NB','KN','LR'] #LR
    encode_method_item = ["AAC", "BE", "BLOSUM62", "CKSAAP", "CTDC","CTDD","CTDT", "CTriad", "DPC", "EAAC", "GAAC", "GDPC", "GTPC", "PAAC", "ZSCALE","AAINDEX","prott5"]
    # ,"W2V_1_128_100_40_1", "W2V_2_128_100_40_1","W2V_3_128_100_40_1","ESM2"]

    valid_AUC_matrix = pd.DataFrame([], columns= machine_method_item)
    valid_ACC_matrix = pd.DataFrame([], columns= machine_method_item)
    valid_MCC_matrix = pd.DataFrame([], columns= machine_method_item)
       
    test_AUC_matrix = pd.DataFrame([], columns= machine_method_item)
    test_ACC_matrix = pd.DataFrame([], columns= machine_method_item)
    test_MCC_matrix = pd.DataFrame([], columns= machine_method_item)
    
    for machine_method in machine_method_item :
        valid_measure=[]
        test_measure=[]
        for i, encode_method in enumerate(encode_method_item ):

          infile_path = "./data/%sresult_%s/%s/%s" %( kinds, species, machine_method, encode_method)
          infile_name = ["val_measures.csv", "test_measures.csv" ]

          infile1 = infile_path + '/' + infile_name[0] #val
          infile2 = infile_path + '/' + infile_name[1] #test

          valid_measure.append(  (pd.read_csv(infile1, index_col=0).iloc[-1].values.tolist())) # means
          test_measure.append( (pd.read_csv(infile2, index_col=0).iloc[-1].values.tolist())) # means

        pd_valid_measure  = pd.DataFrame(data=valid_measure, index=encode_method_item, columns=columns_measure)
        pd_test_measure = pd.DataFrame(data=test_measure, index=encode_method_item, columns=columns_measure)

        print(pd_valid_measure)
        print(pd_test_measure)

        valid_AUC_matrix[machine_method] = pd_valid_measure['AUC']
        valid_ACC_matrix[machine_method] = pd_valid_measure['Accuracy']
        valid_MCC_matrix[machine_method] = pd_valid_measure['MCC']
                
        test_AUC_matrix[machine_method] = pd_test_measure['AUC']
        test_ACC_matrix[machine_method] = pd_test_measure['Accuracy']
        test_MCC_matrix[machine_method] = pd_test_measure['MCC']
        
    print(test_AUC_matrix)
    
    encode_method_item = [ 'Binary' if name == 'binary' else name for name in encode_method_item]
    encode_method_item = [ name[:5] if 'W2V' in name else name for name in encode_method_item]
    
    fig = plt.figure(figsize=(15, 12))
    fig.subplots_adjust(bottom=0.05, left=0.1, top=0.90, right=0.9, wspace=0.2, hspace=0.3)

    ax = fig.add_subplot(2,3,1)
    ax = sns.heatmap(valid_AUC_matrix, annot=True, cmap="Spectral", cbar=False)
    plt.title('AUC Training', fontsize=16 )  #fontweight="bold"
    ax.set_xticklabels(machine_method_item, fontsize=12) #, fontweight="semibold")
    ax.set_yticklabels(encode_method_item, fontsize=12)

    ax = fig.add_subplot(2,3,4)
    ax = sns.heatmap(test_AUC_matrix, annot=True, cmap="Spectral", cbar=False) #
    plt.title('AUC Testing', fontsize=16)  
    ax.set_xticklabels(machine_method_item, fontsize=12)#
    ax.set_yticklabels(encode_method_item, fontsize=12)

    encode_method_item = [ '' for name in encode_method_item]
        
    ax = fig.add_subplot(2,3,2)
    ax = sns.heatmap(valid_MCC_matrix, annot=True, cmap="Spectral", cbar=False)
    plt.title('MCC Training', fontsize=16)  
    ax.set_xticklabels(machine_method_item, fontsize=12)
    ax.set_yticklabels(encode_method_item, fontsize=12)
      
    ax = fig.add_subplot(2,3,5)
    ax = sns.heatmap(test_MCC_matrix, annot=True, cmap="Spectral", cbar=False)
    plt.title('MCC Testing', fontsize=16 )  #fontweight="bold"
    ax.set_xticklabels(machine_method_item, fontsize=12) #, fontweight="semibold")
    ax.set_yticklabels(encode_method_item, fontsize=12)

    ax = fig.add_subplot(2,3,3)
    ax = sns.heatmap(valid_ACC_matrix, annot=True, cmap="Spectral", cbar=False) #
    plt.title('ACC Training', fontsize=16)  
    ax.set_xticklabels(machine_method_item, fontsize=12)#
    ax.set_yticklabels(encode_method_item, fontsize=12)
    
    ax = fig.add_subplot(2,3,6)
    ax = sns.heatmap(test_ACC_matrix, annot=True, cmap="Spectral", cbar=False)
    plt.title('ACC Testing', fontsize=16)  
    ax.set_xticklabels(machine_method_item, fontsize=12)
    ax.set_yticklabels(encode_method_item, fontsize=12)
         
    plt.savefig(outfile, dpi=300)
    #plt.show()
