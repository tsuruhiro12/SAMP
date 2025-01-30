#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import argparse

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--species', type=str, help='term')
    parser.add_argument('--kinds', type=str, help='term')
    args = parser.parse_args()
    #species='h_b'
    species = args.species
    kinds = args.kinds
    outfile = '%sheatmap_md_F1.png' %( kinds )
    columns_measure= ['Threshold', 'Sensitivity', 'Specificity', 'Precision', 'accuracy', 'MCC', 'F1', 'AUC', 'AUPRC']

    machine_method_item = ['LGBM','RF','XGB','SVM','NB','KN','LR'] #LR
    encode_method_item = ["AAC", "BE", "BLOSUM62", "CKSAAP", "CTDC","CTDD","CTDT", "CTriad", "DPC", "EAAC", "GAAC", "GDPC", "GTPC", "PAAC", "ZSCALE","AAINDEX","W2V_1_128_100_40_1", "W2V_2_128_100_40_1","W2V_3_128_100_40_1"]

    valid_F1_matrix = pd.DataFrame([], columns= machine_method_item)
    valid_Precision_matrix = pd.DataFrame([], columns= machine_method_item)
       
    test_F1_matrix = pd.DataFrame([], columns= machine_method_item)
    test_Precision_matrix = pd.DataFrame([], columns= machine_method_item)
    
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

        valid_F1_matrix[machine_method] = pd_valid_measure['F1']
        valid_Precision_matrix[machine_method] = pd_valid_measure['Precision']
                
        test_F1_matrix[machine_method] = pd_test_measure['F1']
        test_Precision_matrix[machine_method] = pd_test_measure['Precision']

        
    print(test_F1_matrix)
    
    encode_method_item = [ 'Binary' if name == 'binary' else name for name in encode_method_item]
    encode_method_item = [ name[:5] if 'W2V' in name else name for name in encode_method_item]
    
    fig = plt.figure(figsize=(15, 12))
    fig.subplots_adjust(bottom=0.05, left=0.1, top=0.90, right=0.9, wspace=0.2, hspace=0.3)

    ax = fig.add_subplot(2,2,1)
    ax = sns.heatmap(valid_F1_matrix, annot=True, cmap="Spectral", cbar=False)
    plt.title('F1 Training', fontsize=16 )  #fontweight="bold"
    ax.set_xticklabels(machine_method_item, fontsize=12) #, fontweight="semibold")
    ax.set_yticklabels(encode_method_item, fontsize=12)

    ax = fig.add_subplot(2,2,3)
    ax = sns.heatmap(test_F1_matrix, annot=True, cmap="Spectral", cbar=False) #
    plt.title('F1 Testing', fontsize=16)  
    ax.set_xticklabels(machine_method_item, fontsize=12)#
    ax.set_yticklabels(encode_method_item, fontsize=12)

    encode_method_item = [ '' for name in encode_method_item]
        
    ax = fig.add_subplot(2,2,2)
    ax = sns.heatmap(valid_Precision_matrix, annot=True, cmap="Spectral", cbar=False) #
    plt.title('Precision Training', fontsize=16)  
    ax.set_xticklabels(machine_method_item, fontsize=12)#
    ax.set_yticklabels(encode_method_item, fontsize=12)
    
    ax = fig.add_subplot(2,2,4)
    ax = sns.heatmap(test_Precision_matrix, annot=True, cmap="Spectral", cbar=False)
    plt.title('Precision Testing', fontsize=16)  
    ax.set_xticklabels(machine_method_item, fontsize=12)
    ax.set_yticklabels(encode_method_item, fontsize=12)
         
    plt.savefig(outfile, dpi=300)
    #plt.show()
