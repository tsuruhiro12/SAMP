#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('-i1', '--infile1', help='file')
parser.add_argument('-o1', '--outfile1', help='file')
parser.add_argument('-o2', '--outfile2', help='file')

test_txt1 = parser.parse_args().infile1
test_fasta1 = parser.parse_args().outfile2
test_csv1 = parser.parse_args().outfile1


test1 = pd.read_csv(test_txt1 , header=None)
test1 = test1.rename(columns={0:'seq',1:'label'})
print(test1)

with open(test_fasta1, 'w') as fout:
   for i in range(test1.shape[0]):
      if test1.iloc[i,1] == 1:
         fout.write('>pep_%s|1|label\n'%i)
         fout.write(test1.iloc[i,0])
         fout.write('\n')
      else:
         fout.write('>pep_%s|0|label\n'%i)
         fout.write(test1.iloc[i,0])
         fout.write('\n')
test1.to_csv(test_csv1, index=None)
