#!/usr/bin/env python
'''
Transethnic local genetic correlation estimation

HYPERGNOVA

Created on 2020-5-20

@author: Yiliang
'''

import argparse, os.path, sys
from subprocess import call
import multiprocessing
import pandas as pd
import numpy as np
from prep import prep
from ldsc_thin import ldscore
from heritability import heritability
from pheno import pheno
from calculate import calculate


try:
    x = pd.DataFrame({'A': [1, 2, 3]})
    x.drop_duplicates(subset='A')
except TypeError:
    raise ImportError('HYPERGNOVA requires pandas version > 0.15.2')


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('precision', 4)
pd.set_option('max_colwidth',1000)
np.set_printoptions(linewidth=1000)
np.set_printoptions(precision=4)


# returns whether the parent directory of path exists
def parent_dir_exists(path):
    return os.path.exists(os.path.abspath(os.path.join(path, os.pardir)))

def pipeline(args):
    pd.options.mode.chained_assignment = None

    # Sanity check args
    if not parent_dir_exists(args.out):
        raise ValueError('--out flag points to an invalid path.')

    print('Preparing files for analysis...')
    gwas_snps, reversed_alleles_ref, bed, N1, N2 = prep(args.bfile1, args.bfile2, args.partition, args.sumstats1, args.sumstats2, args.N1, args.N2)
    print('Calculating LD scores for the first population...')
    ld_scores1 = ldscore(args.bfile1, gwas_snps)
    print('Calculating LD scores for the second population...')
    ld_scores2 = ldscore(args.bfile2, gwas_snps)
    ld_scores1 = ld_scores1[ld_scores1['SNP'].isin(ld_scores2['SNP'])]
    ld_scores2 = ld_scores2[ld_scores2['SNP'].isin(ld_scores1['SNP'])]
    gwas_snps = gwas_snps[gwas_snps['SNP'].isin(ld_scores1['SNP'])]
    print('{} SNPs included in our analysis...'.format(len(gwas_snps)))
    print('Calculating heritability...')
    h_1, h_2 = heritability(gwas_snps, ld_scores1, ld_scores2, N1, N2)
    print('The genome-wide heritability of the first trait is {}.\nThe genome-wide heritability of the second trait is {}.'.format(h_1, h_2))
    print('Calculating local genetic covariance...')
    out = calculate(args.bfile, bed, args.thread, gwas_snps, reversed_alleles_ref, N1, N2, args.genome_wide)
    out.to_csv(args.out, sep=' ', na_rep='NA', index=False)


parser = argparse.ArgumentParser()

parser.add_argument('sumstats1',
    help='The first sumstats file.')
parser.add_argument('sumstats2',
    help='The second sumstats file.')

parser.add_argument('--bfile1', required=True, type=str,
    help='Prefix for Plink .bed/.bim/.fam file of the sumstats1 file.')
parser.add_argument('--bfile2', required=True, type=str,
    help='Prefix for Plink .bed/.bim/.fam file of the sumstats2 file.')
parser.add_argument('--partition', required=True, type=str,
    help='Genome partition file in bed format')
parser.add_argument('--N1', type=int,
    help='N of the sumstats1 file. If not provided, this value will be inferred '
    'from the sumstats1 arg.')
parser.add_argument('--N2', type=int,
    help='N of the sumstats2 file. If not provided, this value will be inferred '
    'from the sumstats2 arg.')

parser.add_argument('--out', required=True, type=str,
    help='Location to output results.')
parser.add_argument('--thread', default= multiprocessing.cpu_count(), type=int,
    help='Thread numbers used for calculation. Default = CPU numbers.')
parser.add_argument('--genome_wide', default= False, action='store_true',
    help='Whether to estimate global genetic covariance. Default = F')

if __name__ == '__main__':
    pipeline(parser.parse_args())

