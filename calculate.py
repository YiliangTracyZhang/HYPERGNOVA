#!/usr/bin/python
from __future__ import division, print_function
import multiprocessing
from subprocess import call
import numpy as np
import pandas as pd
import numpy.linalg as linalg
from math import sqrt
import ld.ldscore as ld
import ld.parse as ps
from ldsc_thin import __filter_bim__
from scipy.stats import norm
from collections import OrderedDict


def nearest_Corr(input_mat):
    d, v = linalg.eigh(input_mat)
    A = (v * np.maximum(d, 0)).dot(v.T)
    A = (A + A.T) / 2
    multiplier = 1 / np.sqrt(np.diag(A))
    A = A * multiplier
    A = (A.T * multiplier).T
    return A


def calLocalCov(i, tmp_partition, geno_array1, geno_array2, coords, bps, tmp_gwas_snps, tmp_flip, n1, n2):
    m = len(tmp_gwas_snps)
    CHR = tmp_partition.iloc[i, 0]
    START = tmp_partition.iloc[i, 1]
    END = tmp_partition.iloc[i, 2]

    idx = np.logical_and(np.logical_and(tmp_gwas_snps['CHR']==CHR, bps <= END), bps >= START)
    m0 = np.sum(idx)
    if m0 < 120:
        df = pd.DataFrame(OrderedDict({"chr":[], "start":[], "end":[], "rho":[], "m":[]}))
        return df
    
    tmp_coords = coords[idx]

    block_gwas_snps = tmp_gwas_snps[idx]
    block_flip = tmp_flip[idx]
    max_dist = 1
    block_left = ld.getBlockLefts(tmp_coords, max_dist)

    blockLD1 = geno_array1.ldCorrVarBlocks(block_left, idx)
    local_LD1 = nearest_Corr(blockLD1)
    blockLD2 = geno_array2.ldCorrVarBlocks(block_left, idx)
    flip_multiply = block_flip * 2 - 1
    blockLD2 = blockLD2 * flip_multiply
    blockLD2 = blockLD2.T * flip_multiply
    local_LD2 = nearest_Corr(blockLD2)

    d1, v1 = linalg.eigh(local_LD1)
    d2, v2 = linalg.eigh(local_LD2)
    order1 = d1.argsort()[::-1]
    order2 = d2.argsort()[::-1]

    d1 = d1[order1]
    v1 = v1[:,order1]
    d2 = d2[order2]
    v2 = v2[:,order2]
    if np.sum(np.logical_and(d1>0, d2>0)) < 120:
        df = pd.DataFrame(OrderedDict({"chr":[], "start":[], "end":[], "rho":[], "m":[]}))
        return df
    

    sub_d1 = d1[np.logical_and(d1>0, d2>0)]
    sub_v1 = v1[:,np.logical_and(d1>0, d2>0)]

    sub_d2 = d2[np.logical_and(d1>0, d2>0)]
    sub_v2 = v2[:,np.logical_and(d1>0, d2>0)]

    tz1 = np.dot(sub_v1.T, block_gwas_snps['Z_x'])
    tz2 = np.dot(sub_v2.T, block_gwas_snps['Z_y'])
    y = tz1 * tz2
    sub_v = sub_v1 * sub_v2
    u = sub_v.sum(axis=0)
    numerator = y.T.dot(u)
    w = sub_d1 * sub_d2
    denominator = w.T.dot(np.square(u))
    rho = m0 / sqrt(n1 * n2) * numerator / denominator

    df = pd.DataFrame(OrderedDict({"chr":[CHR], "start":[START], "end":[END], "rho":[rho], "m":[m0]}))

    return df

def calGlobalCov(i, tmp_partition, geno_array1, geno_array2, coords, bps, tmp_gwas_snps, tmp_flip, n1, n2):
    m = len(tmp_gwas_snps)
    CHR = tmp_partition.iloc[i, 0]
    START = tmp_partition.iloc[i, 1]
    END = tmp_partition.iloc[i, 2]

    idx = np.logical_and(np.logical_and(tmp_gwas_snps['CHR']==CHR, bps <= END), bps >= START)
    m0 = np.sum(idx)
    if m0 < 120:
        df = pd.DataFrame(OrderedDict({"numerator":[], "denominator":[], "m":[]}))
        return df
    
    tmp_coords = coords[idx]

    block_gwas_snps = tmp_gwas_snps[idx]
    block_flip = tmp_flip[idx]
    max_dist = 1
    block_left = ld.getBlockLefts(tmp_coords, max_dist)

    blockLD1 = geno_array1.ldCorrVarBlocks(block_left, idx)
    local_LD1 = nearest_Corr(blockLD1)
    blockLD2 = geno_array2.ldCorrVarBlocks(block_left, idx)
    flip_multiply = block_flip * 2 - 1
    blockLD2 = blockLD2 * flip_multiply
    blockLD2 = blockLD2.T * flip_multiply
    local_LD2 = nearest_Corr(blockLD2)

    d1, v1 = linalg.eigh(local_LD1)
    d2, v2 = linalg.eigh(local_LD2)
    order1 = d1.argsort()[::-1]
    order2 = d2.argsort()[::-1]

    d1 = d1[order1]
    v1 = v1[:,order1]
    d2 = d2[order2]
    v2 = v2[:,order2]
    if np.sum(np.logical_and(d1>0, d2>0)) < 120:
        df = pd.DataFrame(OrderedDict({"numerator":[], "denominator":[], "m":[]}))
        return df
    

    sub_d1 = d1[np.logical_and(d1>0, d2>0)]
    sub_v1 = v1[:,np.logical_and(d1>0, d2>0)]

    sub_d2 = d2[np.logical_and(d1>0, d2>0)]
    sub_v2 = v2[:,np.logical_and(d1>0, d2>0)]

    tz1 = np.dot(sub_v1.T, block_gwas_snps['Z_x'])
    tz2 = np.dot(sub_v2.T, block_gwas_snps['Z_y'])
    y = tz1 * tz2
    sub_v = sub_v1 * sub_v2
    u = sub_v.sum(axis=0)
    numerator = y.T.dot(u)
    w = sub_d1 * sub_d2
    denominator = w.T.dot(np.square(u))

    df = pd.DataFrame(OrderedDict({"numerator":[numerator], "denominator":[denominator], "m":[m0]}))

    return df

def _supergnova(bfile1, bfile2, partition, thread, gwas_snps, reversed_alleles_ref, n1, n2):
    m = len(gwas_snps)

    snp_file1, snp_file2, snp_obj = bfile1+'.bim', bfile2+'.bim', ps.PlinkBIMFile
    ind_file1, ind_file2, ind_obj = bfile1+'.fam', bfile2+'.fam', ps.PlinkFAMFile
    array_file1, array_file2, array_obj = bfile1+'.bed', bfile2+'.bed', ld.PlinkBEDFile

    # read bim/snp
    array_snps1, array_snps2 = snp_obj(snp_file1), snp_obj(snp_file2)
    chr_bfile = list(set(array_snps1.df['CHR']).intersection(set(array_snps2.df['CHR'])))
    tmp_partition = partition[partition.iloc[:,0].isin(chr_bfile)]
    tmp_gwas_snps = gwas_snps[gwas_snps.iloc[:,0].isin(chr_bfile)].reset_index(drop=True)
    tmp_flip = reversed_alleles_ref[gwas_snps.iloc[:,0].isin(chr_bfile)]
    blockN = len(tmp_partition)
    # snp list

    keep_snps1 = __filter_bim__(tmp_gwas_snps, array_snps1)
    keep_snps2 = __filter_bim__(tmp_gwas_snps, array_snps2)

    array_indivs1, array_indivs2 = ind_obj(ind_file1), ind_obj(ind_file2)
    n_ref1 = len(array_indivs1.IDList)
    n_ref2 = len(array_indivs2.IDList)
    keep_indivs = None

    ## reading genotype

    geno_array1 = array_obj(array_file1, n_ref1, array_snps1, keep_snps=keep_snps1,
        keep_indivs=keep_indivs, mafMin=None)
    geno_array2 = array_obj(array_file2, n_ref2, array_snps2, keep_snps=keep_snps2,
        keep_indivs=keep_indivs, mafMin=None)
    coords = np.array(array_snps1.df['CM'])[geno_array1.kept_snps]
    bps = np.array(array_snps1.df['BP'])[geno_array1.kept_snps]

    ## Calculating local genetic covariance
    
    results = []
    def collect_results(result):
        results.append(result)
    pool = multiprocessing.Pool(processes = thread)
    for i in range(blockN):
        pool.apply_async(calGlobalCov, args=(i, tmp_partition, geno_array1, geno_array2, coords, 
            bps, tmp_gwas_snps, tmp_flip, n1, n2),
            callback=collect_results)
    pool.close()
    pool.join()
    df = pd.concat(results, ignore_index=True)
    #df = pd.DataFrame(results)
    #df.columns = ["chr", "start", "end", "rho", "corr", "h1", "h2", "var", "p", "m"]
    convert_dict = {"chr": int, "start": int, "end":int, "m":int}
    df = df.astype(convert_dict)
    return df

def _supergnova_global(bfile1, bfile2, partition, thread, gwas_snps, reversed_alleles_ref, n1, n2):

    snp_file1, snp_file2, snp_obj = bfile1+'.bim', bfile2+'.bim', ps.PlinkBIMFile
    ind_file1, ind_file2, ind_obj = bfile1+'.fam', bfile2+'.fam', ps.PlinkFAMFile
    array_file1, array_file2, array_obj = bfile1+'.bed', bfile2+'.bed', ld.PlinkBEDFile

    # read bim/snp
    array_snps1, array_snps2 = snp_obj(snp_file1), snp_obj(snp_file2)
    chr_bfile = list(set(array_snps1.df['CHR']).intersection(set(array_snps2.df['CHR'])))
    tmp_partition = partition[partition.iloc[:,0].isin(chr_bfile)]
    tmp_gwas_snps = gwas_snps[gwas_snps.iloc[:,0].isin(chr_bfile)].reset_index(drop=True)
    tmp_flip = reversed_alleles_ref[gwas_snps.iloc[:,0].isin(chr_bfile)]
    blockN = len(tmp_partition)
    # snp list

    keep_snps1 = __filter_bim__(tmp_gwas_snps, array_snps1)
    keep_snps2 = __filter_bim__(tmp_gwas_snps, array_snps2)

    array_indivs1, array_indivs2 = ind_obj(ind_file1), ind_obj(ind_file2)
    n_ref1 = len(array_indivs1.IDList)
    n_ref2 = len(array_indivs2.IDList)
    keep_indivs = None

    ## reading genotype

    geno_array1 = array_obj(array_file1, n_ref1, array_snps1, keep_snps=keep_snps1,
        keep_indivs=keep_indivs, mafMin=None)
    geno_array2 = array_obj(array_file2, n_ref2, array_snps2, keep_snps=keep_snps2,
        keep_indivs=keep_indivs, mafMin=None)
    coords = np.array(array_snps1.df['CM'])[geno_array1.kept_snps]
    bps = np.array(array_snps1.df['BP'])[geno_array1.kept_snps]

    ## Calculating local genetic covariance
    
    results = []
    def collect_results(result):
        results.append(result)
    pool = multiprocessing.Pool(processes = thread)
    for i in range(blockN):
        pool.apply_async(calGlobalCov, args=(i, tmp_partition, geno_array1, geno_array2, coords, 
            bps, tmp_gwas_snps, tmp_flip, n1, n2),
            callback=collect_results)
    pool.close()
    pool.join()
    df = pd.concat(results, ignore_index=True)
    #df = pd.DataFrame(results)
    #df.columns = ["chr", "start", "end", "rho", "corr", "h1", "h2", "var", "p", "m"]
    #convert_dict = {"chr": int, "start": int, "end":int, "m":int}
    #df = df.astype(convert_dict)
    return df

def calculate(bfile1, bfile2, partition, thread, gwas_snps, reversed_alleles_ref, n1, n2, genome_wide):
    if thread is None:
        thread = multiprocessing.cpu_count()
        print('{C} CPUs are detected. Using {C} threads in computation  ... '.format(C=str(thread)))
    else:
        cpuNum = multiprocessing.cpu_count()
        thread = min(thread, cpuNum)
        print('{C} CPUs are detected. Using {N} threads in computation  ... '.format(C=str(cpuNum), N=str(thread)))

    df = None
    if genome_wide:
        if '@' in bfile1:
            all_dfs = []
            chrs = list(set(partition.iloc[:,0]))
            if '@' in bfile2:
                for i in range(len(chrs)):
                    cur_bfile1 = bfile1.replace('@', str(chrs[i]))
                    cur_bfile2 = bfile2.replace('@', str(chrs[i]))
                    all_dfs.append(_supergnova_global(cur_bfile1, cur_bfile2, partition, thread, gwas_snps, reversed_alleles_ref, n1, n2))
                    print('Done with SNPs in chromosome {}'.format(chrs[i]))
            else:
                for i in range(len(chrs)):
                    cur_bfile1 = bfile1.replace('@', str(chrs[i]))
                    all_dfs.append(_supergnova_global(cur_bfile1, bfile2, partition, thread, gwas_snps, reversed_alleles_ref, n1, n2))
                    print('Done with SNPs in chromosome {}'.format(chrs[i]))
            df = pd.concat(all_dfs, ignore_index=True)
        else:
            if '@' in bfile2:
                all_dfs = []
                chrs = list(set(partition.iloc[:,0]))
                for i in range(len(chrs)):
                    cur_bfile2 = bfile2.replace('@', str(chrs[i]))
                    all_dfs.append(_supergnova_global(bfile1, cur_bfile2, partition, thread, gwas_snps, reversed_alleles_ref, n1, n2))
                    print('Done with SNPs in chromosome {}'.format(chrs[i]))
                df = pd.concat(all_dfs, ignore_index=True)
            else:
                df = _supergnova_global(bfile1, bfile2, partition, thread, gwas_snps, reversed_alleles_ref, n1, n2)
        total_m = np.sum(df['m'])
        global_Cov = total_m / sqrt(n1 * n2) * np.sum(df['numerator']) / np.sum(df['denominator'])
        results = pd.DataFrame(OrderedDict({"rho":[global_Cov], "m":[total_m]}))
        convert_dict = {"m":int}
        results = results.astype(convert_dict)
        return results
    
    else:
        if '@' in bfile1:
            all_dfs = []
            chrs = list(set(partition.iloc[:,0]))
            if '@' in bfile2:
                for i in range(len(chrs)):
                    cur_bfile1 = bfile1.replace('@', str(chrs[i]))
                    cur_bfile2 = bfile2.replace('@', str(chrs[i]))
                    all_dfs.append(_supergnova(cur_bfile1, cur_bfile2, partition, thread, gwas_snps, reversed_alleles_ref, n1, n2))
                    print('Computed local genetic covariance for chromosome {}'.format(chrs[i]))
            else:
                for i in range(len(chrs)):
                    cur_bfile1 = bfile1.replace('@', str(chrs[i]))
                    all_dfs.append(_supergnova(cur_bfile1, bfile2, partition, thread, gwas_snps, reversed_alleles_ref, n1, n2))
                    print('Computed local genetic covariance for chromosome {}'.format(chrs[i]))
            df = pd.concat(all_dfs, ignore_index=True)
        else:
            if '@' in bfile2:
                all_dfs = []
                chrs = list(set(partition.iloc[:,0]))
                for i in range(len(chrs)):
                    cur_bfile2 = bfile2.replace('@', str(chrs[i]))
                    all_dfs.append(_supergnova(bfile1, cur_bfile2, partition, thread, gwas_snps, reversed_alleles_ref, n1, n2))
                    print('Computed local genetic covariance for chromosome {}'.format(chrs[i]))
                df = pd.concat(all_dfs, ignore_index=True)
            else:
                df = _supergnova(bfile1, bfile2, partition, thread, gwas_snps, reversed_alleles_ref, n1, n2)
    
        return df
