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


def calLocalCov(i, partition, geno_array, coords, bps, gwas_snps, ld_scores, n1, n2, pheno_corr, pheno_corr_var):
    m = len(gwas_snps)
    CHR = partition.iloc[i, 0]
    START = partition.iloc[i, 1]
    END = partition.iloc[i, 2]

    idx = np.logical_and(np.logical_and(gwas_snps['CHR']==CHR, bps <= END), bps >= START)
    m0 = np.sum(idx)
    if m0 < 120:
        df = pd.DataFrame(OrderedDict({"chr":[], "start":[], "end":[], "rho":[], "corr":[], "h2_1":[], "h2_2":[], "var":[], "p":[], "m":[]}))
        return df
    
    tmp_coords = coords[idx]

    block_gwas_snps = gwas_snps[idx]
    block_ld_scores = ld_scores[idx]
    max_dist = 0.03
    block_left = ld.getBlockLefts(tmp_coords, max_dist)

    lN, blockLD = geno_array.ldCorrVarBlocks(block_left, idx)
    lN = block_ld_scores["L2"]
    meanLD = np.mean(lN)
    local_LD = nearest_Corr(blockLD)

    d, v = linalg.eigh(local_LD)
    order = d.argsort()[::-1]
    d = d[order]
    v = v[:,order]
    if np.sum(d>0) < 120:
        df = pd.DataFrame(OrderedDict({"chr":[], "start":[], "end":[], "rho":[], "corr":[], "h2_1":[], "h2_2":[], "var":[], "p":[], "m":[]}))
        return df
    
    sub_d = d[d>0]
    sub_v = v[:,d>0]

    tz1 = np.dot(sub_v.T, block_gwas_snps['Z_x'])
    tz2 = np.dot(sub_v.T, block_gwas_snps['Z_y'])
    y = tz1 * tz2 - pheno_corr * sub_d

    Localh1 = (np.mean(block_gwas_snps['Z_x'] ** 2) - 1) / meanLD * m0 / n1
    Localh2 = (np.mean(block_gwas_snps['Z_y'] ** 2) - 1) / meanLD * m0 / n2

    Z_x = gwas_snps['Z_x']
    Z_y = gwas_snps['Z_y']

    h1 = (np.mean(Z_x ** 2) - 1) / np.mean(ld_scores['L2']) * m / n1
    h2 = (np.mean(Z_y ** 2) - 1) / np.mean(ld_scores['L2']) * m / n2

    wh1 = h1 * m0 / m
    wh2 = h2 * m0 / m
    #wh12 = np.max([Localh1, 0])
    #wh22 = np.max([Localh2, 0])
    #wh1 = (wh11 + wh12) / 2
    #wh2 = (wh21 + wh22) / 2
    Localrho = (np.sum(block_gwas_snps['Z_x'] * block_gwas_snps['Z_y']) - pheno_corr * m0) / meanLD / sqrt(n1 * n2)

    threshold = 1
    cur_d = sub_d[sub_d>threshold]
    cur_y = y[sub_d>threshold]
    cur_dsq = cur_d ** 2
    denominator = (wh1 * cur_d / m0 + 1 / n1) * (wh2 * cur_d / m0 + 1 / n2)
    cur_v1 = np.sum(cur_dsq / denominator)
    cur_v2 = np.sum(cur_y / sqrt(n1 * n2) / denominator)
    cur_v3 = np.sum(cur_y ** 2 / (n1 * n2) / (denominator * cur_dsq))

    emp_var = [(cur_v3 - (cur_v2 ** 2) / cur_v1) / (cur_v1 * (len(cur_d) - 1))]
    theo_var = [1 / cur_v1]

    for K in range(len(cur_d), len(sub_d)):
        eig = sub_d[K]
        tmp_y = y[K]
        cur_v1 += eig ** 2 / ((wh1 * eig / m0 + 1 / n1) * (wh2 * eig / m0 + 1 / n2))
        cur_v2 += tmp_y / sqrt(n1 * n2) / ((wh1 * eig / m0 + 1 / n1) * (wh2 * eig / m0 + 1 / n2))
        cur_v3 += tmp_y ** 2 / (n1 * n2) / ((wh1 * eig ** 2 / m0 + eig / n1) * (wh2 * eig ** 2 / m0 + eig / n2))
        emp_var.append((cur_v3 - (cur_v2 ** 2) / cur_v1) / (cur_v1 * K))
        theo_var.append(1 / cur_v1)
    
    max_emp_theo = np.maximum(emp_var, theo_var)
    min_idx = np.argmin(max_emp_theo)

    y = y[:(len(cur_d)+min_idx-1)]
    sub_d = sub_d[:(len(cur_d)+min_idx-1)]
    sub_dsq = sub_d ** 2

    var_rho = m0 ** 2 * min(max_emp_theo)
    q = (wh1 * sub_d / m0 + 1 / n1) * (wh2 * sub_d / m0 + 1 / n2)
    v4 = np.sum(sub_d/q)/np.sum(sub_dsq/q)
    var_phencorr = pheno_corr_var / (n1 * n2) * m0 ** 2 * v4 ** 2
    var_rho += var_phencorr

    se_rho = sqrt(var_rho)
    p_value = norm.sf(abs(Localrho / se_rho)) * 2

    if Localh1 < 0 or Localh2 < 0:
        corr = np.nan
    else:
        corr = Localrho / sqrt(Localh1 * Localh2)

    df = pd.DataFrame(OrderedDict({"chr":[CHR], "start":[START], "end":[END], "rho":[Localrho], "corr":[corr], "h2_1":[Localh1], "h2_2":[Localh2], "var":[var_rho], "p":[p_value], "m":[m0]}))

    return df

def calGlobalCov(i, tmp_partition, geno_array1, geno_array2, coords, bps, tmp_gwas_snps, tmp_flip, n1, n2):
    m = len(tmp_gwas_snps)
    CHR = tmp_partition.iloc[i, 0]
    START = tmp_partition.iloc[i, 1]
    END = tmp_partition.iloc[i, 2]

    idx = np.logical_and(np.logical_and(tmp_gwas_snps['CHR']==CHR, bps <= END), bps >= START)
    m0 = np.sum(idx)
    if m0 < 120:
        df = pd.DataFrame(OrderedDict({"chr":[], "start":[], "end":[], "rho":[], "corr":[], "h2_1":[], "h2_2":[], "var":[], "p":[], "m":[]}))
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
    order = d1.argsort()[::-1]
    d = d[order]
    v = v[:,order]
    if np.sum(d>0) < 120:
        df = pd.DataFrame(OrderedDict({"chr":[], "start":[], "end":[], "rho":[], "corr":[], "h2_1":[], "h2_2":[], "var":[], "p":[], "m":[]}))
        return df
    
    sub_d = d[d>0]
    sub_v = v[:,d>0]

    tz1 = np.dot(sub_v.T, block_gwas_snps['Z_x'])
    tz2 = np.dot(sub_v.T, block_gwas_snps['Z_y'])
    y = tz1 * tz2 - pheno_corr * sub_d

    Localh1 = (np.mean(block_gwas_snps['Z_x'] ** 2) - 1) / meanLD * m0 / n1
    Localh2 = (np.mean(block_gwas_snps['Z_y'] ** 2) - 1) / meanLD * m0 / n2

    Z_x = gwas_snps['Z_x']
    Z_y = gwas_snps['Z_y']

    h1 = (np.mean(Z_x ** 2) - 1) / np.mean(ld_scores['L2']) * m / n1
    h2 = (np.mean(Z_y ** 2) - 1) / np.mean(ld_scores['L2']) * m / n2

    wh1 = h1 * m0 / m
    wh2 = h2 * m0 / m
    #wh12 = np.max([Localh1, 0])
    #wh22 = np.max([Localh2, 0])
    #wh1 = (wh11 + wh12) / 2
    #wh2 = (wh21 + wh22) / 2
    Localrho = (np.sum(block_gwas_snps['Z_x'] * block_gwas_snps['Z_y']) - pheno_corr * m0) / meanLD / sqrt(n1 * n2)

    threshold = 1
    cur_d = sub_d[sub_d>threshold]
    cur_y = y[sub_d>threshold]
    cur_dsq = cur_d ** 2
    denominator = (wh1 * cur_d / m0 + 1 / n1) * (wh2 * cur_d / m0 + 1 / n2)
    cur_v1 = np.sum(cur_dsq / denominator)
    cur_v2 = np.sum(cur_y / sqrt(n1 * n2) / denominator)
    cur_v3 = np.sum(cur_y ** 2 / (n1 * n2) / (denominator * cur_dsq))

    emp_var = [(cur_v3 - (cur_v2 ** 2) / cur_v1) / (cur_v1 * (len(cur_d) - 1))]
    theo_var = [1 / cur_v1]

    for K in range(len(cur_d), len(sub_d)):
        eig = sub_d[K]
        tmp_y = y[K]
        cur_v1 += eig ** 2 / ((wh1 * eig / m0 + 1 / n1) * (wh2 * eig / m0 + 1 / n2))
        cur_v2 += tmp_y / sqrt(n1 * n2) / ((wh1 * eig / m0 + 1 / n1) * (wh2 * eig / m0 + 1 / n2))
        cur_v3 += tmp_y ** 2 / (n1 * n2) / ((wh1 * eig ** 2 / m0 + eig / n1) * (wh2 * eig ** 2 / m0 + eig / n2))
        emp_var.append((cur_v3 - (cur_v2 ** 2) / cur_v1) / (cur_v1 * K))
        theo_var.append(1 / cur_v1)
    
    max_emp_theo = np.maximum(emp_var, theo_var)
    min_idx = np.argmin(max_emp_theo)

    y = y[:(len(cur_d)+min_idx-1)]
    sub_d = sub_d[:(len(cur_d)+min_idx-1)]
    sub_dsq = sub_d ** 2

    var_rho = m0 ** 2 * min(max_emp_theo)
    q = (wh1 * sub_d / m0 + 1 / n1) * (wh2 * sub_d / m0 + 1 / n2)
    v4 = np.sum(sub_d/q)/np.sum(sub_dsq/q)
    var_phencorr = pheno_corr_var / (n1 * n2) * m0 ** 2 * v4 ** 2
    var_rho += var_phencorr

    se_rho = sqrt(var_rho)
    p_value = norm.sf(abs(Localrho / se_rho)) * 2

    if Localh1 < 0 or Localh2 < 0:
        corr = np.nan
    else:
        corr = Localrho / sqrt(Localh1 * Localh2)

    df = pd.DataFrame(OrderedDict({"chr":[CHR], "start":[START], "end":[END], "rho":[Localrho], "corr":[corr], "h2_1":[Localh1], "h2_2":[Localh2], "var":[var_rho], "p":[p_value], "m":[m0]}))

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
        
        return df
    
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
