import polars as pl
import numpy as np
from pathlib import Path

from scipy.stats import binom
from scipy.special import logsumexp
def get_snv_cluster_assignments_binomial(cluster_labels,variant_data):

    mult_one_vaf = variant_data['purity']/(variant_data['purity']*variant_data['total_cn'] +(1-variant_data['purity'])*variant_data['normal_total_cn'])
    
    variant_coverage = variant_data['tumor_alt_count'] + variant_data['tumor_ref_count']

    peak_probabilities = {}
    for cluster_row in cluster_labels.iter_rows(named=True):
        max_mult = 1 if cluster_row['cluster_ccf']<0.9 else min(variant_data['major_cn'].max(),10) 

        
        for mult in range(1,max_mult+1):
            peak_name = f'{cluster_row["cluster_label"]}-{cluster_row["cluster"]}-{mult}'
            peak_frac = cluster_row['cluster_ccf']*mult*mult_one_vaf
            
            mult_prob = binom.logpmf(variant_data['tumor_alt_count'],variant_coverage,peak_frac)
            mult_prob += np.log(cluster_row['cluster_fraction'])
            mult_prob = np.where(variant_data['major_cn']<mult,-np.inf,mult_prob)
            peak_probabilities[peak_name] = mult_prob
    norm_data = np.array([v for v in peak_probabilities.values()])
    norm_data = logsumexp(norm_data,axis=0)
    peak_probabilities = {c:np.exp(v-norm_data) for c,v in peak_probabilities.items()}
    cluster_probabilities = np.zeros((len(cluster_labels),len(variant_data)))
    cluster_names = cluster_labels['cluster_label'].to_list()
    
    best_multiplicity = np.ones(variant_data.height).astype(int)
    best_probs = np.zeros(variant_data.height)
    
    for peak_name,probs in peak_probabilities.items():
        cluster = peak_name.split('-')[0]
        mult = int(peak_name.split('-')[-1])

        cluster_probabilities[cluster_names.index(cluster)]+=probs

        best_multiplicity = np.where(probs>best_probs,mult,best_multiplicity)
    best_cluster = [cluster_names[i] for i in np.argmax(cluster_probabilities,axis=0)]

    variant_data = variant_data.select(['chromosome','position'])
    variant_data = variant_data.with_columns(pl.Series('cluster_label',best_cluster),
    pl.lit(best_multiplicity).alias('n_copies'))

    return variant_data
def get_variant_cn(variant_data,sample_cn):
    snv_data = variant_data.select(['chromosome','position']).unique()
    snv_data = snv_data.join(sample_cn,how='inner',on='chromosome')
    snv_data = snv_data.filter(pl.col('segment_start')<=pl.col('position'))
    snv_data = snv_data.filter(pl.col('segment_end')>=pl.col('position'))

    return variant_data.join(snv_data,how='inner',on=['chromosome','position'])
def load_labelled_variants(pretrain_data):
    
    variant_data_with_cn = get_variant_cn(pretrain_data.sample_variants,pretrain_data.sample_copy_number)
    variant_data = pretrain_data.sample_variants.join(variant_data_with_cn,on=['chromosome','position'],how='inner')
   
    if pretrain_data.snv_cluster_assignments is None:
        snv_cluster_assignments = get_snv_cluster_assignments_binomial(pretrain_data.cluster_labels,variant_data_with_cn)
    else:
        snv_cluster_assignments = pretrain_data.snv_cluster_assignments
    variant_data = variant_data.join(snv_cluster_assignments,how='inner',on=['chromosome','position'])
    return variant_data
