import polars as pl
import numpy as np
from pathlib import Path

from scipy.stats import binom
from scipy.special import logsumexp

class ClusterValidationError(Exception):
    ''' Raises if clusters don't pass QC'''

def _validate_cluster_labels(cluster_labels:pl.DataFrame,max_fail_fraction:float=0.2) -> None:

    unique_cluster_labels = cluster_labels['cluster_label'].unique()
    if not 'pass_clonal' in unique_cluster_labels:
        raise ClusterValidationError(f'Provided SNVs clusters do not contain a clonal peak')
    fail_clusters = cluster_labels.filter(pl.col('cluster_label')=='fail')
    if fail_clusters['cluster_fraction'].sum()>=max_fail_fraction:
        raise ClusterValidationError(f"{fail_clusters['cluster_fraction'].sum():.1%} of SNVs are in fail cluster. This is more than the permitted {max_fail_fraction:.1%}.")
def label_snv_clusters(snv_clusters:pl.DataFrame,min_clonal_cluster:float=0.9,max_clonal_cluster:float=1.1,min_clonal_fraction:float=0.3)-> pl.DataFrame:
    cluster_label_enum = pl.Enum(['pass_clonal','pass','fail'])
    # Define clonal cluster criteria
    is_clonal = (
        pl.col("cluster_ccf").is_between(min_clonal_cluster, max_clonal_cluster)
        & (pl.col("cluster_fraction") > min_clonal_fraction)
    )
    is_above_clonal = pl.col("cluster_ccf") >= max_clonal_cluster

    # Assign cluster labels
    cluster_label_expr = (
        pl.when(is_clonal).then(pl.lit("pass_clonal"))
        .when(is_above_clonal).then(pl.lit("fail"))
        .otherwise(pl.lit("pass"))
        .alias("cluster_label").cast(cluster_label_enum)
    )
    cluster_labels = snv_clusters.with_columns(cluster_label_expr)

    _validate_cluster_labels(cluster_labels)
    return cluster_labels


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
def load_labelled_variants(somatic_data):
    
    variant_data_with_cn = get_variant_cn(somatic_data.sample_variants,somatic_data.sample_copy_number)
    variant_data = somatic_data.sample_variants.join(variant_data_with_cn,on=['chromosome','position'],how='inner')
    cluster_labels = label_snv_clusters(somatic_data.snv_clusters)
    
    if somatic_data.snv_cluster_assignments is None:
        snv_cluster_assignments = get_snv_cluster_assignments_binomial(cluster_labels,variant_data_with_cn)
    else:
        snv_cluster_assignments = somatic_data.snv_cluster_assignments
        snv_cluster_assignments = snv_cluster_assignments.join(cluster_labels,how='inner',on=['cluster_id'])
    
    variant_data = variant_data.join(snv_cluster_assignments,how='inner',on=['chromosome','position'])
    return variant_data
