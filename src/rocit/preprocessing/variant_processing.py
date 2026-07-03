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
def label_snv_clusters(snv_clusters:pl.DataFrame,min_clonal_cluster:float=0.9,max_clonal_cluster:float=1.1,min_clonal_fraction:float=0.3,max_fail_fraction:float=0.2)-> pl.DataFrame:
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

    _validate_cluster_labels(cluster_labels,max_fail_fraction)
    return cluster_labels


def get_snv_cluster_assignments_binomial(cluster_labels,variant_data,min_clonal_ccf:float=0.9,max_multiplicity:int=10):

    mult_one_vaf = variant_data['purity']/(variant_data['purity']*variant_data['total_cn'] +(1-variant_data['purity'])*variant_data['normal_total_cn'])

    variant_coverage = variant_data['tumor_alt_count'] + variant_data['tumor_ref_count']

    cluster_label_list = cluster_labels['cluster_label'].to_list()

    # Collect one log-probability vector per (cluster, multiplicity) "peak".
    # Peaks are keyed by cluster row index (not label) so that clusters sharing
    # a label remain distinct.
    peaks = []  # (cluster_idx, mult, logprob over variants)
    for cluster_idx, cluster_row in enumerate(cluster_labels.iter_rows(named=True)):
        # Subclonal clusters only support a single copy; clonal clusters may carry
        # the variant on up to major_cn copies, capped at max_multiplicity.
        max_mult = 1 if cluster_row['cluster_ccf']<min_clonal_ccf else min(variant_data['major_cn'].max(),max_multiplicity)

        for mult in range(1,max_mult+1):
            peak_frac = cluster_row['cluster_ccf']*mult*mult_one_vaf
            # Expected VAF can exceed 1 for high-CCF/high-multiplicity peaks;
            # clip so binom.logpmf stays finite.
            peak_frac = np.clip(peak_frac,0.0,1.0)

            mult_prob = binom.logpmf(variant_data['tumor_alt_count'],variant_coverage,peak_frac)
            mult_prob += np.log(cluster_row['cluster_fraction'])
            mult_prob = np.where(variant_data['major_cn']<mult,-np.inf,mult_prob)
            peaks.append((cluster_idx,mult,mult_prob))

    log_probs = np.array([lp for _,_,lp in peaks])
    norm_data = logsumexp(log_probs,axis=0)
    posteriors = np.exp(log_probs-norm_data)

    # Assign each variant to the cluster with the highest total posterior.
    cluster_probabilities = np.zeros((len(cluster_labels),len(variant_data)))
    for i,(cluster_idx,_,_) in enumerate(peaks):
        cluster_probabilities[cluster_idx]+=posteriors[i]
    best_cluster_idx = np.argmax(cluster_probabilities,axis=0)

    # Pick the multiplicity of the most probable peak *within* the assigned cluster.
    best_multiplicity = np.ones(variant_data.height).astype(int)
    best_probs = np.full(variant_data.height,-np.inf)
    for i,(cluster_idx,mult,_) in enumerate(peaks):
        update = (best_cluster_idx==cluster_idx) & (posteriors[i]>best_probs)
        best_multiplicity = np.where(update,mult,best_multiplicity)
        best_probs = np.where(update,posteriors[i],best_probs)

    best_cluster = [cluster_label_list[i] for i in best_cluster_idx]

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
    qc = somatic_data.qc

    variant_data_with_cn = get_variant_cn(somatic_data.sample_variants,somatic_data.sample_copy_number)
    variant_data = somatic_data.sample_variants.join(variant_data_with_cn,on=['chromosome','position'],how='inner')
    cluster_labels = label_snv_clusters(
        somatic_data.snv_clusters,
        min_clonal_cluster=qc.min_clonal_cluster,
        max_clonal_cluster=qc.max_clonal_cluster,
        min_clonal_fraction=qc.min_clonal_fraction,
        max_fail_fraction=qc.max_fail_fraction,
    )

    if somatic_data.snv_cluster_assignments is None:
        snv_cluster_assignments = get_snv_cluster_assignments_binomial(cluster_labels,variant_data_with_cn,min_clonal_ccf=qc.min_clonal_ccf,max_multiplicity=qc.max_multiplicity)
    else:
        snv_cluster_assignments = somatic_data.snv_cluster_assignments
        snv_cluster_assignments = snv_cluster_assignments.join(cluster_labels,how='inner',on=['cluster_id'])
    
    variant_data = variant_data.join(snv_cluster_assignments,how='inner',on=['chromosome','position'])
    return variant_data
