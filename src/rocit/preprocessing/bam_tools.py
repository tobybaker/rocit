import polars as pl
import pysam

def should_filter_read(read) -> bool:
    if read.is_unmapped:
        return True

    if read.is_secondary:
        return True

    if read.is_supplementary:
        return True

    if read.is_qcfail:
        return True

    if read.is_duplicate:
        return True
    return False
def get_reads_from_cn_row(region_row,bam_filepath):
    read_data =[]

    with pysam.AlignmentFile(bam_filepath, "rb") as bam_file:
        for read in bam_file.fetch(region_row['chromosome'],region_row['segment_start'],region_row['segment_end']):

            if should_filter_read(read):
                continue
            read_entry = {'read_index':read.query_name,'chromosome':read.reference_name}
            reference_positions = read.get_reference_positions()
            read_entry['read_start'] = min(reference_positions)
            read_entry['read_end'] = max(reference_positions)
            
            read_data.append(read_entry)
            
    read_data = pl.DataFrame(read_data)
    read_data =read_data.with_columns(pl.col('chromosome').cast(pl.Categorical),pl.col('read_start').cast(pl.Int32),pl.col('read_end').cast(pl.Int32))

    return read_data
def pileup_read_contains_snv(pileup_read,vcf_row):
    

    if pileup_read.is_refskip or pileup_read.is_del:
        return None

    read = pileup_read.alignment
    qpos = pileup_read.query_position
    base_at_pos = read.query_sequence[qpos].upper()
    
    if base_at_pos ==vcf_row['ref']:
        return False
    if base_at_pos ==vcf_row['alt']:
        return True
    return None
        

def check_variant_is_snv(vcf_row):
    #these statements can be made more efficient but it's better to explicitly state the logic
    bases = ['A','C','T','G']
    for label in ['ref','alt']:
        if not vcf_row[label] in bases:
            raise ValueError(f'The reference alelle {vcf_row[label]} is not SNV compatible.')    
    
def get_variant_reads(vcf_row,bam_filepath):
    
    #read flags to filter in the pileup
    flag_filter = (
    pysam.FUNMAP |
    pysam.FSECONDARY |
    pysam.FQCFAIL |
    pysam.FDUP |
    pysam.FSUPPLEMENTARY
    )

    check_variant_is_snv(vcf_row)
    
    read_store = []

    with pysam.AlignmentFile(bam_filepath, "rb") as bam_file:
        pileup_columns = bam_file.pileup(vcf_row['chromosome'],vcf_row['position']-1,vcf_row['position'],flag_filter=flag_filter,truncate=True)
        for pileup_column in pileup_columns:
            for pileup_read in pileup_column.pileups:
            
                contains_snv = pileup_read_contains_snv(pileup_read,vcf_row)

                read_data =  {'read_index':pileup_read.alignment.query_name,'contains_snv':contains_snv}
                read_data.update(vcf_row)
                read_store.append(read_data)
                
    read_store= pl.DataFrame(read_store)
    read_store =read_store.with_columns(pl.col('chromosome').cast(pl.Categorical))
    return read_store.drop_nulls()
