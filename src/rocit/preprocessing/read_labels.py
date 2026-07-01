import polars as pl
from rocit.constants import HUMAN_CHROMOSOME_ENUM

LABELLED_READ_SCHEMA = {
    'chromosome': HUMAN_CHROMOSOME_ENUM,
    'read_index': pl.String,
    'tumor_read': pl.Boolean,
}

def concat_labelled_reads(read_store):
    if not read_store:
        return pl.DataFrame(schema=LABELLED_READ_SCHEMA)
    return pl.concat(read_store)
