import polars as pl


HUMAN_CHROMOSOMES: list[str] = [
    f"chr{i}" for i in range(1, 23)
] + ["chrX", "chrY"]

HUMAN_CHROMOSOME_ENUM = pl.Enum(HUMAN_CHROMOSOMES)