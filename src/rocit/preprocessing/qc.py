from dataclasses import dataclass

from rocit.config import ConfigError


@dataclass(frozen=True)
class QCThresholds:
    """Immutable, validated set of preprocessing QC thresholds."""

    # SNV read-labelling (snv_data_labeller.py)
    min_variant_reads: int = 3
    phasing_min_p_value: float = 0.1
    all_copies_min_p_value: float = 0.1

    # Variant clustering (variant_processing.py)
    max_fail_fraction: float = 0.2
    min_clonal_cluster: float = 0.9
    max_clonal_cluster: float = 1.1
    min_clonal_fraction: float = 0.3
    min_clonal_ccf: float = 0.9
    max_multiplicity: int = 10

    # Haploblock sizing (shared by SNV + LOH labelling)
    min_haploblock_size: float = 5e5

    # LOH read-labelling (loh_data_labeller.py)
    loh_max_major_cn: int = 4
    loh_min_segment_length: float = 1e6
    loh_subblock_size: int = 100_000
    loh_min_coverage: int = 20
    loh_max_freq_diff: float = 0.05

    # Methylation extraction (extract_pacbio_cpg_info.py)
    min_mapq: int = 0

    # CpG distribution (process_cpg_distribution.py)
    min_n_cpgs: int = 10

    def __post_init__(self) -> None:
        # Probabilities and fractions must lie in the unit interval.
        for name in (
            "phasing_min_p_value",
            "all_copies_min_p_value",
            "max_fail_fraction",
            "min_clonal_fraction",
            "min_clonal_ccf",
            "loh_max_freq_diff",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ConfigError(
                    f"QCThresholds.{name} must be between 0 and 1, got {value}."
                )

        # Clonal CCF window must be a non-empty, correctly ordered interval.
        if self.min_clonal_cluster >= self.max_clonal_cluster:
            raise ConfigError(
                "QCThresholds.min_clonal_cluster must be less than "
                f"max_clonal_cluster, got {self.min_clonal_cluster} >= "
                f"{self.max_clonal_cluster}."
            )

        # Sizes, counts and multiplicities must be strictly positive.
        for name in (
            "min_variant_reads",
            "max_multiplicity",
            "min_haploblock_size",
            "loh_max_major_cn",
            "loh_min_segment_length",
            "loh_subblock_size",
            "loh_min_coverage",
            "min_n_cpgs",
        ):
            value = getattr(self, name)
            if value <= 0:
                raise ConfigError(
                    f"QCThresholds.{name} must be positive, got {value}."
                )

        if self.min_mapq < 0:
            raise ConfigError(
                f"QCThresholds.min_mapq must be non-negative, got {self.min_mapq}."
            )
