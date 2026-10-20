"""Evaluation: experiment runners (autocorrelation, gap-filling, propagation,
temporal transport, anomaly suite, attribution, basins) and shared
significance tests."""

from .attribution import AttributionResult, run_attribution
from .basins import (
    AccretionPoint,
    accretion_curve,
    basin_of,
    basins,
    random_order_curves,
    summarize,
)
from .conditional import (
    BIN_ORDER,
    covariate_matrix,
    incremental_auc,
    monotonic_trend,
    partial_morans,
    residualize,
    strahler_bin,
    variograms,
)
from .autocorrelation import AutocorrResult, morans_autocorr, segment_means
from .gapfill import GapfillResult, gapfill_eval
from .propagation_test import PropagationResult, propagation_test
from .anomaly import (
    RecoveryResult,
    DenoiseResult,
    TracingResult,
    AnomalyResult,
    evaluate_scores,
    graph_expectation_residual,
    neighbor_residual_at,
    node_temporal_score,
    run_residual,
    run_break,
    run_recovery,
    run_denoise,
    run_tracing,
)
from .temporal_transport import (
    TransportResult,
    common_mode_anomaly,
    load_series,
    seasonal_anomaly,
    temporal_transport,
)
from .robust import (
    RepeatedPairedResult,
    RepeatedResult,
    repeated_cv,
    repeated_paired,
)
from .significance import (
    BootstrapResult,
    CIResult,
    PairedResult,
    bh_adjust,
    bootstrap_delta_ci,
    cluster_bootstrap_ci,
    cluster_bootstrap_delta,
    paired_seed_test,
)

__all__ = ["PairedResult", "BootstrapResult", "CIResult", "paired_seed_test",
           "bootstrap_delta_ci", "cluster_bootstrap_delta", "cluster_bootstrap_ci",
           "bh_adjust",
           "RepeatedResult", "RepeatedPairedResult", "repeated_cv", "repeated_paired",
           "AutocorrResult", "morans_autocorr", "segment_means",
           "AttributionResult", "run_attribution",
           "GapfillResult", "gapfill_eval",
           "PropagationResult", "propagation_test",
           "TransportResult", "temporal_transport", "load_series",
           "common_mode_anomaly", "seasonal_anomaly",
           "AnomalyResult", "RecoveryResult", "DenoiseResult", "TracingResult",
           "run_residual", "run_break", "run_recovery", "run_denoise", "run_tracing",
           "graph_expectation_residual", "node_temporal_score",
           "neighbor_residual_at", "evaluate_scores",
           "strahler_bin", "BIN_ORDER", "covariate_matrix", "residualize",
           "partial_morans", "variograms", "incremental_auc", "monotonic_trend",
           "basin_of", "basins", "accretion_curve", "random_order_curves",
           "AccretionPoint", "summarize"]
