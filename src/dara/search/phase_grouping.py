"""
Phase-grouping logic: deciding which candidate crystal structures are
"practically indistinguishable via powder XRD" and should be merged into one
answer during search.

Two similarity metrics are supported via the `grouping_metric` flag:
  - "gaussian_cosine" (`peak_pattern_similarity`, the default): symmetric BY
    CONSTRUCTION, L2-intensity-normalized (so a phase's total "similarity
    budget" is capped at 1.0 regardless of how many peaks it has --
    directly fixes the density-bias failure mode confirmed on a real
    Fe2O3_92 (367 peaks) vs. TiO2_136 (20 peaks) comparison, where naive
    position-overlap looked artificially high purely because one pattern
    is dense). Validated against the experimental_eval sweep harness before
    becoming the default. The core similarity function mirrors galaxi's
    `calculate_peak_similarity` (`galaxi/src/galaxi/core/pattern_utils.py`)
    -- the two projects deliberately keep independent copies since each
    calls it in a different context (grouping here vs. scoring there); if
    the similarity formula changes in one, check whether the other should
    follow. This copy adds a weak-peak intensity floor applied before
    normalization.
  - "legacy_jaccard": `PeakMatcher.jaccard_index()` via `batch_peak_matching`
    -- greedy, many-to-one, intensity-descending peak matching. Asymmetric
    by construction (A->B != B->A), patched with a post-hoc (D + D.T)/2
    symmetrization that does not restore the triangle inequality
    AgglomerativeClustering implicitly relies on, and has no normalization
    for differing peak-list density between two phases. Kept only for
    backward compatibility with callers that relied on its exact behavior;
    pass `grouping_metric="legacy_jaccard"` explicitly to use it.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from dara.refine import RefinementPhase
from dara.utils import get_number, load_symmetrized_structure

if TYPE_CHECKING:
    from dara.result import RefinementResult


def calculate_fom_and_strain(
    phase: RefinementPhase, result: "RefinementResult"
) -> tuple[float, float, bool]:
    """
    Calculate the figure of merit for a phase, its lattice strain, and
    whether its structure is fully site-occupancy-ordered.

    For more detail, refer to https://journals.iucr.org/j/issues/2019/03/00/nb5231/.
    The published formula referenced there also has weight-fraction and
    particle-size terms; every prior version of this function had both
    hardcoded to 0 ("disabled for now"), which made them pure dead weight
    (the particle-size term's own coefficient was computed but always
    multiplied by that same hardcoded 0, so removing it changes nothing
    numerically). Removed here rather than silently carried forward --
    properly implementing them is a separate, higher-risk change needing
    its own validation, out of scope for this rewrite. What remains is
    exactly what was actually being computed:
    `1 / (Rietveld_rho + lattice_strain_deviation_pct + eps)`.

    Args:
        result: the refinement result

    Returns
    -------
        (fom, lattice_strain, is_ordered). fom/lattice_strain are 0 if they
        cannot be calculated (e.g. a failed refinement). is_ordered is
        `structure.is_ordered` (pymatgen's own occupancy-ordering
        convention, already used elsewhere in this codebase) -- used by
        `select_group_winner` in place of the old filename-regex heuristic
        (`is_integer_compound`), which could be silently wrong for any
        file-naming inconsistency.
    """
    phase_path = phase.path

    structure, _ = load_symmetrized_structure(phase_path)
    initial_lattice_abc = structure.lattice.abc
    is_ordered = structure.is_ordered

    refined_a = result.lst_data.phases_results[phase_path.stem].a
    refined_b = result.lst_data.phases_results[phase_path.stem].b
    refined_c = result.lst_data.phases_results[phase_path.stem].c

    geweicht = result.lst_data.phases_results[phase_path.stem].gewicht
    geweicht = get_number(geweicht)

    if refined_a is None or geweicht is None or result.lst_data.rho is None:
        return 0, 0, is_ordered

    refined_lattice_abc = [
        refined_a,
        refined_b if refined_b is not None else refined_a,
        refined_c if refined_c is not None else refined_a,
    ]
    refined_lattice_abc = [get_number(x) for x in refined_lattice_abc]

    initial_lattice_abc = np.array(initial_lattice_abc) / 10  # convert to nm
    refined_lattice_abc = np.array(refined_lattice_abc)

    delta_u = (
        np.sum(np.abs(initial_lattice_abc - refined_lattice_abc) / initial_lattice_abc)
        * 100
    )

    lattice_strain = np.mean(
        (refined_lattice_abc - initial_lattice_abc) / initial_lattice_abc
    )

    return 1 / (result.lst_data.rho + delta_u + 1e-4), lattice_strain, is_ordered


def peak_pattern_similarity(
    peaks_a: np.ndarray,
    peaks_b: np.ndarray,
    *,
    sigma: float = 0.1,
    theta_tol: float = 0.2,
    min_intensity_fraction: float = 0.005,
) -> float:
    """Symmetric, density-normalized pattern-overlap similarity in [0, 1].

    `peaks_a`/`peaks_b` are (N, 2) arrays with columns [two_theta, intensity]
    (intensity >= 0). Computes a Gaussian-windowed, L2-intensity-normalized
    cross-correlation over every peak pair within `theta_tol` of each other:
    every pair contributes continuously, weighted by a Gaussian kernel on
    Delta-2theta, with no discrete matched-or-not assignment decision and
    hence no order dependence to get subtly wrong.

    `peak_pattern_similarity(A, B) == peak_pattern_similarity(B, A)` exactly
    -- symmetric by construction (both the Gaussian weight and the L2
    normalization are symmetric under swapping A/B), no post-hoc
    symmetrization needed.

    This is a SIMILARITY, not a distance -- callers doing clustering must
    convert via `1 - peak_pattern_similarity(...)` themselves (see
    `cluster_phases`).

    Peaks with intensity below `min_intensity_fraction` of that pattern's own
    max intensity are dropped before normalization (mirrors DARA's
    `PeakMatcher`'s `intensity_resolution` weak-peak floor) -- this keeps a
    long tail of near-zero-intensity peaks in a dense pattern from
    contaminating the L2 normalization constant.
    """
    theta_a = np.asarray(peaks_a[:, 0], dtype=float) if len(peaks_a) else np.array([])
    int_a = np.asarray(peaks_a[:, 1], dtype=float) if len(peaks_a) else np.array([])
    theta_b = np.asarray(peaks_b[:, 0], dtype=float) if len(peaks_b) else np.array([])
    int_b = np.asarray(peaks_b[:, 1], dtype=float) if len(peaks_b) else np.array([])

    if len(theta_a) == 0 or len(theta_b) == 0:
        return 0.0

    def _apply_weak_peak_floor(theta, intensity):
        if len(intensity) == 0:
            return theta, intensity
        max_intensity = np.max(intensity)
        if max_intensity <= 0:
            return theta, intensity
        keep = intensity >= (min_intensity_fraction * max_intensity)
        return theta[keep], intensity[keep]

    theta_a, int_a = _apply_weak_peak_floor(theta_a, int_a)
    theta_b, int_b = _apply_weak_peak_floor(theta_b, int_b)

    if len(theta_a) == 0 or len(theta_b) == 0:
        return 0.0

    # L2 normalization (unit vectors) makes the cosine similarity
    # mathematically sound, and caps each pattern's own contribution to the
    # score at 1.0 regardless of how many peaks it has (the density-bias fix).
    int_a = int_a / np.sqrt(np.sum(int_a**2) + 1e-9)
    int_b = int_b / np.sqrt(np.sum(int_b**2) + 1e-9)

    dist_matrix = np.abs(theta_a[:, None] - theta_b[None, :])
    weight_matrix = np.exp(-0.5 * (dist_matrix / sigma) ** 2)
    weight_matrix[dist_matrix > theta_tol] = 0.0

    intensity_matrix = int_a[:, None] * int_b[None, :]
    total_score = np.sum(intensity_matrix * weight_matrix)

    return float(np.clip(total_score, 0.0, 1.0))


def cluster_phases(
    distance_matrix: np.ndarray, distance_threshold: float
) -> np.ndarray:
    """Thin, tested wrapper around AgglomerativeClustering with the project's
    standard grouping settings (average linkage, precomputed metric,
    n_clusters=None) -- exists so every grouping call site uses IDENTICAL
    clustering code, not just identical library defaults that could silently
    drift apart if each call site configured sklearn separately.

    `distance_matrix` must be square, symmetric, entries in [0, 1], zero
    diagonal. Returns an (N,) array of integer cluster labels.
    """
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage="average",
    )
    clusterer.fit(distance_matrix)
    return clusterer.labels_


GroupingMetric = Literal["legacy_jaccard", "gaussian_cosine"]


def group_phases(
    all_phases_result: dict[RefinementPhase, "RefinementResult | None"],
    distance_threshold: float = 0.05,
    grouping_metric: GroupingMetric = "gaussian_cosine",
) -> dict[RefinementPhase, dict[str, float | int]]:
    """
    Group the phases based on their similarity.

    Args:
        all_phases_result: the result of all the phases
        distance_threshold: the distance threshold for clustering, default to 0.05
        grouping_metric: which similarity metric to use -- "gaussian_cosine"
            (default, the symmetric, density-normalized metric; see module
            docstring) or "legacy_jaccard" (the older asymmetric
            greedy-matched pseudo-Jaccard metric, patched with a
            (D+D.T)/2 symmetrization; kept for backward compatibility).

    Returns
    -------
        a dictionary containing the group id and the figure of merit for each phase
    """
    grouped_result = {}

    # handle the case where there is no result for a phase. is_ordered
    # defaults to False here (rather than loading the structure just for
    # this rarely-hit failed-refinement branch) -- these entries are
    # already deprioritized via fom=0, so the sentinel only affects
    # select_group_winner's tie-break among other equally-fom=0 failures.
    for phase, result in all_phases_result.items():
        if result is None:
            grouped_result[phase] = {
                "group_id": -1,
                "fom": 0,
                "lattice_strain": 0,
                "is_ordered": False,
            }

    all_phases_result = {
        phase: result
        for phase, result in all_phases_result.items()
        if result is not None
    }

    if len(all_phases_result) <= 1:
        for phase, result in all_phases_result.items():
            fom, lattice_strain, is_ordered = calculate_fom_and_strain(phase, result)
            grouped_result[phase] = {
                "group_id": 0,
                "fom": fom,
                "lattice_strain": lattice_strain,
                "is_ordered": is_ordered,
            }
        return grouped_result

    peaks = []

    for phase, result in all_phases_result.items():
        all_peaks = result.peak_data
        peaks.append(
            all_peaks[all_peaks["phase"] == phase.path.stem][
                ["2theta", "intensity"]
            ].values
        )

    if grouping_metric == "gaussian_cosine":
        n = len(peaks)
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                sim = peak_pattern_similarity(peaks[i], peaks[j])
                d = 1.0 - sim
                distance_matrix[i, j] = d
                distance_matrix[j, i] = d
    elif grouping_metric == "legacy_jaccard":
        # Imported lazily to avoid a hard dependency on ray/tree.py's batch
        # dispatcher from this module for callers that only want the new
        # metric -- this is legacy-metric plumbing scheduled for removal
        # once "gaussian_cosine" becomes the only supported metric.
        from dara.search.tree import batch_peak_matching

        pairwise_similarity = batch_peak_matching(
            [p for p in peaks for _ in peaks],
            [p for _ in peaks for p in peaks],
            return_type="jaccard",
        )
        distance_matrix = 1 - np.array(pairwise_similarity).reshape(
            len(peaks), len(peaks)
        )
        # current peak matching algorithm is not a symmetric metric.
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
    else:
        raise ValueError(f"Unknown grouping_metric: {grouping_metric!r}")

    labels = cluster_phases(distance_matrix, distance_threshold)

    for i, cluster in enumerate(labels):
        phase = list(all_phases_result.keys())[i]
        result = list(all_phases_result.values())[i]
        fom, lattice_strain, is_ordered = calculate_fom_and_strain(phase, result)
        grouped_result[phase] = {
            "group_id": cluster,
            "fom": fom,
            "lattice_strain": lattice_strain,
            "is_ordered": is_ordered,
        }

    return grouped_result


def is_integer_compound(phase: RefinementPhase) -> bool:
    """Legacy winner-selection proxy: True unless the CIF filename contains a
    decimal-point number (e.g. "Fe2.5O4"), in which case it's treated as a
    "fractional"/disordered-occupancy variant. This is a filename-string
    heuristic, not an inspection of actual structure/occupancy data.
    Superseded by `structure.is_ordered` (threaded through
    `calculate_fom_and_strain` and used by `select_group_winner`) -- kept
    only as a debug/cross-check helper so old-vs-new winner choice can be
    diffed during rollout validation (expected to agree except in genuine
    ordered/disordered-mix cases). Not used for any selection decision.
    """
    return not bool(re.search(r"\d+\.\d+", phase.path.name))


def select_group_winner(members: list[dict]) -> dict:
    """Pick the representative phase for one cluster of phases considered
    indistinguishable by XRD.

    Each item in `members` must have at least `"phase"` (a RefinementPhase),
    `"fom"` (float), and `"is_ordered"` (bool, from
    `calculate_fom_and_strain`/`structure.is_ordered`) keys. Prefers ordered
    (fully-occupied) members over disordered/partial-occupancy ones,
    breaking ties by highest figure-of-merit; falls back to highest-FOM
    among all members if none are ordered.

    This consolidates two previously independent, differently-shaped copies
    of the same logic (SearchTree.expand_node's tuple-sort-key `max()` over
    a flat grouped_results dict, and SearchTree.__init__'s filter-then-max
    over a phase_group_mapping list) into one function. The two were
    verified equivalent: since Python's tuple comparison is lexicographic,
    `max(members, key=lambda x: (x["is_ordered"], x["fom"]))` always selects
    among ordered members first (any (True, *) tuple compares greater than
    any (False, *) tuple regardless of the second element), falling back to
    comparing fom only when every member has the same is_ordered value --
    exactly the filter-then-max behavior, just expressed as a single sort
    key instead of two branches.
    """
    return max(members, key=lambda m: (m["is_ordered"], m["fom"]))
