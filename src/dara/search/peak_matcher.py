from typing import Any, Literal, Optional

import numpy as np
from scipy.spatial.distance import cdist

DEFAULT_ANGLE_TOLERANCE = 0.2  # maximum difference in angle
DEFAULT_INTENSITY_TOLERANCE = 2  # maximum ratio of the intensities
# maximum ratio of the intensities to be considered as missing instead of wrong intensity
DEFAULT_MAX_INTENSITY_TOLERANCE = 5


def absolute_log_error(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculate the absolute error of two arrays in log space.
    """
    x = np.clip(x, 1e-10, None)
    y = np.clip(y, 1e-10, None)
    return np.abs(np.log(x) - np.log(y))


def distance_matrix(peaks1: np.ndarray, peaks2: np.ndarray) -> np.ndarray:
    """
    Return the distance matrix between two sets of peaks.
    """
    position_distance = cdist(
        peaks1[:, 0].reshape(-1, 1), peaks2[:, 0].reshape(-1, 1), metric="cityblock"
    )
    intensity_distance = cdist(
        peaks1[:, 1].reshape(-1, 1),
        peaks2[:, 1].reshape(-1, 1),
        metric=absolute_log_error,
    )
    return np.sum(np.array([position_distance, intensity_distance]), axis=0)


def find_best_match(
    peak_calc: np.ndarray,
    peak_obs: np.ndarray,
    angle_tolerance: float = DEFAULT_ANGLE_TOLERANCE,
    intensity_tolerance: float = DEFAULT_INTENSITY_TOLERANCE,
    max_intensity_tolerance: float = DEFAULT_MAX_INTENSITY_TOLERANCE,
) -> dict[str, Any]:
    matched = []
    extra = []
    wrong_intens = []

    if len(peak_obs) == 0:
        return {
            "missing": np.array([]).reshape(-1),
            "matched": np.array([]).reshape(-1, 2),
            "extra": np.arange(len(peak_calc)),
            "wrong_intensity": np.array([]).reshape(-1, 2),
        }
        
    # Nothing has been calculated yet
    if len(peak_calc) == 0:
        return {
            "missing": np.arange(len(peak_obs)),
            "matched": np.array([]).reshape(-1, 2),
            "extra": np.array([]).reshape(-1),
            "wrong_intensity": np.array([]).reshape(-1, 2),
        }

    residual_peak_obs = peak_obs.copy()
    
    # Process calculated peaks (sorted by intensity descending)
    sorted_indices = np.argsort(peak_calc[:, 1])[::-1]
    
    #for peak in peak_calc:
    #    print(f'DEBUG peak_calc: {peak}')
    
    for peak_idx in sorted_indices:
        peak = peak_calc[peak_idx]

        all_close_obs_peaks_idx = np.where(
            np.abs(residual_peak_obs[:, 0] - peak[0]) <= angle_tolerance
        )[0]
        all_close_obs_peaks = residual_peak_obs[all_close_obs_peaks_idx]

        if len(all_close_obs_peaks) == 0:
            extra.append(peak_idx)
            continue

        best_match_sub_idx = np.argmin(
            distance_matrix(np.array([peak]), all_close_obs_peaks).reshape(-1)
        )
        best_match_idx = all_close_obs_peaks_idx[best_match_sub_idx]

        # at this point, so long as angle difference is low, peaks are counted as match 
        matched.append((peak_idx, best_match_idx))
        
        # Substracting observed residual intensity with the calculated peak intensity
        #obs_I_before = residual_peak_obs[best_match_idx, 1]
        residual_peak_obs[best_match_idx, 1] -= peak[1]

    all_assigned = {m[1] for m in matched}
    missing = [i for i in range(len(peak_obs)) if i not in all_assigned]

    to_be_deleted = set()
    for i in range(len(matched)):
        peak_idx = matched[i][1]
        peak_intensity_diff = absolute_log_error(
            peak_obs[peak_idx][1],
            peak_obs[peak_idx][1] - residual_peak_obs[peak_idx][1],
        )
        
        if peak_intensity_diff > np.log(max_intensity_tolerance):
            missing.append(peak_idx)
            extra.append(matched[i][0])
            to_be_deleted.add(i)
        elif peak_intensity_diff > np.log(intensity_tolerance):
            wrong_intens.append(matched[i])

    matched = [m for i, m in enumerate(matched) if i not in to_be_deleted]

    return {
        "missing": missing,
        "matched": matched,
        "extra": extra,
        "wrong_intensity": wrong_intens,
    }


def merge_peaks(peaks: np.ndarray, resolution: float = 0.02) -> np.ndarray:
    if len(peaks) <= 1 or resolution == 0.1:
        return peaks

    peaks = peaks[np.argsort(peaks[:, 0])]
    merge_to = np.arange(len(peaks))
    two_thetas = peaks[:, 0]

    for i in range(1, len(peaks)):
        if np.abs(two_thetas[i - 1] - two_thetas[i]) <= resolution:
            merge_to[i] = merge_to[i - 1]

    ptr_1 = ptr_2 = merge_to[0]
    new_peaks_list = []
    while ptr_1 < len(peaks):
        while ptr_2 < len(peaks) and merge_to[ptr_2] == ptr_1:
            ptr_2 += 1
        angles = peaks[ptr_1:ptr_2, 0]
        intensities = peaks[ptr_1:ptr_2, 1]

        updated_angle = angles @ intensities / np.sum(intensities)
        updated_intensity = np.sum(intensities)
        new_peaks_list.append([updated_angle, updated_intensity])

        ptr_1 = ptr_2

    return np.array(new_peaks_list)


class PeakMatcher:
    """
    Peak matcher class to match the calculated peaks with the observed peaks.
    """

    def __init__(
        self,
        peak_calc: np.ndarray,
        peak_obs: np.ndarray,
        peak_obs_orig: np.ndarray | None = None,
        intensity_resolution: float = 0,
        angle_resolution: float = 0.3,
        angle_tolerance: float = DEFAULT_ANGLE_TOLERANCE,
        intensity_tolerance: float = DEFAULT_INTENSITY_TOLERANCE,
        max_intensity_tolerance: float = DEFAULT_MAX_INTENSITY_TOLERANCE,
        debug: bool = False,
    ):
        self.intensity_resolution = intensity_resolution
        self.angle_resolution = angle_resolution
        self.debug = debug

        # Initialize overrides (Fix for AttributeError)
        self._extra_override = None
        
        peak_calc = peak_calc.reshape(-1, 2)
        peak_obs = peak_obs.reshape(-1, 2)

        peak_calc = peak_calc[
            (peak_calc[:, 1] > 0)
            & (peak_calc[:, 1] > intensity_resolution * peak_calc[:, 1].max(initial=0))
        ]
        self.peak_calc = merge_peaks(peak_calc, resolution=0.02)
        self.overlap = (np.array([]).reshape(-1, 2), np.array([]).reshape(-1, 2))

        peak_obs = peak_obs[
            (peak_obs[:, 1] > 0)
            & (peak_obs[:, 1] > intensity_resolution * peak_obs[:, 1].max(initial=0))
        ]
        self.peak_obs = merge_peaks(peak_obs, resolution=0.02)

        #print(f'DEBUG init in PeakMatcher DEBUG initial peak_calc: {self.peak_calc}')
        #print(f'DEBUG init in PeakMatcher initial peak_obs: {self.peak_obs}')

        # 1. Run Standard Match
        self._result = find_best_match(
            self.peak_calc,
            self.peak_obs,
            angle_tolerance=angle_tolerance,
            intensity_tolerance=intensity_tolerance,
            max_intensity_tolerance=max_intensity_tolerance,
        )

        # 2. Salvage Step (Now safe to access self.extra)
        if peak_obs_orig is not None and len(self.extra) > 0:
            self._salvage_extras(peak_obs_orig, angle_tolerance)

    @property
    def missing(self) -> np.ndarray:
        missing = self._result["missing"]
        missing = np.array(missing).reshape(-1)
        return (
            self.peak_obs[missing] if len(missing) > 0 else np.array([]).reshape(-1, 2)
        )

    @property
    def matched(self) -> tuple[np.ndarray, np.ndarray]:
        matched = self._result["matched"]
        matched = np.array(matched).reshape(-1, 2)
        return (
            self.peak_calc[matched[:, 0]]
            if len(matched) > 0
            else np.array([]).reshape(-1, 2),
            self.peak_obs[matched[:, 1]]
            if len(matched) > 0
            else np.array([]).reshape(-1, 2),
        )

    @property
    def extra(self) -> np.ndarray:
        # Check override first
        if self._extra_override is not None:
            return self._extra_override
            
        extra = self._result["extra"]
        extra = np.array(extra).reshape(-1)
        
        return self.peak_calc[extra] if len(extra) > 0 else np.array([]).reshape(-1, 2)

    @property
    def wrong_intensity(self) -> tuple[np.ndarray, np.ndarray]:
        wrong_intens = self._result["wrong_intensity"]
        wrong_intens = np.array(wrong_intens).reshape(-1, 2)
        return (
            self.peak_calc[wrong_intens[:, 0]]
            if len(wrong_intens) > 0
            else np.array([]).reshape(-1, 2),
            self.peak_obs[wrong_intens[:, 1]]
            if len(wrong_intens) > 0
            else np.array([]).reshape(-1, 2),
        )   
            
    def _salvage_extras(self, peak_obs_orig: np.ndarray, tolerance: float):
        if len(peak_obs_orig) == 0: 
            return

        # Calculate distances between Extra peaks and Original Observed peaks
        dists = cdist(self.extra[:, 0:1], peak_obs_orig[:, 0:1], metric='cityblock')
        min_dists = np.min(dists, axis=1)
        closest_indices = np.argmin(dists, axis=1)

        is_salvageable = min_dists <= tolerance
        
        if not np.any(is_salvageable):
            return

        salvaged_calc = self.extra[is_salvageable]
        salvaged_obs = peak_obs_orig[closest_indices[is_salvageable]]
        
        #print(f'DEBUG extra peaks before salvage: {self.extra}')
        #print(f'DEBUG peak_obs_orig: {peak_obs_orig}')
        #print(f'DEBUG salvageable extras: {salvaged_calc}')
        #print(f'DEBUG corresponding original obs peaks: {salvaged_obs}')

        # Construct overlap tuple, delete salvaged peaks from extra, and update overlap
        self.overlap = (salvaged_calc, salvaged_obs)
        self._extra_override = self.extra[~is_salvageable]
        

    def calculate_intensity_score(
        self,
        I_matched: float,
        I_wrong_intensity: float,
        I_missing: float,
        I_extra: float,
        I_overlap: float,
        I_obs_total: float,
        matched_coeff: float = 1,
        wrong_intensity_coeff: float = 0,
        missing_coeff: float = 0,
        extra_coeff: float = 0,
        overlap_coeff: float = 0,
    ) -> float:
        eps = 1e-12
        #I_phase = I_matched + I_wrong_intensity + I_extra + I_overlap + eps
        I_phase = I_matched + I_wrong_intensity + I_extra + eps
        
        ratio_matched = I_matched / I_phase
        ratio_wrong_intensity = I_wrong_intensity / I_phase
        ratio_missing = I_missing / I_obs_total
        ratio_extra = I_extra / I_phase
        ratio_overlap = I_overlap / I_phase

        return (
            matched_coeff * ratio_matched
            + wrong_intensity_coeff * ratio_wrong_intensity
            + missing_coeff * ratio_missing
            + extra_coeff * ratio_extra
            + overlap_coeff * ratio_overlap
        )

    def score(
        self,
        matched_coeff: float = 1,
        wrong_intensity_coeff: float = 0,
        missing_coeff: float = 0,
        extra_coeff: float = 0,
        overlap_coeff: float = 0,
    ) -> float:
        matched_obs, matched_calc = self.matched
        wrong_intens_obs, wrong_intens_calc = self.wrong_intensity
        
        matched_peaks = min([matched_obs, matched_calc], key=lambda x: x[:, 1].sum()) if len(matched_obs) > 0 else np.empty((0, 2))
        wrong_intens_peaks = min([wrong_intens_obs, wrong_intens_calc], key=lambda x: x[:, 1].sum()) if len(wrong_intens_obs) > 0 else np.empty((0, 2))
        
        I_matched = np.sum(np.abs(matched_peaks[:, 1]))
        I_wrong = np.sum(np.abs(wrong_intens_peaks[:, 1]))
        I_missing = np.sum(np.abs(self.missing[:, 1]))
        I_extra = np.sum(np.abs(self.extra[:, 1]))
        I_overlap = np.sum(np.abs(self.overlap[0][:, 1])) if len(self.overlap[0]) > 0 else 0
        I_obs_total = np.sum(np.abs(self.peak_obs[:, 1])) + 1e-12

        return self.calculate_intensity_score(
            I_matched, I_wrong, I_missing, I_extra, I_overlap, I_obs_total, 
            matched_coeff, wrong_intensity_coeff, missing_coeff, extra_coeff, overlap_coeff
        )

    def jaccard_index(self) -> float:
        matched_calc = self.matched[0]
        wrong_intens_calc = self.wrong_intensity[0]
        matched_obs = self.matched[1]
        wrong_intens_obs = self.wrong_intensity[1]

        total_intensity = np.sum(np.abs(self.peak_obs[:, 1])) + np.sum(
            np.abs(self.peak_calc[:, 1])
        )

        matched_intensity = np.sum(np.abs(matched_calc[:, 1])) + np.sum(
            np.abs(matched_obs[:, 1])
        )
        wrong_intens_intensity = np.sum(np.abs(wrong_intens_calc[:, 1])) + np.sum(
            np.abs(wrong_intens_obs[:, 1])
        )

        if total_intensity == 0:
            return 0

        return (matched_intensity + wrong_intens_intensity) / total_intensity

    def get_isolated_peaks(
        self,
        peak_type: Literal["missing", "extra"],
        min_angle_difference: float = 0.3,
        min_intensity_ratio: float = 0.005,
    ) -> np.ndarray:
        if peak_type == "missing":
            peaks = self.missing
            matched = self.matched[1]
            wrong_intens = self.wrong_intensity[1]
        else:
            peaks = self.extra
            matched = self.matched[0]
            wrong_intens = self.wrong_intensity[0]

        #print(f'DEBUG {peak_type} peaks before isolation: {peaks}')
        #print(f'DEBUG matched obs peaks for isolation: {self.matched[0]}')
        #print(f'DEBUG matched calc peaks for isolation: {self.matched[1]}')
        #print(f'DEBUG wrong intensity obs peaks for isolation: {self.wrong_intensity[0]}')
        #print(f'DEBUG wrong intensity calc peaks for isolation: {self.wrong_intensity[1]}')
        matched = np.concatenate([matched, wrong_intens])

        if len(peaks) == 0:
            return np.array([]).reshape(-1, 2)
        if len(matched) == 0:
            return peaks[peaks[:, 1] > min_intensity_ratio * self.peak_obs[:, 1].max()]

        distance = cdist(
            peaks[:, 0].reshape(-1, 1),
            matched[:, 0].reshape(-1, 1),
            metric="cityblock",
        )
        distance = np.min(distance, axis=1)
        min_intensity = self.peak_obs[:, 1].max() * min_intensity_ratio
        
        peaks = peaks[(distance > min_angle_difference) & (peaks[:, 1] > min_intensity)]
        
        #for peak in peaks:
        #    print(f'DEBUG {peak_type} peak after isolation: {peak}')
        #print(f'DEBUG peak_obs_orig: {self.peak_obs}')

        return peaks

    def visualize(self):
        import matplotlib.pyplot as plt

        missing_obs = self.missing
        matched_obs = self.matched[1]
        wrong_intensity_obs = self.wrong_intensity[1]

        extra_calc = self.extra
        matched_calc = self.matched[0]
        wrong_intensity_calc = self.wrong_intensity[0]

        extra_calc = np.abs(extra_calc)
        matched_calc = np.abs(matched_calc)
        wrong_intensity_calc = np.abs(wrong_intensity_calc)

        extra_calc[:, 1] *= -1
        matched_calc[:, 1] *= -1
        wrong_intensity_calc[:, 1] *= -1

        extra_peaks = extra_calc
        missing_peaks = missing_obs
        matched_peaks = np.concatenate([matched_calc, matched_obs])
        wrong_intensity_peaks = np.concatenate(
            [wrong_intensity_calc, wrong_intensity_obs]
        )

        _, ax = plt.subplots()

        ax.vlines(
            missing_peaks[:, 0],
            0,
            missing_peaks[:, 1],
            color="red",
            alpha=0.5,
            label="missing",
        )
        ax.vlines(
            matched_peaks[:, 0],
            0,
            matched_peaks[:, 1],
            color="green",
            alpha=0.5,
            label="matched",
        )
        ax.vlines(
            extra_peaks[:, 0],
            0,
            extra_peaks[:, 1],
            color="blue",
            alpha=0.5,
            label="extra",
        )
        ax.vlines(
            wrong_intensity_peaks[:, 0],
            0,
            wrong_intensity_peaks[:, 1],
            color="orange",
            alpha=0.5,
            label="wrong intens",
        )

        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("2theta")
        ax.set_ylabel("Intensity")
        ax.legend()