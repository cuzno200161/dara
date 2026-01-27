"""Phase search module."""
from __future__ import annotations

import copy
from collections import deque
from traceback import print_exc
from turtle import done
from typing import TYPE_CHECKING, Literal

import ray
import os
import numpy as np

from dara.search.tree import BaseSearchTree, SearchTree
from dara.xrd import rasx2xy, raw2xy, xrdml2xy
from pathlib import Path

from scipy.ndimage import gaussian_filter1d

if TYPE_CHECKING:
    from dara.refine import RefinementPhase
    from dara.search.data_model import SearchResult

DEFAULT_PHASE_PARAMS = {
    "gewicht": "0_0",
    "lattice_range": 0.01,
    "k1": "0_0^0.01",
    "k2": "fixed",
    "b1": "0_0^0.005",
    "rp": 4,
}
DEFAULT_REFINEMENT_PARAMS = {"n_threads": 8, "eps1": 0, "eps2": "0_-0.05^0.05"}

@ray.remote
class StopFlag:
    def __init__(self):
        self.stop = False

    def set(self):
        self.stop = True

    def get(self):
        return self.stop
    
    def reset(self):
        self.stop = False

@ray.remote
class PhaseCombinationRegistry:
    def __init__(self):
        self.registered_combinations = set()

    def try_reserve(self, phase_names: list[str]) -> bool:
        """
        Returns True if the combination is new and successfully reserved.
        Returns False if it has already been processed or is currently running.
        """
        combo = frozenset(phase_names)
        if combo in self.registered_combinations:
            return False
        
        self.registered_combinations.add(combo)
        return True

@ray.remote
def _remote_expand_node(search_tree: BaseSearchTree, stop_flag, registry) -> BaseSearchTree:
    try:
        # Check if we should even start
        if ray.get(stop_flag.get.remote()):
            return search_tree

        # Pass the stop_flag into the expansion logic
        search_tree.expand_root(stop_flag, registry) 
        return search_tree
    except ray.exceptions.TaskCancelledError:
        print("Task was cancelled remotely. Returning current state.")
        return search_tree  # Return the tree as it is even if partial
    except Exception as e:
        print_exc()
        raise e


def remote_expand_node(search_tree, nid, stop_flag, registry):
    # Check shared stop flag
    if ray.get(stop_flag.get.remote()):
        return None  # skip expansion

    subtree = BaseSearchTree.from_search_tree(root_nid=nid, search_tree=search_tree)
    return _remote_expand_node.remote(subtree, stop_flag, registry)

def downsample_xy(input_path: Path, output_path: Path, n_points: int, sigma: float =1.0):
    """
    Downsample an XY pattern to n_points between min and max 2θ in the file.
    Assumes file format: two columns, 2theta and intensity.
    """
    # Load pattern
    data = np.loadtxt(input_path)
    twotheta = data[:, 0]
    intensity = data[:, 1]
    
    intensity_smooth = gaussian_filter1d(intensity, sigma=sigma)

    # Create new 2theta grid
    wmin, wmax = twotheta[0], twotheta[-1]
    new_twotheta = np.linspace(wmin, wmax, n_points)

    # Interpolate intensities onto new grid
    new_intensity = np.interp(new_twotheta, twotheta, intensity_smooth)

    # Save downsized pattern
    np.savetxt(output_path, np.column_stack([new_twotheta, new_intensity]), fmt="%.6f %.6f")
    return output_path

def search_phases(
    pattern_path: Path | str,
    downsized_length: int | None = None,
    phases: list[Path | str | RefinementPhase] = [],
    pinned_phases: list[Path | str | RefinementPhase] | None = None,
    max_phases: int = 5,
    wavelength: Literal["Cu", "Co", "Cr", "Fe", "Mo"] | float = "Cu",
    instrument_profile: str | Path = "Aeris-fds-Pixcel1d-Medipix3",
    express_mode: bool = True,
    enable_angular_cut: bool = True,
    phase_params: dict[str, ...] | None = None,
    refinement_params: dict[str, ...] | None = None,
    return_search_tree: bool = False,
    record_peak_matcher_scores: bool = False,
    score_coefficients: dict[str, float] | None = None,
    false_peak_threshold: float = 0.05,
    rpb_threshold: float = 2,
    strain_threshold: float = 0.02,
    early_stopping: bool = False,
) -> list[SearchResult] | SearchTree:
    """
    Search for the best phases to use for refinement.

    Args:
        pattern_path: the path to the pattern file. It has to be in `.xrdml`, `.xy`, `.raw`, or `.rasx` format
        phases: the paths to the CIF files
        pinned_phases: the paths to the pinned phases, which will be included in all the results
        max_phases: the maximum number of phases to refine
        wavelength: the wavelength of the X-ray. It can be either a float or one of the following strings:
            "Cu", "Co", "Cr", "Fe", "Mo", indicating the material of the X-ray source
        instrument_profile: the name of the instrument, or the path to the instrument configuration file (.geq)
        express_mode: whether to use express mode. In express mode, the phases will be grouped first before
            searching, which can significantly speed up the search process.
        enable_angular_cut: whether to enable angular cut, which will run the search on a reduced pattern range
            (wmin, wmax) to speed up the search process.
        phase_params: the parameters for the phase search
        refinement_params: the parameters for the refinement
        return_search_tree: whether to return the search tree. This is mainly used for debugging purposes.
        record_peak_matcher_scores: whether to record the peak matcher scores. This is mainly used for
            debugging purposes.
        score_coefficients: the coefficients for the peak match score calculation    
        false_peak_threshold: the false peak threshold
        rpb_threshold: the RWP threshold
        strain_threshold: the strain threshold
        early_stopping: whether to enable early stopping
    """
    if phase_params is None:
        phase_params = {}

    if refinement_params is None:
        refinement_params = {}

    if not ray.is_initialized():
        num_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", 4))
        #ray.init(num_cpus=num_cpus, local_mode=True)
        ray.init(num_cpus=num_cpus)
        print("DEBUG: Ray resources:", ray.available_resources())

    phase_params = {**DEFAULT_PHASE_PARAMS, **phase_params}
    refinement_params = {**DEFAULT_REFINEMENT_PARAMS, **refinement_params}

    if downsized_length is not None:
        parent_dir = pattern_path.parent if isinstance(pattern_path, Path) else Path(pattern_path).parent
        down_size_dir = parent_dir / (parent_dir.stem + "_downsized")
        os.makedirs(down_size_dir, exist_ok=True)
        original_pattern_path = pattern_path  # keep the original
        downsized_pattern_path = down_size_dir / f"{pattern_path.stem}_downsized.xy"

        # Convert to XY
        if pattern_path.suffix == ".xrdml":
            xy_path = xrdml2xy(pattern_path, downsized_pattern_path)
        elif pattern_path.suffix == ".raw":
            xy_path = raw2xy(pattern_path, downsized_pattern_path)
        elif pattern_path.suffix == ".rasx":
            xy_path = rasx2xy(pattern_path, downsized_pattern_path)
        else:
            xy_path = pattern_path  # assume already XY

        # Downsample
        downsized_path = downsample_xy(xy_path, downsized_pattern_path, n_points=downsized_length)

    # build the search tree
    search_tree = SearchTree(
        pattern_path=downsized_path if downsized_length is not None else original_pattern_path,
        cif_paths=phases,
        pinned_phases=pinned_phases,
        refine_params=refinement_params,
        phase_params=phase_params,
        wavelength=wavelength,
        instrument_profile=instrument_profile,
        express_mode=express_mode,
        enable_angular_cut=enable_angular_cut,
        max_phases=max_phases,
        rpb_threshold=rpb_threshold,
        false_peak_threshold=false_peak_threshold,
        strain_threshold=strain_threshold,
        record_peak_matcher_scores=record_peak_matcher_scores,
        score_coefficients=score_coefficients,
        early_stopping=early_stopping,
    )

    max_worker = ray.cluster_resources()["CPU"]
    #max_worker = 1
    stop_flag = StopFlag.remote()
    registry = PhaseCombinationRegistry.remote()
    pending = [remote_expand_node(search_tree, search_tree.root, stop_flag, registry)]
    to_be_submitted = deque()

    while pending:
        # Standard loop: Check for finished tasks with a short timeout
        done, pending = ray.wait(pending, timeout=0.1)
        early_stop_found = False

        # 1. Process any tasks that are already done
        for task in done:
            try:
                remote_search_tree = ray.get(task)
                if remote_search_tree is None: continue
                
                # --- NEW: RACE CONDITION GATEKEEPER ---
                # Check if the new nodes collide with the Master Tree
                for nid in list(remote_search_tree.nodes.keys()):
                    if nid == remote_search_tree.root: continue # Skip the anchor
                    
                    node = remote_search_tree.get_node(nid)
                    
                    # Check against the MASTER tree (the source of truth)
                    # We assume find_duplicate_node returns the ID of an existing node with same phases
                    duplicate_id = search_tree.find_duplicate_node(node.data.current_phases)
                    
                    if duplicate_id:
                        # We found a collision that the worker missed!
                        # Mark this new node as 'duplicate' so it is NOT added to the queue
                        print(f"Race condition caught: {nid} is a duplicate of {duplicate_id}. Pruning.")
                        node.data.status = "duplicate"
                
                search_tree.add_subtree(anchor_nid=remote_search_tree.root, search_tree=remote_search_tree)
                
                # Check if this is the winner
                if any(node.data.status == "early_stop" for node in remote_search_tree.nodes.values()):
                    early_stop_found = True
                
                # Normal expansion (only if we aren't stopping yet)
                if not early_stop_found:
                    for nid in search_tree.get_expandable_children(remote_search_tree.root):
                        # Double check to ensure we don't queue duplicates
                        if search_tree.get_node(nid).data.status == "duplicate":
                            continue
                        to_be_submitted.append(nid)

            except ray.exceptions.TaskCancelledError:
                continue

        # 2. Check the Kill Conditions
        
        # Condition A: We have the result in hand.
        if early_stop_found:
            print("Refinement result: Early stop found. Terminating all other workers.")
            to_be_submitted.clear()
            for task_ref in pending:
                ray.cancel(task_ref, force=True, recursive=True)
            pending = []
            break

        # Condition B: The global flag is set (Winner finished but is floating in 'pending').
        if ray.get(stop_flag.get.remote()):
            print("Stop flag detected. Retrieving winning result...")
            to_be_submitted.clear() # Stop new work
            
            # HUNT MODE: Blocking wait until we find the winner.
            # We assume the winner is fast/finished, so this loop will exit almost instantly.
            while pending:
                # Wait for the NEXT task to finish (blocking, but fast if winner is ready)
                done_drain, pending = ray.wait(pending, num_returns=1)
                
                for task in done_drain:
                    try:
                        remote_search_tree = ray.get(task)
                        if remote_search_tree:
                            search_tree.add_subtree(anchor_nid=remote_search_tree.root, search_tree=remote_search_tree)
                            if any(node.data.status == "early_stop" for node in remote_search_tree.nodes.values()):
                                early_stop_found = True
                    except Exception:
                        pass
                
                # Found it! Kill everyone else immediately.
                if early_stop_found:
                    print("Winner retrieved. Killing remaining tasks.")
                    for task_ref in pending:
                        ray.cancel(task_ref, force=True, recursive=True)
                    pending = []
                    break
            
            # Break the outer main loop
            break

        # 3. Refill the queue (Normal operation)
        while len(pending) < max_worker and to_be_submitted:
            nid = to_be_submitted.popleft()
            pending.append(remote_expand_node(search_tree, nid, stop_flag, registry))

    if not return_search_tree:
        return search_tree.get_search_results()
    return search_tree