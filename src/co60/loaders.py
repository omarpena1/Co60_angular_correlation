# src/co60/loaders.py
from __future__ import annotations
import pickle
from pathlib import Path
from dataclasses import dataclass, field
import numpy as np

from co60.drs4 import readDRS4Binary   # ← imports from sibling module

CACHE_DIR = Path(__file__).parents[2] / "cache"

@dataclass
class ChannelEvents:
    energy: list = field(default_factory=list)
    pedestal: list = field(default_factory=list)
    # voltage: list = field(default_factory=list)
    # time: list = field(default_factory=list)
    tthresh: list = field(default_factory=list)

    # def finalize(self):#, use_abs_energy=True):
    def finalize(self, use_abs_energy=True) -> None:
        """Convert event lists to NumPy arrays. Call once after all events are appended"""
        self.energy = np.asarray(self.energy, dtype=float)
        if use_abs_energy:  # Setting that makes physical energy values when dealing with negative polarity waveforms
            self.energy = np.abs(self.energy)
        self.pedestal = np.asarray(self.pedestal, dtype=float)
        self.tthresh = np.asarray(self.tthresh, dtype=float)
        # voltage/time remain lists of arrays

    def truncate(self, n: int) -> None:
        """Cut all arrays down to the first n events in place"""
        self.energy   = self.energy[:n]
        self.pedestal = self.pedestal[:n]
        self.tthresh  = self.tthresh[:n]

@dataclass
class RunData:
    angle_deg: float
    fname: str
    board_id: int = 3067
    channels: dict = field(default_factory=dict)  # {1: ChannelEvents(), 2: ChannelEvents()}

    def truncate(self, n: int) -> None:
        """Truncate all channels to the first n events."""
        for ch in self.channels.values():
            ch.truncate(n)

def load_run(fname, angle_deg, board_id=3067, channel_ids=(1, 2), max_events=None):
    channels = {ch: ChannelEvents() for ch in channel_ids}

    for i, event_results in enumerate(readDRS4Binary(fname, nevents=max_events, is_negative_polarity=True)):
        board_results = event_results[board_id]

        if any(board_results[ch]["Tthres"] == -99999 for ch in channel_ids):
            continue

        for ch in channel_ids:
            evt = board_results[ch]
            ch_data = channels[ch]
            ch_data.energy.append(evt["Energy"])
            ch_data.pedestal.append(evt["Pedestal"])
            # ch_data.voltage.append(evt["Voltages"])
            # ch_data.time.append(evt["Times"])
            ch_data.tthresh.append(evt["Tthres"])

    for ch in channel_ids:
        # channels[ch].finalize()
        channels[ch].finalize(use_abs_energy=True)


    return RunData(angle_deg=angle_deg, fname=fname, board_id=board_id, channels=channels)


def load_run_single_trigger(fname, angle_deg, triggering_channel, board_id=3067, channel_ids=(1, 2), max_events=None):
    channels = {ch: ChannelEvents() for ch in channel_ids}

    for i, event_results in enumerate(readDRS4Binary(fname, nevents=max_events, is_negative_polarity=True)):
        board_results = event_results[board_id]

        if any(board_results[ch]["Tthres"] == -99999 for ch in [triggering_channel]):
            continue

        for ch in channel_ids:
            evt = board_results[ch]
            ch_data = channels[ch]
            ch_data.energy.append(evt["Energy"])
            ch_data.pedestal.append(evt["Pedestal"])
            # ch_data.voltage.append(evt["Voltages"])
            # ch_data.time.append(evt["Times"])
            ch_data.tthresh.append(evt["Tthres"])                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
 
    for ch in channel_ids:
        # channels[ch].finalize()
        channels[ch].finalize(use_abs_energy=True)


    return RunData(angle_deg=angle_deg, fname=fname, board_id=board_id, channels=channels)

def load_waveforms_for_inspection(
    fname: str,
    board_id: int = 3067,
    channel_ids: tuple = (1, 2),
    event_indices: list[int] = None,   # e.g. [0, 1, 2] for first 3 events
    max_events: int = 100,             # or just grab the first N
) -> dict:
    """Load raw waveforms from a single file for inspection. Not cached."""
    waveforms = {ch: {"voltage": [], "time": [], "tthresh": [], "energy": []} 
                 for ch in channel_ids}

    for i, event_results in enumerate(readDRS4Binary(fname, nevents=max_events,
                                                      is_negative_polarity=True)):
        if event_indices is not None and i not in event_indices:
            continue
        board_results = event_results[board_id]
        for ch in channel_ids:
            evt = board_results[ch]
            waveforms[ch]["voltage"].append(evt["Voltages"])
            waveforms[ch]["time"].append(evt["Times"])
            waveforms[ch]["tthresh"].append(evt["Tthres"])
            waveforms[ch]["energy"].append(evt["Energy"])

    return waveforms

def build_run_configs(
    run_specs: list[dict],   # list of {"fname": ..., "angle": ...} dicts — YOU provide these
    cache_key: str,          # a name for the cache file, e.g. "coincidence"
    load_fn=load_run,        # which loader to use; defaults to coincidence loader
    load_fn_kwargs: dict = None,
) -> list[dict]:
    """Load runs and cache to disk. Subsequent calls return the cache instantly."""

    if load_fn is None:
        load_fn = load_run
    if load_fn_kwargs is None:
        load_fn_kwargs = {}

    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / f"{cache_key}.pkl"

    if cache_path.exists():
        print(f"[loaders] Cache hit — loading '{cache_key}' from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"[loaders] No cache found — running {load_fn.__name__} for '{cache_key}'...")
    configs = []
    for spec in run_specs:
        run = dict(fname=spec["fname"], angle=spec["angle"])
        run["loaded_run"] = load_fn(
            fname=spec["fname"],
            angle_deg=spec["angle"],
            **load_fn_kwargs,        # ← unpacks {"principle_trigger": 1} as a keyword argument
        )
        configs.append(run)
        print(f"  loaded angle={spec['angle']}°")

    with open(cache_path, "wb") as f:
        pickle.dump(configs, f)
    print(f"[loaders] Saved to {cache_path}")
    return configs

def load_run_configs(
  cache_key: str,          # a name for the cache file
) -> list[dict]:
    """ Function used to load run from cache knowing knowing only its cache key"""
    
    cache_path = CACHE_DIR / f"{cache_key}.pkl"
    if cache_path.exists():
        print(f"[loaders] Cache hit — loading '{cache_key}' from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"No such file or directory: '{cache_path}'")

def find_greatest_common_count_num(
    run_configs: list[dict],
) -> int :
    """Function will """

    num_angles = len(run_configs)

    run_config_valid_counts = np.full(num_angles, np.nan)

    for i, run in enumerate(run_configs):
        loaded_run = run['loaded_run']
        arbitrary_channel = 1 # Since there should be the same number of oscilloscope snapshots between channels, this could be any channel

        num_valid_counts = len(loaded_run.channels[arbitrary_channel].energy)
        num_valid_counts_other_channel = len(loaded_run.channels[2].energy)

        assert num_valid_counts == num_valid_counts_other_channel, "Both channels must have the same number of valid events"

        run_config_valid_counts[i] = num_valid_counts

    greatest_common_count_num = np.min(run_config_valid_counts)

    return int(greatest_common_count_num)


