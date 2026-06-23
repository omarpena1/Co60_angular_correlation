import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def symmetric_1D_gate_mask(
    E: list,  
    mu: float, 
    tol: float, 
) -> list[bool]:

    """
    Vectorized 1D rectangular gate for events.
    """
    E_keV = np.asarray(E, dtype=float)
    x = E_keV - mu

    return np.abs(x / tol) <= 1.0

# def asymmetric_1D_gate_mask ### Possible extension

def trace_ellipse(
    mu1: float, 
    mu2: float, 
    a: float, 
    b: float, 
    *, 
    theta_rad: float = 0.0,
    n_points: int = 300, 
    color="red", 
    lw=2, 
    label=None
) -> None:

    """
    Plot an ellipse boundary in (E1,E2) keV space. Purely a visualization tool.

    mu1, mu2 : center (keV)
    a, b     : full-length of semi-axes (keV) 
    theta_rad: rotation angle in radians
    """

    t = np.linspace(0, 2*np.pi, n_points)

    # Parametric ellipse before rotation
    xp = a * np.cos(t)
    yp = b * np.sin(t)

    if theta_rad != 0.0:
        c = np.cos(theta_rad)
        s = np.sin(theta_rad)
        x =  c*xp - s*yp
        y =  s*xp + c*yp
    else:
        x, y = xp, yp

    E1 = mu1 + x
    E2 = mu2 + y

    plt.plot(E1, E2, color=color, lw=lw, label=label)

def ellipse_2D_gate_mask(
    E1_keV: list | np.ndarray, # plotted on x-axis, associated with ch1
    E2_keV: list | np.ndarray, # plotted on y-axis, associated with ch2
    mu1: float,                # associated with ch1
    mu2: float,                # associated with ch2
    a: float, # x-axis, associated with ch1
    b: float, # y-axis, associated with ch1
    *, 
    theta_rad=0.0
) -> list[bool]:

    """
    Vectorized ellipse gate for events.
    Default theta=0 => axis-aligned ellipse. Otherwise, will define an oblique ellipse

    """

    E1_keV = np.asarray(E1_keV, dtype=float)
    E2_keV = np.asarray(E2_keV, dtype=float)

    x = E1_keV - mu1
    y = E2_keV - mu2

    if theta_rad != 0.0:
        c = np.cos(theta_rad)
        s = np.sin(theta_rad)
        xp =  c*x + s*y
        yp = -s*x + c*y
    else:
        xp, yp = x, y

    # avoid divide-by-zero
    a = float(a)
    b = float(b)
    if a <= 0 or b <= 0:
        raise ValueError(f"Ellipse axes must be positive. Got a={a}, b={b}")

    return (xp/a)**2 + (yp/b)**2 <= 1.0

def coincidence_count_within_2D_gate_co60(
    run: dict,
    co60_tols: dict,
    *,
    gate_mask = ellipse_2D_gate_mask,
    k_tol=1.0,
    theta_rad=0.0,
    return_masks=False, 
    return_gated_points=False
) -> dict:

    """
    Count coincidences in two ellipse gates (union). 
    Hard-coded to assume these coincidences are registered between Channels 1 and 2

    """

    loaded_run = run['loaded_run']

    ch1 = loaded_run.channels[1]
    ch2 = loaded_run.channels[2]

    ch1_energies_raw = ch1.energy
    ch2_energies_raw = ch2.energy

    # ch1_bins = np.arange(min(ch1_energies_raw), 200, 2)
    # ch2_bins = np.arange(min(ch2_energies_raw), 200, 2)

    # plt.hist(ch1_energies_raw, bins=ch1_bins, alpha=0.5)
    # plt.hist(ch2_energies_raw, bins=ch2_bins, alpha=0.5)
    # plt.xlim(20,200)
    # plt.title('Sanity Check Plot, uncalibrated')
    # plt.grid(True)
    # plt.show()
    # plt.close()

    # calib_ch1 = run[1]['calib_function']
    # calib_ch2 = run[2]['calib_function']

    # # 1) Calibrate event energies to keV
    # E1 = np.asarray(calib_ch1(ch1_energies_raw), dtype=float)
    # E2 = np.asarray(calib_ch2(ch2_energies_raw), dtype=float)

    # ch1_bins = np.arange(min(E1), 2000, 20)
    # ch2_bins = np.arange(min(E2), 2000, 20)
    # plt.hist(E1, bins=ch1_bins, alpha=0.5)
    # plt.hist(E2, bins=ch2_bins, alpha=0.5)

    # plt.title('Sanity Check Plot, post calibration')
    # plt.grid(True)
    # plt.show()
    # plt.close()

    calib_ch1_m = run[1]['calib_m']
    calib_ch1_b = run[1]['calib_b']

    calib_ch2_m = run[2]['calib_m']
    calib_ch2_b = run[2]['calib_b']

    # 1) Calibrate event energies to keV
    E1 = np.asarray(calib_ch1_m*(ch1_energies_raw)+calib_ch1_b , dtype=float)
    E2 = np.asarray(calib_ch2_m*(ch2_energies_raw)+calib_ch2_b, dtype=float)

    # ch1_bins = np.arange(min(E1_alt), 2000, 20)
    # ch2_bins = np.arange(min(E2_alt), 2000, 20)
    # plt.hist(E1_alt, bins=ch1_bins, alpha=0.5)
    # plt.hist(E2_alt, bins=ch2_bins, alpha=0.5)
    # plt.xlim(200,2000)
    # plt.title('Sanity Check Plot, post calibration')
    # plt.grid(True)
    # plt.show()
    # plt.close()

    if E1.shape != E2.shape:
        raise ValueError(f"Energy arrays must have same shape. Got {E1.shape} vs {E2.shape}")

    # # 2) Extract peak centroids from info dicts (keV)
    # Assumes that both channels have been calibrated
    centroids_ch1 = np.asarray([1173.228, 1332.492], dtype=float)
    centroids_ch2 = np.asarray([1173.228, 1332.492], dtype=float)

    tol_1173_ch1 = co60_tols[1]['tol_1173']
    tols_1332_ch1 = co60_tols[1]['tol_1332']

    tol_1173_ch2 = co60_tols[2]['tol_1173']
    tols_1332_ch2 = co60_tols[2]['tol_1332']

    tols_ch1 = np.asarray([tol_1173_ch1, tols_1332_ch1], dtype=float)
    tols_ch2 = np.asarray([tol_1173_ch2, tols_1332_ch2], dtype=float)

    # assume from this point forward that values in centroids_ and tols_ correspond index-to-index for BOTH ch1 and ch2 

    if (len(tols_ch1)+len(tols_ch2)+len(centroids_ch1)+len(centroids_ch2) != 8):
        err = len(tols_ch1)+len(tols_ch2)+len(centroids_ch1)+len(centroids_ch2)
        raise ValueError(f"This method can only be use to count coincidences across 2 windows on 2 detectors for a total of 4 windows. Got {err}")
        # this raise is misleadingly worded pero tira tio
    
    masks = []

    # add threshold mask
    union_mask = np.zeros_like(E1, dtype=bool)

    ellipse_params_list = []

    for i in range(2):
        ch1_key = i%2 # This logic switches between ch1 receving 1173 keV gamma and ch2 receiving the 1332 keV and vice versa
        ch2_key = (i+1)%2

        mu_ch1 = centroids_ch1[ch1_key]
        mu_ch2 = centroids_ch2[ch2_key]
        a = k_tol * tols_ch1[ch1_key]
        b = k_tol * tols_ch2[ch2_key]

        ellipse_params_list.append(tuple([mu_ch1, mu_ch2, a, b]))

    for ellipse_init in ellipse_params_list:
        mask = gate_mask(E1, E2, *ellipse_init, theta_rad=theta_rad)
        masks.append(mask)
        union_mask |= mask

    # threshold mask ### this should be tested more thoroughly
    tthresh1 = np.asarray(ch1.tthresh, dtype=float)
    tthresh2 = np.asarray(ch2.tthresh, dtype=float)

    dtthresh = tthresh1 - tthresh2
    threshold_mask = (np.abs(dtthresh) <= 10)  & (dtthresh > 0) # 10 ns  # (dtthresh > 0) because there should always be an order in which coincidences are received

    union_mask &= threshold_mask

    mask_1173_1332 = masks[0] & threshold_mask 

    mask_1332_1173 = masks[1] & threshold_mask

    results = {
        "n_total": int(len(E1)),
        "n_in_union": int(union_mask.sum()),
        "n_in_1173_1332": int(mask_1173_1332.sum()),
        "n_in_1332_1173": int(mask_1332_1173.sum()),
        "frac_in_union": float(union_mask.mean()) if len(E1) else np.nan,
    }

    if return_masks:
        results["union_mask"] = union_mask
        results["masks"] = masks
        results["ellipse_params_list"] = ellipse_params_list  # handy for plotting

    if return_gated_points:
        results["E1_keV"] = E1
        results["E2_keV"] = E2
        results["E1_gate_keV"] = E1[union_mask]
        results["E2_gate_keV"] = E2[union_mask]

        results["E1_gate_keV_1173_1332"] = E1[mask_1173_1332]
        results["E2_gate_keV_1173_1332"] = E2[mask_1173_1332]

        results["E1_gate_keV_1332_1173"] = E1[mask_1332_1173]
        results["E2_gate_keV_1332_1173"] = E2[mask_1332_1173]


    return results

def collect_coincidence_counts(
    run_configs: list[dict],
    co60_tols: dict,
    *,
    norm_angle: float = 90.0,
    debug_plot_fn=None,   # optional plotting callable, to keep plotting out of logic
) -> pd.DataFrame:

    """
    Run coincidence_count_within_2D_gate_co60 over all runs and return
    a single DataFrame with columns N, N_unc, W, W_unc, indexed by angle.

    Parameters
    ----------
    run_configs   : list of run dicts, each with a loaded_run and calib_function
    co60_tols     : tolerance dict keyed by channel number
    norm_angle    : angle to normalize counts against (default 90.0)
    debug_plot_fn : optional callable(run, res) for debug plots — keeps
                    plotting logic in the notebook where it belongs
    """
    records = []   

    for run in run_configs:
        angle = float(run["angle"])

        res = coincidence_count_within_2D_gate_co60(
            run,
            co60_tols,
            return_masks=True,
            return_gated_points=True,
        )

        if debug_plot_fn is not None:
            debug_plot_fn(run, res)

        N = int(res["n_in_union"])
        N_unc = np.sqrt(N)

        N_in_1173_1332 = int(res["n_in_1173_1332"])
        N_in_1173_1332_unc = np.sqrt(N_in_1173_1332)

        N_in_1332_1173 = int(res["n_in_1332_1173"])
        N_in_1332_1173_unc = np.sqrt(N_in_1332_1173)

        records.append({
            "angle":   angle,
            "N":       N,
            "N_unc":   N_unc,

            "N_in_1173_1332": N_in_1173_1332,
            "N_in_1173_1332_unc": N_in_1173_1332_unc,

            "N_in_1332_1173": N_in_1332_1173,
            "N_in_1332_1173_unc": N_in_1332_1173_unc,
        })

    df = pd.DataFrame(records).set_index("angle")

    N_norm     = df.loc[norm_angle, "N"]
    N_norm_unc = df.loc[norm_angle, "N_unc"]

    df["W"] = df["N"] / N_norm

    df["W_unc"] = np.where(
        df.index == norm_angle,
        0.0,
        np.abs(df["W"]) * np.sqrt(
            (df["N_unc"] / df["N"])**2 + (N_norm_unc / N_norm)**2
        ),
    )

    return df[["N", "N_unc", "W", "W_unc", "N_in_1173_1332", "N_in_1173_1332_unc", "N_in_1332_1173", "N_in_1332_1173_unc"]]



def single_channel_count_within_1D_gate_co60(
    run: dict,
    chan_num: int,
    co60_tols: dict, 
    *, # This asterisk means that every argument after this must be passed by name, not positionally
    gate_mask = symmetric_1D_gate_mask,
    return_masks=False, 
    return_gated_points=False
) -> dict:

    """
    counts within a 1D, symmetrical energy gate around the two Gamma emission peaks of a single Co-60 spectrum. 

    Designed for counting events in single channel trigger analysis 
    """
    loaded_run = run['loaded_run']

    ch = loaded_run.channels[chan_num]

    ch_energies_raw = ch.energy

    # calib_ch = run[chan_num]['calib_function']
    # calib_ch = run[chan_num]['calib_function']
    calib_ch_m = run[chan_num]['calib_m']
    calib_ch_b = run[chan_num]['calib_b']

    # 1) Calibrate event energies to keV
    # E = np.asarray(calib_ch(ch_energies_raw), dtype=float)
    E = np.asarray(calib_ch_m*ch_energies_raw+calib_ch_b, dtype=float)


    # 2) Extract peak centroids from info dicts (keV)

    TRUE_ENERGIES = [1173.228, 1332.492]  # Co-60 lines in keV

    # This assumes that calibration was done w.r.t `TRUE_ENERGIES`
    centroids = np.asarray(TRUE_ENERGIES, dtype=float)

    tol_1173 = co60_tols[chan_num]['tol_1173']
    tol_1332 = co60_tols[chan_num]['tol_1332']

    tols_ch = np.asarray([tol_1173, tol_1332], dtype=float)

    masks = []

    # add threshold mask
    union_mask = np.zeros_like(E, dtype=bool)

    gate_params_list = []

    for i in range(2):
        mu = centroids[i]
        tol = tols_ch[i]

        gate_params_list.append(tuple([mu, tol]))

    for gate_init in gate_params_list:
        mask = gate_mask(E, *gate_init)
        masks.append(mask)
        union_mask |= mask

    mask_1173 = masks[0]
    mask_1332 = masks[1]

    results = {
        "n_total": int(len(E)),
        "n_in_1173_peak": int(mask_1173.sum()),
        "n_in_1332_peak": int(mask_1332.sum()),
        "n_in_1173_or_1332_peaks": int(union_mask.sum()),
        "frac_in_1173_or_1332_peaks": float(union_mask.mean()) if len(E) else np.nan,
    }

    if return_masks:
        results["union_mask"] = union_mask
        results["masks"] = masks
        results["gate_params_list"] = gate_params_list  # handy for plotting

    if return_gated_points:
        results["E_keV"] = E
        results["E_gate_keV"] = E[union_mask]

    return results

def collect_single_trigger_counts(
    run_configs: list[dict],
    chan_num: int, # channel number of the triggering channel
    co60_tols: dict,
    *,
    norm_angle: float = 90.0,
    debug_plot_fn=None,   # optional plotting callable, to keep plotting out of logic
) -> pd.DataFrame:

    """
    Run single_channel_count_within_1D_gate_co60 over all runs and return
    a single DataFrame with columns N, N_unc, indexed by angle.

    Parameters
    ----------
    run_configs   : list of run dicts, each with a loaded_run and calib_function
    chan_num      : int correpsonding to the channel number of the triggering channel
    co60_tols     : tolerance dict keyed by channel number

    debug_plot_fn : optional callable(run, res) for debug plots — keeps
                    plotting logic in the notebook where it belongs
    """
    records = []   

    for run in run_configs:
        angle = float(run["angle"])

        res = single_channel_count_within_1D_gate_co60(
            run,
            chan_num,
            co60_tols,
            return_masks=True,
            return_gated_points=True,
        )

        if debug_plot_fn is not None:
            debug_plot_fn(run, res)

        tot_events = len(run['loaded_run'].channels[chan_num].energy)

        N_1173 = int(res["n_in_1173_peak"])
        N_1173_unc = np.sqrt(N_1173)

        N_1332 = int(res["n_in_1332_peak"])
        N_1332_unc = np.sqrt(N_1332)

        N_comb = int(res["n_in_1173_or_1332_peaks"])
        N_comb_unc = np.sqrt(N_comb)

        records.append({
            "angle":   angle,
            "N_comb":       N_comb,
            "N_comb_unc":   N_comb_unc,
            "N_comb_norm":      N_comb/tot_events,
            "N_comb_unc_norm":  N_comb_unc/tot_events,

            "N_1173":       N_1173,
            "N_1173_unc":   N_1173_unc,
            "N_1173_norm":      N_1173/tot_events,
            "N_1173_unc_norm":  N_1173_unc/tot_events,

            "N_1332":       N_1332,
            "N_1332_unc":   N_1332_unc,
            "N_1332_norm":      N_1332/tot_events,
            "N_1332_unc_norm":  N_1332_unc/tot_events,
        })

    df = pd.DataFrame(records).set_index("angle")

    N_comb_90norm     = df.loc[norm_angle, "N_comb"]
    N_comb_90norm_unc = df.loc[norm_angle, "N_comb_unc"]

    N_1173_90norm     = df.loc[norm_angle, "N_1173"]
    N_1173_90norm_unc = df.loc[norm_angle, "N_1173_unc"]

    N_1332_90norm     = df.loc[norm_angle, "N_1332"]
    N_1332_90norm_unc = df.loc[norm_angle, "N_1332_unc"]

    # df["W_comb"] = df["N_comb"] / N_comb_90norm
    # df["W_1173"] = df["N_1173"] / N_1173_90norm
    # df["W_1332"] = df["N_1332"] / N_1332_90norm

    df["W_comb"] = N_comb_90norm / df["N_comb"]
    df["W_1173"] = N_1173_90norm / df["N_1173"]
    df["W_1332"] = N_1332_90norm / df["N_1332"]

    df["W_comb_unc"] = np.where(
        df.index == norm_angle,
        0.0,
        np.abs(df["W_comb"]) * np.sqrt(
            (df["N_comb_unc"] / df["N_comb"])**2 + (N_comb_90norm_unc / N_comb_90norm)**2
        ),
    )

    df["W_1173_unc"] = np.where(
        df.index == norm_angle,
        0.0,
        np.abs(df["W_1173"]) * np.sqrt(
            (df["N_1173_unc"] / df["N_1173"])**2 + (N_1173_90norm_unc / N_1173_90norm)**2
        ),
    )

    df["W_1332_unc"] = np.where(
        df.index == norm_angle,
        0.0,
        np.abs(df["W_1332"]) * np.sqrt(
            (df["N_1332_unc"] / df["N_1332"])**2 + (N_1332_90norm_unc / N_1332_90norm)**2
        ),
    )

    return df[["N_comb", "N_comb_unc", "N_comb_norm", "N_comb_unc_norm",
               "N_1173", "N_1173_unc", "N_1173_norm", "N_1173_unc_norm",
               "N_1332", "N_1332_unc", "N_1332_norm", "N_1332_unc_norm",
               "W_comb", "W_comb_unc", "W_1173", "W_1173_unc", "W_1332", "W_1332_unc"]]