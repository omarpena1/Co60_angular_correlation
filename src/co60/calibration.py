import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit

def gaussian_plus_linear(
    x: np.ndarray,
    amp: float, 
    mu: float, 
    sigma: float, 
    slope: float, 
    intercept: float,
) -> np.ndarray: 
    """Gaussian on a linear background: approximates Compton continuum + background under peak as linear"""

    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + slope * x + intercept

def fit_single_peak(
    bin_centers: list, 
    counts: list, 
    peak_idx: int, 
    window_sigma_factor=3,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Fit a (Gaussian + linear-background+compton) to a single peak.
    
    Uses the raw FWHM from peak_widths as an initial sigma estimate,
    then fits only within a window of ±window_sigma_factor * sigma_init
    around the peak center. This isolation is what makes the fit stable
    even when the two Co-60 peaks are close.
    
    Returns: (popt, mask, mu, sigma, fwhm)
    popt and mask are only returned for visualizing fits
    mu, sigma, fwhm are used in defining calibration
    """
    peak_pos = bin_centers[peak_idx]
    peak_height = counts[peak_idx]
    
    # Initial sigma estimate from the raw histogram width (determined by scipy.peak_widths)
    widths_bins = peak_widths(counts, [peak_idx], rel_height=0.5)[0]

    bin_width = bin_centers[1] - bin_centers[0]

    sigma_init = (widths_bins[0] * bin_width) / 2.355  # FWHM → sigma

    # Restrict fit window to ±window_sigma_factor * sigma around the peak
    window = window_sigma_factor * sigma_init
    mask = (bin_centers >= peak_pos - window) & (bin_centers <= peak_pos + window)
    x_fit = bin_centers[mask]
    y_fit = counts[mask].astype(float)
    
    if len(x_fit) < 6:
        # Not enough points to define a Gaussian and a linear background; fall back gracefully
        raise ValueError('Not enough points to both define a Gaussian and a linear background')
        # return peak_pos, sigma_init, sigma_init * 2.355, False

    # Initial guess: [amplitude, center, sigma, slope, intercept]
    # Background slope/intercept estimated from the edges of the window
    bg_slope_init = (y_fit[-1] - y_fit[0]) / (x_fit[-1] - x_fit[0]) if len(x_fit) > 1 else 0
    bg_intercept_init = y_fit[0] - bg_slope_init * x_fit[0]
    p0 = [peak_height, peak_pos, sigma_init, bg_slope_init, bg_intercept_init]

    # Bounds: keep amplitude positive, center near the found peak,
    # sigma in a physically reasonable range, background loosely bounded
    bounds_lo = [0,          peak_pos - 3*sigma_init, 0.3*sigma_init, -np.inf, -np.inf]
    bounds_hi = [np.inf,     peak_pos + 3*sigma_init, 5*sigma_init,    np.inf,  np.inf]

    popt, pcov = curve_fit(
        gaussian_plus_linear, x_fit, y_fit,
        p0=p0, bounds=(bounds_lo, bounds_hi),
        maxfev=10000
    )
    amp_fit, mu_fit, sigma_fit, slope_fit, intercept_fit = popt
    perr = np.sqrt(np.diag(pcov))
    
    fwhm_fit = 2.355 * sigma_fit
    return popt, mask, mu_fit, sigma_fit, fwhm_fit

def establish_gaussian_calibration_co60(
    run_configs: list[dict], 
    fwhm_tol_factor=1.0 # This might have to be removed in the future, the energy window for fitting is already too big
) -> list[dict]:

    """
    fwhm_tol_factor: the tolerance window stored per peak will be
                     ± fwhm_tol_factor * fwhm_fit (in eV, post-calibration).
                     0.5 → tight (half-FWHM each side), 1.5 → generous.
    """
    TRUE_ENERGIES = [1173.228, 1332.492]  # Co-60 lines in keV

    num_channels = len(run_configs[0]['loaded_run'].channels)

    for chan_num in range(1, num_channels + 1):

        for i, run in enumerate(run_configs):

            binned_counts = run[chan_num]['binned_counts']
            edges         = run[chan_num]['edges']

            bin_centers   = (edges[:-1] + edges[1:]) / 2

            # 1. Find peaks
            peaks, _ = find_peaks(binned_counts, prominence=50) ### Hard-coded assumption, `prominence=50`
            peak_positions = bin_centers[peaks]

            # Take the two highest-index (highest-energy) peaks as Co-60 lines
            selected_peak_indices = [peaks[-2], peaks[-1]] ### BIG Hard-coded assumption

            # 2. Gaussian fit each peak individually
            fit_results = []
            for pk_idx in selected_peak_indices:
                _, _, mu, sigma, fwhm = fit_single_peak(bin_centers, binned_counts, pk_idx)
                fit_results.append({
                    'mu': mu, 'sigma': sigma, 'fwhm': fwhm,
                })

            # 3. Linear calibration from fitted centers
            uncalib_centers = np.array([res['mu'] for res in fit_results])

            def line(x, m, b):
                return m * x + b
            (m, b), _ = curve_fit(line, uncalib_centers, TRUE_ENERGIES)

            def apply_calib(x, m=m, b=b):
                return m * np.asarray(x) + b

            calib_edges = apply_calib(edges)

            # 4. Convert FWHM to eV and compute tolerances
            # m is eV-per-bin-unit, so fwhm_eV = m * fwhm_bins
            fwhm_eV = [abs(m) * res['fwhm'] for res in fit_results]
            tol_eV  = [fwhm_tol_factor * fw for fw in fwhm_eV]

            # 5. Store results (same keys as before + new Gaussian ones)
        
            ### Highly Co-60-specific Hard-Coded assumptions
            run[chan_num]['x_uncalib_1173']   = fit_results[0]['mu']
            run[chan_num]['x_uncalib_1332']   = fit_results[1]['mu']

            run[chan_num]['FWHM_bins_1173']   = fit_results[0]['fwhm'] / 2.355 * 2.355  # Why do I even need this?
            run[chan_num]['FWHM_bins_1332']   = fit_results[1]['fwhm'] / 2.355 * 2.355

            run[chan_num]['FWHM_energy_1173']    = fwhm_eV[0]
            run[chan_num]['FWHM_energy_1332']    = fwhm_eV[1]

            run[chan_num]['tol_energy_1173']     = tol_eV[0]
            run[chan_num]['tol_energy_1332']     = tol_eV[1]

            # run[f'fit_ok_1173_chan{chan_num}']       = fit_results[0]['fit_ok']
            # run[f'fit_ok_1332_chan{chan_num}']       = fit_results[1]['fit_ok']

            run[chan_num][f'calib_function']   = apply_calib
            run[chan_num][f'calib_m']          = m
            run[chan_num][f'calib_b']          = b
            run[chan_num][f'calib_edges']      = calib_edges

    return run_configs

def copy_calibration(
    run_configs_uncalibrated: list[dict],
    run_configs_calibrated: list[dict],
)-> list[dict]:

    num_channels = len(run_configs_calibrated[0]['loaded_run'].channels)

    for chan_num in range(1, num_channels + 1):
            
        for run_uncalib, run_calib in zip(run_configs_uncalibrated, run_configs_calibrated):
            # Extract calib info from `run_calib` 
            calib_function = run_calib[chan_num]['calib_function']
            calib_m = run_calib[chan_num]['calib_m']
            calib_b = run_calib[chan_num]['calib_b']

            # assign calib infor to `run_uncalib`
            run_uncalib[chan_num]['calib_function'] = calib_function
            run_uncalib[chan_num]['calib_m'] = calib_m
            run_uncalib[chan_num]['calib_b'] = calib_b

            uncalib_edges = run_uncalib[chan_num]['edges'] 

            run_uncalib[chan_num]['calib_edges'] = calib_function(uncalib_edges)

    return run_configs_uncalibrated