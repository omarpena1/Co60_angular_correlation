# This script will handle binning
import numpy as np

def establish_binning(
    run_configs: list[dict], # dict in the form prepared by `loaders.build_run_configs`

) -> list[dict]:

    num_channels = len(run_configs[0]['loaded_run'].channels)

    for chan_num in range(1, num_channels + 1):
        
        for run in run_configs:
            loaded_run = run['loaded_run']
            
            energies = loaded_run.channels[chan_num].energy
            bins = np.arange(min(energies), max(energies)+1, 2)

            # plot
            counts, edges = np.histogram(energies, bins=bins)

            # run[f'binned_counts_chan{chan_num}'] = counts
            run[chan_num] = dict(binned_counts=counts, edges=edges)

    return run_configs 