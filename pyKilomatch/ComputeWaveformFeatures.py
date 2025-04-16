import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import os
from .utils import waveformEstimation

def computeWaveformFeatures(user_settings):
    """ Compute the corrected waveforms based on the motion of the probe.
    The corrected waveforms on the reference probe are computed using the Kriging interpolation method
    and saved to the output folder.

    Arguments:
        - user_settings (dict): User settings

    Outputs:
        - waveforms_corrected.npy: The corrected waveforms.
    
    """

    data_folder = user_settings["path_to_data"]
    output_folder = user_settings["output_folder"]

    waveform_all = np.load(os.path.join(data_folder , 'waveform_all.npy'))
    channel_locations = np.load(os.path.join(data_folder, 'channel_locations.npy'))
    sessions = np.load(os.path.join(data_folder , 'session_index.npy'))

    locations = np.load(os.path.join(output_folder, 'locations.npy'))
    positions = np.load(os.path.join(output_folder,'motion.npy'))

    n_sample = waveform_all.shape[2]
    n_channel = waveform_all.shape[1]
    n_unit = waveform_all.shape[0]

    chanMap = {
        'xcoords': channel_locations[:, 0],
        'ycoords': channel_locations[:, 1],
    }

    def process_spike(locations_this, dy, channel_locations, waveform_this, chanMap):
        location_new = locations_this.copy()
        location_new[1] -= dy

        waveforms_corrected = np.zeros((n_channel, n_sample))
        for j in range(n_channel):
            x = channel_locations[j, 0]
            y = channel_locations[j, 1]
            
            waveforms_corrected[j,:] = waveformEstimation(
                waveform_this, locations_this, chanMap, location_new, x, y)
        
        return waveforms_corrected

    # Run parallel processing with progress bar
    out = Parallel(n_jobs=user_settings["n_jobs"])(
        delayed(process_spike)(locations[k,:2], positions[0, sessions[k]-1], channel_locations, waveform_all[k,:,:], chanMap) 
        for k in tqdm(range(n_unit), desc='Computing waveform features')
    )

    waveforms_corrected = np.zeros((n_unit, n_channel, n_sample))
    for k in range(n_unit):
        waveforms_corrected[k, :, :] = out[k]

    # Save the corrected waveforms
    output_folder = user_settings['output_folder']
    np.save(os.path.join(output_folder, 'waveforms_corrected.npy'), waveforms_corrected)
