import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import os
from .utils import waveformEstimation

def computeWaveformFeatures(user_settings):
    data_folder = user_settings["path_to_data"]
    output_folder = user_settings["output_folder"]

    waveform_all = np.load(os.path.join(data_folder , 'waveform_all.npy'))
    channel_locations = np.load(os.path.join(data_folder, 'channel_locations.npy'))
    sessions = np.load(os.path.join(data_folder , 'session_index.npy'))

    locations = np.load(os.path.join(output_folder, 'locations.npy'))
    positions = np.load(os.path.join(output_folder,'motion.npy'))

    n_nearest_channels = user_settings['waveformCorrection']['n_channels_precomputed']
    n_sample = waveform_all.shape[2]
    n_unit = waveform_all.shape[0]

    chanMap = {
        'xcoords': channel_locations[:, 0],
        'ycoords': channel_locations[:, 1],
    }

    def process_spike(locations_this, dy, channel_locations, n_nearest_channels, waveform_this, chanMap):
        location_new = locations_this.copy()
        location_new[1] -= dy

        distances = np.sqrt(np.sum((channel_locations - location_new[:2])**2, axis=1))
        idx_sort = np.argsort(distances)
        idx_included = idx_sort[:n_nearest_channels]

        waveforms = np.zeros((n_nearest_channels, n_sample))
        waveforms_corrected = np.zeros((n_nearest_channels, n_sample))
        for j in range(n_nearest_channels):
            x = channel_locations[idx_included[j], 0]
            y = channel_locations[idx_included[j], 1]
            
            waveforms[j,:] = waveform_this[idx_included[j],:]
            waveforms_corrected[j,:] = waveformEstimation(
                waveform_this, locations_this, chanMap, location_new, x, y)
        
        return (idx_included, waveforms_corrected, waveforms)

    #%% Run parallel processing with progress bar
    out = Parallel(n_jobs=user_settings["n_jobs"])(
        delayed(process_spike)(locations[k,:2], positions[0, sessions[k]-1], channel_locations, n_nearest_channels, waveform_all[k,:,:], chanMap) 
        for k in tqdm(range(n_unit), desc='Computing waveform features')
    )

    waveforms = np.zeros((n_unit, n_nearest_channels, n_sample))
    waveforms_corrected = np.zeros((n_unit, n_nearest_channels, n_sample))
    waveform_channels = np.zeros((n_unit, n_nearest_channels), dtype=int)
    for k in range(n_unit):
        waveforms[k, :, :] = out[k][2]
        waveforms_corrected[k, :, :] = out[k][1]
        waveform_channels[k, :] = out[k][0]

    # Save the corrected waveforms
    output_folder = user_settings['output_folder']
    np.save(os.path.join(output_folder, 'waveforms.npy'), waveforms)
    np.save(os.path.join(output_folder, 'waveforms_corrected.npy'), waveforms_corrected)
    np.save(os.path.join(output_folder, 'waveform_channels.npy'), waveform_channels)
