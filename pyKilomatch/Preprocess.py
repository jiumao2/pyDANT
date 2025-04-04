import numpy as np
import os
import h5py
from scipy.ndimage import gaussian_filter1d
from joblib import Parallel, delayed
from tqdm import tqdm
from .utils import spikeLocation

def preprocess(user_settings):
    # load the data
    print(f'Loading {user_settings["path_to_data"]}...')
    spikeInfo = h5py.File(user_settings["path_to_data"])

    # make a folder to store the data
    output_folder = user_settings["output_folder"]
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(f'The output will be saved to {output_folder}!')

    # make a folder to store the figures
    figures_folder = os.path.join(output_folder, 'Figures')
    if not os.path.exists(figures_folder):
        os.makedirs(figures_folder)

    # %% Preprocess the spikeInfo
    n_unit = len(spikeInfo['spikeInfo']['RatName'])
    keys = spikeInfo['spikeInfo'].keys()

    Kcoords = np.array(spikeInfo[spikeInfo['spikeInfo']['Kcoords'][0][0]][0])
    Xcoords = np.array(spikeInfo[spikeInfo['spikeInfo']['Xcoords'][0][0]][0])
    Ycoords = np.array(spikeInfo[spikeInfo['spikeInfo']['Ycoords'][0][0]][0])
    SpikeTimes = [
        np.squeeze(
            np.array(spikeInfo[spikeInfo['spikeInfo']['SpikeTimes'][k][0]])) for k in range(n_unit)]
    SessionIndex = np.array([np.array(spikeInfo[spikeInfo['spikeInfo']['SessionIndex'][k][0]][0][0]) for k in range(n_unit)],
        dtype=np.int64)
    Waveform = [np.transpose(np.array(
        spikeInfo[spikeInfo['spikeInfo']['Waveform'][k][0]])) for k in range(n_unit)]

    if "PETH" in keys:
        PETH = [np.squeeze(
            np.array(
                spikeInfo[spikeInfo['spikeInfo']['PETH'][k][0]])) for k in range(n_unit)]

    # %% validate the data
    n_session = np.max(SessionIndex)
    if n_session != len(np.unique(SessionIndex)):
        raise ValueError('SessionIndex should start from 1 and be coninuous without any gaps!')

    print(n_session, 'sessions found!')
    # %% preprocessing data
    chanMap = {
        'xcoords': Xcoords,
        'ycoords': Ycoords,
        'kcoords': Kcoords
    }

    def process_spike_info(waveform, spike_times):
        # compute the location of each unit
        x, y, z, amp = spikeLocation(waveform, chanMap,
                                    user_settings['spikeLocation']['n_nearest_channels'],
                                    user_settings['spikeLocation']['location_algorithm'])
        
        # compute the peak channel
        peaks_to_trough = np.max(waveform, axis=1) - np.min(waveform, axis=1)
        channel = np.argmax(peaks_to_trough)
        
        spike_times = spike_times - spike_times[0]

        # compute the autocorrelogram feauture
        auto_corr = None
        if "AutoCorr" in user_settings['motionEstimation']['features'] or \
                "AutoCorr" in user_settings['clustering']['features']:
            window = user_settings['autoCorr']['window']  # ms
            
            s = np.zeros(int(np.max(spike_times))+1, dtype=np.int64)
            s[np.int64(spike_times)] = 1

            auto_corr = np.zeros(window)
            for i in range(window):
                auto_corr[i] = np.correlate(s[i+1:], s[:-i-1])[0]

            auto_corr = gaussian_filter1d(auto_corr, user_settings['autoCorr']['gaussian_sigma'])
            auto_corr = auto_corr / np.max(auto_corr)

        # compute the ISI feature
        isi_out = None
        if "ISI" in user_settings['motionEstimation']['features'] or \
                "ISI" in user_settings['clustering']['features']:
            isi = np.diff(spike_times)
            isi_hist = np.histogram(isi, bins=np.arange(0, user_settings['ISI']['window'], user_settings['ISI']['binwidth']))[0]
            isi_freq = isi_hist / np.sum(isi_hist)
            isi_out = gaussian_filter1d(isi_freq, user_settings['ISI']['gaussian_sigma'])
        
        return (x,y,z,amp,channel,auto_corr,isi_out)

    # %%
    # print('Start preprocessing spikeInfo!')
    out = Parallel(n_jobs=user_settings["n_jobs"])(
        delayed(process_spike_info)(Waveform[k], SpikeTimes[k]) for k in tqdm(range(len(Waveform))))

    locations = np.zeros((n_unit, 3), dtype=np.float64)
    amp = np.zeros(n_unit, dtype=np.float64)
    channel = np.zeros(n_unit, dtype=np.int64)
    auto_corr = np.zeros((n_unit, user_settings['autoCorr']['window']), dtype=np.float64)
    isi = np.zeros((n_unit, int(user_settings['ISI']['window']/user_settings['ISI']['binwidth'])), dtype=np.float64)
    peth = np.zeros((n_unit, len(PETH[0])), dtype=np.float64)
    waveform_all = np.array(Waveform)
    channel_locations = np.column_stack((Xcoords, Ycoords))

    for k in range(n_unit):
        locations[k, :] = out[k][0:3]
        amp[k] = out[k][3]
        channel[k] = out[k][4]
        auto_corr[k, :] = out[k][5]
        isi[k, :] = out[k][6]
        peth[k, :] = PETH[k]

    # %% Save the preprocessed data
    print(f'Saving to {output_folder}...')

    np.save(os.path.join(output_folder, 'locations.npy'), locations)
    np.save(os.path.join(output_folder, 'amplitude.npy'), amp)
    np.save(os.path.join(output_folder, 'peak_channels.npy'), channel)
    np.save(os.path.join(output_folder, 'auto_corr.npy'), auto_corr)
    np.save(os.path.join(output_folder, 'isi.npy'), isi)
    np.save(os.path.join(output_folder, 'peth.npy'), peth)
    np.save(os.path.join(output_folder, 'waveform_all.npy'), waveform_all)
    np.save(os.path.join(output_folder, 'session_index.npy'), SessionIndex)
    np.save(os.path.join(output_folder,'channel_locations.npy'), channel_locations)
    print('Done!')
