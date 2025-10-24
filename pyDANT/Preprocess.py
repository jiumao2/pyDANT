import numpy as np
import os
import h5py
from scipy.ndimage import gaussian_filter1d
from joblib import Parallel, delayed
from tqdm import tqdm
from .utils import spikeLocation, computeAutoCorr
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt

def preprocess(user_settings):
    """Preprocess the data and save the features.
    Compute the ISI, autocorrelogram, and location of each unit and save the features to the output folder.

    Arguments:
        - user_settings (dict): User settings

    Outputs:
        - locations.npy: The location of each unit in the 3D space
        - amplitude.npy: The amplitude of each unit
        - peak_channels.npy: The peak channel of each unit
        - auto_corr.npy: The autocorrelogram of each unit
        - isi.npy: The ISI of each unit
        - peth.npy: The peri-event time histogram of each unit
        - waveforms_centered.npy: The centered waveforms of each unit
    
    """

    # load the data
    print('Loading data...')
    data_folder = user_settings["path_to_data"]
    waveform_all = np.load(os.path.join(data_folder, 'waveform_all.npy'))
    channel_locations = np.load(os.path.join(data_folder, 'channel_locations.npy'))
    sessions = np.load(os.path.join(data_folder, 'session_index.npy'))

    n_unit = waveform_all.shape[0]
    spike_times_all = [
        np.load(os.path.join(data_folder, 'spike_times/', 'Unit'+str(k)+'.npy')) for k in range(n_unit)]

    peth = None
    if os.path.isfile(os.path.join(data_folder, 'peth.npy')):
        peth = np.load(os.path.join(data_folder, 'peth.npy'))

    # make a folder to store the data
    output_folder = user_settings["output_folder"]
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    print(f'The output will be saved to {output_folder}!')

    # make a folder to store the figures
    figures_folder = os.path.join(output_folder, 'Figures')
    if not os.path.isdir(figures_folder):
        os.makedirs(figures_folder)

    # validate the data
    n_session = np.max(sessions)
    if n_session != len(np.unique(sessions)):
        raise ValueError('SessionIndex should start from 1 and be coninuous without any gaps!')

    print(n_session, 'sessions found!')

    # preprocessing data
    def process_spike_info(waveform, spike_times):
        # compute the location of each unit
        x, y, z, amp = spikeLocation(waveform, channel_locations,
                                    user_settings['spikeLocation']['n_nearest_channels'],
                                    user_settings['spikeLocation']['location_algorithm'])
        
        # compute the peak channel
        peaks_to_trough = np.max(waveform, axis=1) - np.min(waveform, axis=1)
        channel = np.argmax(peaks_to_trough)

        # centering the waveforms
        if user_settings['centering_waveforms']:
            n_channel = np.shape(waveform)[0]
            n_sample = np.shape(waveform)[1]
            center_position = np.ceil(n_sample/2) - 1

            # get the trough positions
            waveform_peak = np.squeeze(waveform[channel, :])
            idx_min = np.argmin(waveform_peak)

            # compute the delay
            delay = np.int64(center_position - idx_min)
            # if delay != 0:
            #     print(f'Correcting unit with delay = {delay} ...')

            # filling the borders with nearest values
            if delay == 0:
                waveforms_centered = waveform
            elif delay > 0:
                waveforms_centered = np.concatenate((waveform[:,0][:, np.newaxis] * np.ones((n_channel, delay)), np.squeeze(waveform[:,:n_sample-delay])), axis=1)
            else:
                waveforms_centered = np.concatenate((waveform[:,-delay:], waveform[:,-1][:, np.newaxis] * np.ones((n_channel, -delay))), axis=1)

            assert(np.shape(waveforms_centered) == np.shape(waveform))
        
        spike_times = spike_times - np.min(spike_times)

        # compute the autocorrelogram feauture
        auto_corr = None
        if "AutoCorr" in user_settings['motionEstimation']['features'] or \
                "AutoCorr" in user_settings['clustering']['features']:
            window = user_settings['autoCorr']['window']  # ms
            binwidth = user_settings['autoCorr']['binwidth']  # ms
            
            auto_corr, lag = computeAutoCorr(spike_times, window, binwidth)

            auto_corr[lag<0] = gaussian_filter1d(auto_corr[lag<0], user_settings['autoCorr']['gaussian_sigma'])
            auto_corr[lag>0] = gaussian_filter1d(auto_corr[lag>0], user_settings['autoCorr']['gaussian_sigma'])
            auto_corr = auto_corr / np.max(auto_corr)

        # compute the ISI feature
        isi_out = None
        if "ISI" in user_settings['motionEstimation']['features'] or \
                "ISI" in user_settings['clustering']['features']:
            isi = np.diff(spike_times)
            isi_hist = np.histogram(isi, bins=np.arange(0, user_settings['ISI']['window'], user_settings['ISI']['binwidth']))[0]
            isi_freq = isi_hist / np.sum(isi_hist)
            isi_out = gaussian_filter1d(isi_freq, user_settings['ISI']['gaussian_sigma'])
        
        if user_settings['centering_waveforms']:
            return (x,y,z,amp,channel,auto_corr,isi_out,waveforms_centered)
        else:
            return (x,y,z,amp,channel,auto_corr,isi_out,None)

    # print('Start preprocessing spikeInfo!')
    out = Parallel(n_jobs=user_settings["n_jobs"])(
        delayed(process_spike_info)(waveform_all[k], spike_times_all[k]) for k in tqdm(range(n_unit)))

    locations = np.zeros((n_unit, 3), dtype=np.float64)
    amp = np.zeros(n_unit, dtype=np.float64)
    channel = np.zeros(n_unit, dtype=np.int64)
    auto_corr = np.zeros((n_unit, np.shape(out[0][5])[0]), dtype=np.float64)
    isi = np.zeros((n_unit, int(user_settings['ISI']['window']/user_settings['ISI']['binwidth'])), dtype=np.float64)
    waveforms_centered = np.zeros((n_unit, waveform_all.shape[1], waveform_all.shape[2]), dtype=np.float64)

    for k in range(n_unit):
        locations[k, :] = out[k][0:3]
        amp[k] = out[k][3]
        channel[k] = out[k][4]
        auto_corr[k, :] = out[k][5]
        isi[k, :] = out[k][6]
        waveforms_centered[k, :, :] = out[k][7]

    # Save the preprocessed data
    print(f'Saving to {output_folder}...')

    np.save(os.path.join(output_folder, 'locations.npy'), locations)
    np.save(os.path.join(output_folder, 'amplitude.npy'), amp)
    np.save(os.path.join(output_folder, 'peak_channels.npy'), channel)
    np.save(os.path.join(output_folder, 'auto_corr.npy'), auto_corr)
    np.save(os.path.join(output_folder, 'isi.npy'), isi)
    np.save(os.path.join(output_folder, 'peth.npy'), peth)

    if user_settings['centering_waveforms']:
        np.save(os.path.join(output_folder, 'waveforms_centered.npy'), waveforms_centered)
    
    print('Done!')

    # plot the data
    plt.figure(figsize=(10, 5))
    
    # plot the number of units in each session
    plt.subplot(121)
    n_unit_session = [np.sum(sessions==k) for k in range(1, n_session+1)]
    plt.plot(range(1, n_session+1), n_unit_session, 'k.-')
    plt.xlabel('Sessions')
    plt.ylabel('Number of units')
    plt.xlim([0.5, n_session+0.5])

    # plot the locations of the units in each session
    plt.subplot(122)
    x_plot = []
    y_plot = []
    x_scale = 0.8
    for k in range(n_unit):
        x_plot.append(sessions[k] + 1 + (np.random.rand()-0.5)*x_scale)
        y_plot.append(locations[k,1])
    
    plt.plot(x_plot, y_plot, '.', markersize=1, color='k', alpha=0.5)
    plt.xlabel('Sessions')
    plt.ylabel('Y location (μm)')
    plt.title('Unit locations')
    plt.xlim([0.5, n_session+0.5])

    plt.savefig(os.path.join(figures_folder, 'unitLocations.png'), dpi=300)
    plt.close()

def spikeInfo2npy(user_settings):
    """Convert the spikeInfo.mat file from MATLAB to numpy arrays that can be used in pyDANT.

    Arguments:
        - user_settings (dict): User settings

    Outputs:
        - waveform_all.npy: The waveform of each unit
        - session_index.npy: The session index of each unit
        - channel_locations.npy: The location of each channel
        - spike_times/: A folder that contains the spike times of each unit
        - spike_times/UnitA.npy: The spike times of unit A
        - peth.npy: The peri-event time histogram of each unit

    """

    # load the data
    data_folder = user_settings["path_to_data"]
    print('Loading', os.path.join(data_folder, 'spikeInfo.mat'), '...')
    spikeInfo = h5py.File(os.path.join(data_folder, 'spikeInfo.mat'))

    # make a folder to store the data
    data_folder = user_settings["path_to_data"]

    # Preprocess the spikeInfo
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

    PETH = None
    if "PETH" in keys:
        PETH = [np.squeeze(
            np.array(
                spikeInfo[spikeInfo['spikeInfo']['PETH'][k][0]])) for k in range(n_unit)]

    # validate the data
    n_session = np.max(SessionIndex)
    if n_session != len(np.unique(SessionIndex)):
        raise ValueError('SessionIndex should start from 1 and be coninuous without any gaps!')

    print(n_session, 'sessions found!')

    locations = np.zeros((n_unit, 3), dtype=np.float64)
    amp = np.zeros(n_unit, dtype=np.float64)
    channel = np.zeros(n_unit, dtype=np.int64)
    auto_corr = np.zeros((n_unit, user_settings['autoCorr']['window']), dtype=np.float64)
    isi = np.zeros((n_unit, int(user_settings['ISI']['window']/user_settings['ISI']['binwidth'])), dtype=np.float64)
    waveform_all = np.array(Waveform)
    channel_locations = np.column_stack((Xcoords, Ycoords))

    if PETH is not None:
        peth = np.zeros((n_unit, len(PETH[0])), dtype=np.float64)
        for k in range(n_unit):
            peth[k, :] = PETH[k]
    else:
        peth = None

    # Save the preprocessed data
    print(f'Saving to {data_folder}...')

    np.save(os.path.join(data_folder, 'waveform_all.npy'), waveform_all)
    np.save(os.path.join(data_folder, 'session_index.npy'), SessionIndex)
    np.save(os.path.join(data_folder, 'channel_locations.npy'), channel_locations)

    if not os.path.isdir(os.path.join(data_folder, 'spike_times/')):
        os.makedirs(os.path.join(data_folder, 'spike_times/'))

    for k in range(n_unit):
        np.save(os.path.join(data_folder,'spike_times/', 'Unit'+str(k)+'.npy'), SpikeTimes[k])

    if PETH is not None:
        np.save(os.path.join(data_folder, 'peth.npy'), peth)

    print('Done!')