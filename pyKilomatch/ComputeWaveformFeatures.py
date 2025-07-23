import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import os
from .utils import waveformEstimation
import copy

def computeWaveformFeatures(user_settings, waveform_all, motion):
    """ Compute the corrected waveforms based on the motion of the probe.
    The corrected waveforms on the reference probe are computed using the Kriging interpolation method
    and saved to the output folder.

    Arguments:
        - user_settings (dict): User settings
        - waveform_all (numpy.ndarray): The waveforms of all units (n_unit, n_channel, n_sample)
        - motion (Motion): The motion object containing the linear and constant parameters for correction
    Outputs:
        - waveforms_corrected.npy: The corrected waveforms.
    
    """

    data_folder = user_settings["path_to_data"]
    output_folder = user_settings["output_folder"]
    n_templates = user_settings["waveformCorrection"]["n_templates"]

    channel_locations = np.load(os.path.join(data_folder, 'channel_locations.npy'))
    sessions = np.load(os.path.join(data_folder , 'session_index.npy'))

    locations = np.load(os.path.join(output_folder, 'locations.npy'))

    n_sample = waveform_all.shape[2]
    n_channel = waveform_all.shape[1]
    n_unit = waveform_all.shape[0]

    min_channel_depth = np.min(channel_locations[:,1])
    max_channel_depth = np.max(channel_locations[:,1])

    motion_bottom = motion.LinearScale*motion.Linear*min_channel_depth + motion.Constant
    motion_top = motion.LinearScale*motion.Linear*max_channel_depth + motion.Constant

    min_motion = np.min(np.concatenate((motion_bottom, motion_top)))
    max_motion = np.max(np.concatenate((motion_bottom, motion_top)))
    print('The range of motion: [%.1f μm ~ %.1f μm]\n' % (min_motion, max_motion))

    def process_spike(locations_this, motion, channel_locations, waveform_this, session_this, n_templates, min_motion, max_motion):

        motion_this = copy.deepcopy(motion)
        waveforms_corrected = np.zeros((n_channel, n_sample, n_templates))

        for k in range(n_templates):
            if n_templates == 2:
                if k == 0:
                    motion_this.Constant = motion_this.Constant - min_motion
                else:
                    motion_this.Constant = motion_this.Constant - max_motion

            dy = motion_this.get_motion(session_this, locations_this[1])
            location_new = locations_this.copy()
            location_new[1] -= dy

            waveforms_corrected[:,:,k] = waveformEstimation(
                waveform_this, locations_this, channel_locations, location_new)
            
        return waveforms_corrected

    # Run parallel processing with progress bar
    out = Parallel(n_jobs=user_settings["n_jobs"])(
        delayed(process_spike)(locations[k,:2], motion, channel_locations, waveform_all[k,:,:], sessions[k], n_templates, min_motion, max_motion) 
        for k in tqdm(range(n_unit), desc='Computing waveform features')
    )

    waveforms_corrected = np.zeros((n_unit, n_channel, n_sample, n_templates))
    for k in range(n_unit):
        waveforms_corrected[k,:,:,:] = out[k]

    # Save the corrected waveforms
    output_folder = user_settings['output_folder']
    np.save(os.path.join(output_folder, 'waveforms_corrected.npy'), waveforms_corrected)
