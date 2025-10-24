import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed
import os
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt
from .IterativeClustering import iterativeClustering
from .ComputeWaveformFeatures import computeWaveformFeatures
from .utils import Motion

def computeMotion(user_settings):
    """Compute the motion of the electrode and save the results.
    Compute the features of each unit and do clustering the find the matching units.
    Motion estimation is then performed to minimize the distance between the matching units.

    Arguments:
        - user_settings (dict): User settings

    Outputs:
        - motion.npy: The motion of the electrode
        - SimilarityForCorretion.npz (optional): The similarity information used for motion estimation

    """
    data_folder = user_settings["path_to_data"]
    output_folder = user_settings["output_folder"]

    sessions = np.load(os.path.join(data_folder , 'session_index.npy'))
    locations = np.load(os.path.join(output_folder, 'locations.npy'))
    similarity_matrix = np.load(os.path.join(output_folder, 'SimilarityMatrix.npy'))
    cluster_matrix = np.load(os.path.join(output_folder, 'ClusterMatrix.npy'))
    similarity_thres = np.load(os.path.join(output_folder, 'SimilarityThreshold.npy'))
    idx_unit_pairs = np.load(os.path.join(output_folder, 'SimilarityPairs.npy'))
    n_pairs = idx_unit_pairs.shape[0]
    n_session = np.max(sessions)
    n_units = similarity_matrix.shape[0]

    idx_out = idx_unit_pairs[:,0] * n_units + idx_unit_pairs[:,1] 
    good_matrix = np.logical_and(similarity_matrix > similarity_thres, cluster_matrix > 0)
    idx_good = np.where(np.logical_and(good_matrix.ravel()[idx_out] == 1, sessions[idx_unit_pairs[:,0]] != sessions[idx_unit_pairs[:,1]]))[0]

    similarity = np.zeros(n_pairs)
    for k in range(n_pairs):
        similarity[k] = similarity_matrix[idx_unit_pairs[k, 0], idx_unit_pairs[k, 1]]
    n_pairs_included = len(idx_good)

    print(f'{n_pairs_included} pairs of units are included for drift estimation!')

    # plot the similarity with threshold
    plt.figure(figsize=(5, 5))
    plt.hist(similarity, bins=100)
    plt.axvline(similarity_thres, color='red', linestyle=':', label='Threshold')
    plt.xlabel('Similarity')
    plt.ylabel('Counts')
    plt.title(str(n_pairs_included) + ' pairs are included!')

    plt.savefig(os.path.join(user_settings['output_folder'], 'Figures/SimilarityThresholdForCorrection.png'), dpi=300)
    plt.close()

    # Compute drift
    session_pairs = np.column_stack((
        [sessions[idx] for idx in idx_unit_pairs[idx_good,0]],
        [sessions[idx] for idx in idx_unit_pairs[idx_good,1]]
    ))

    # Get all the good pairs and their distance
    depth = np.zeros(len(idx_good))
    dy = np.zeros(len(idx_good))
    for k in range(len(idx_good)):
        unit1 = idx_unit_pairs[idx_good[k], 0]
        unit2 = idx_unit_pairs[idx_good[k], 1]
        dy[k] = locations[unit2,1] - locations[unit1,1]
        depth[k] = np.mean([locations[unit2,1], locations[unit1,1]])

    # Compute the motion and 95CI
    n_boot = 100
    linear_scale = 0.001

    motion = Motion(num_sessions=n_session)
    if user_settings['waveformCorrection']['linear_correction']:
        def loss_func(params):
            p_linear = np.insert(params[:n_session-1], 0, 0)  # Insert zero for the first session
            p_constant = np.insert(params[n_session-1:], 0, 0)

            p = linear_scale*p_linear[session_pairs-1]*depth[:, np.newaxis] + p_constant[session_pairs-1]

            return np.sum((np.squeeze(p[:,1] - p[:,0]) - dy)**2)

        res = minimize(loss_func, np.random.rand(n_session*2-2), 
                        options={'maxiter': 1e8})
        motion.Linear = np.insert(res.x[:n_session-1], 0, 0)
        motion.Constant = np.insert(res.x[n_session-1:], 0, 0)

        mean_motion = motion.LinearScale*motion.Linear*np.mean(depth) + motion.Constant
        motion.Constant -= np.mean(mean_motion)

        mean_motion = motion.LinearScale*motion.Linear*np.mean(depth) + motion.Constant
        min_motion = motion.LinearScale*motion.Linear*np.min(depth) + motion.Constant
        max_motion = motion.LinearScale*motion.Linear*np.max(depth) + motion.Constant
    else:
        def loss_func(params):
            p = params[session_pairs-1]

            return np.sum((np.squeeze(p[:,1] - p[:,0]) - dy)**2)
          
        res = minimize(loss_func, np.random.rand(n_session), 
                        options={'maxiter': 1e8})
        motion.Constant = res.x - np.mean(res.x)
        mean_motion = motion.Constant
        min_motion = None
        max_motion = None
    
    # Bootstrap
    def bootstrap(dy, session_pairs, n_session):
        idx_rand = np.random.randint(0, len(dy), len(dy))
        dy_this = dy[idx_rand]
        session_pairs_this = session_pairs[idx_rand, :]
        depth_this = depth[idx_rand]

        if user_settings['waveformCorrection']['linear_correction']:
            def loss_func(params):
                p_linear = np.insert(params[:n_session-1], 0, 0)  # Insert zero for the first session
                p_constant = np.insert(params[n_session-1:], 0, 0)

                p = linear_scale*p_linear[session_pairs_this-1]*depth_this[:, np.newaxis] + p_constant[session_pairs_this-1]

                return np.sum((np.squeeze(p[:,1] - p[:,0]) - dy_this)**2)
            
            res = minimize(loss_func, np.random.rand(n_session*2-2), 
                        options={'maxiter': 1e8})
            p_linear = np.insert(res.x[:n_session-1], 0, 0)
            p_constant = np.insert(res.x[n_session-1:], 0, 0)

            mean_motion_this = linear_scale*p_linear*np.mean(depth_this) + p_constant
            p_constant = p_constant - np.mean(mean_motion_this)
            mean_motion_boot = linear_scale*p_linear*np.mean(depth_this) + p_constant
            min_motion_boot = linear_scale*p_linear*np.min(depth_this) + p_constant
            max_motion_boot = linear_scale*p_linear*np.max(depth_this) + p_constant

        else:
            def loss_func(params):
                p = params[session_pairs_this-1]

                return np.sum((np.squeeze(p[:,1] - p[:,0]) - dy_this)**2)
          
            res = minimize(loss_func, np.random.rand(n_session), 
                        options={'maxiter': 1e8})
            mean_motion_boot = res.x - np.mean(res.x)
            min_motion_boot = None
            max_motion_boot = None

        return mean_motion_boot, min_motion_boot, max_motion_boot

    p_boot = Parallel(n_jobs=user_settings["n_jobs"])(delayed(bootstrap)(dy, session_pairs, n_session) 
        for _ in tqdm(range(n_boot), desc='Computing 95CI'))
    
    mean_motion_ci95 = np.zeros((2, n_session))
    min_motion_ci95 = np.zeros((2, n_session))
    max_motion_ci95 = np.zeros((2, n_session))
    for j in range(n_session):
        mean_motion_ci95[0,j] = np.percentile([p_boot[i][0][j] for i in range(n_boot)], 2.5)
        mean_motion_ci95[1,j] = np.percentile([p_boot[i][0][j] for i in range(n_boot)], 97.5)

        if user_settings['waveformCorrection']['linear_correction']:
            min_motion_ci95[0,j] = np.percentile([p_boot[i][1][j] for i in range(n_boot)], 2.5)
            min_motion_ci95[1,j] = np.percentile([p_boot[i][1][j] for i in range(n_boot)], 97.5)
            max_motion_ci95[0,j] = np.percentile([p_boot[i][2][j] for i in range(n_boot)], 2.5)
            max_motion_ci95[1,j] = np.percentile([p_boot[i][2][j] for i in range(n_boot)], 97.5)

    # plot the motion
    plt.figure(figsize=(5, 5))

    plt.plot(np.arange(n_session)+1, mean_motion, 'k-', label='Mean Motion')
    plt.fill_between(np.arange(n_session)+1, mean_motion_ci95[0,:], mean_motion_ci95[1,:], color='gray', alpha=0.5)

    if min_motion is not None:
        plt.plot(np.arange(n_session)+1, min_motion, 'b-', label='Min Motion')
        plt.fill_between(np.arange(n_session)+1, min_motion_ci95[0,:], min_motion_ci95[1,:], color='blue', alpha=0.5)

    if max_motion is not None:
        plt.plot(np.arange(n_session)+1, max_motion, 'r-', label='Max Motion')
        plt.fill_between(np.arange(n_session)+1, max_motion_ci95[0,:], max_motion_ci95[1,:], color='red', alpha=0.5)

    plt.xlabel('Sessions')
    plt.ylabel('Motion (μm)')
    plt.xlim([0.5, n_session+0.5])
    
    plt.savefig(os.path.join(user_settings['output_folder'], 'Figures/Motion.png'), dpi=300)
    plt.close()

    # Save Motion
    motion.save(user_settings['output_folder'])

    return motion


def motionEstimation(user_settings):
    """Estimate the motion of the electrode and save the results.
    Compute the features of each unit and do clustering the find the matching units.
    Motion estimation is then performed to minimize the distance between the matching units.

    Arguments:
        - user_settings (dict): User settings

    Outputs:
        - motion.npy: The motion of the electrode
        - SimilarityForCorretion.npz (optional): The similarity information used for motion estimation

    """
    data_folder = user_settings["path_to_data"]
    output_folder = user_settings["output_folder"]

    if user_settings['centering_waveforms']:
        waveform_all = np.load(os.path.join(output_folder, 'waveforms_centered.npy'))
    else:
        waveform_all = np.load(os.path.join(data_folder , 'waveform_all.npy'))

    similarity_names_all = user_settings['motionEstimation']['features']
    n_iter_motion_estimation = len(similarity_names_all)
    
    for i in range(n_iter_motion_estimation):
        if i == 0:
            iterativeClustering(user_settings, similarity_names_all[i], waveform_all)
        else:
            waveforms_corrected = np.load(os.path.join(output_folder, 'waveforms_corrected.npy'))
            iterativeClustering(user_settings, similarity_names_all[i], waveforms_corrected, Motion)

        Motion = computeMotion(user_settings)
        computeWaveformFeatures(user_settings, waveform_all, Motion)
