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
    idx_good = np.where(good_matrix.ravel()[idx_out] == 1)[0]
    print(idx_good)

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
    idx_1 = np.zeros(len(idx_good), dtype=int)
    idx_2 = np.zeros(len(idx_good), dtype=int)

    for k in range(len(idx_good)):
        unit1 = idx_unit_pairs[idx_good[k], 0]
        unit2 = idx_unit_pairs[idx_good[k], 1]
        d_this = np.mean([locations[unit2,1], locations[unit1,1]])
        
        idx_1[k] = session_pairs[k,0]
        idx_2[k] = session_pairs[k,1]
        dy[k] = locations[unit2,1] - locations[unit1,1]
        depth[k] = d_this

    # Compute the motion and 95CI
    n_boot = 100
    positions = np.zeros(n_session)
    positions_ci95 = np.zeros((2, n_session))
    
    if len(np.unique(np.concatenate((idx_1, idx_2)))) != n_session:
        print('Some sessions are not included! Motion estimation failed!')
    
    def loss_func(y):
        return np.sum((dy - (y[idx_2-1] - y[idx_1-1]))**2)
    
    res = minimize(loss_func, np.random.rand(n_session), 
                    options={'maxiter': 1e8})
    positions = res.x - np.mean(res.x)
    
    # Bootstrap
    def bootstrap(dy, idx_1, idx_2, n_session):
        idx_rand = np.random.randint(0, len(dy), len(dy))
        dy_this = dy[idx_rand]
        idx_1_this = idx_1[idx_rand]
        idx_2_this = idx_2[idx_rand]
        
        def loss_func_boot(y):
            return np.sum((dy_this - (y[idx_2_this-1] - y[idx_1_this-1]))**2)
        
        res_boot = minimize(loss_func_boot, np.random.rand(n_session), 
                            options={'maxiter': 1e8})
        return res_boot.x - np.mean(res_boot.x)
    
    p_boot = Parallel(n_jobs=user_settings["n_jobs"])(delayed(bootstrap)(dy, idx_1, idx_2, n_session) 
        for j in tqdm(range(n_boot), desc='Computing 95CI'))
    
    positions_ci95 = np.zeros((2, n_session))
    for j in range(n_session):
        positions_ci95[0,j] = np.percentile([p[j] for p in p_boot], 2.5)
        positions_ci95[1,j] = np.percentile([p[j] for p in p_boot], 97.5)

    # plot the motion
    plt.figure(figsize=(5, 5))
    plt.plot(np.arange(n_session)+1, positions, 'k-')
    plt.fill_between(np.arange(n_session)+1, positions_ci95[0,:], positions_ci95[1,:], color='gray', alpha=0.5)
    plt.xlabel('Sessions')
    plt.ylabel('Motion (μm)')
    plt.xlim([0.5, n_session+0.5])
    
    plt.savefig(os.path.join(user_settings['output_folder'], 'Figures/Motion.png'), dpi=300)
    plt.close()

    # Save data
    np.save(os.path.join(user_settings['output_folder'], 'motion.npy'), positions)    

    return positions
    

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

    waveform_all = np.load(os.path.join(data_folder , 'waveform_all.npy'))

    similarity_names_all = user_settings['motionEstimation']['features']
    n_iter_motion_estimation = len(similarity_names_all)
    
    for i in range(n_iter_motion_estimation):
        if i == 0:
            iterativeClustering(user_settings, similarity_names_all[i], waveform_all)
        else:
            waveforms_corrected = np.load(os.path.join(output_folder, 'waveforms_corrected.npy'))
            iterativeClustering(user_settings, similarity_names_all[i], waveforms_corrected, positions)

        positions = computeMotion(user_settings)
        computeWaveformFeatures(user_settings, waveform_all)
