from tarfile import data_filter
import numpy as np
import hdbscan
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from joblib import Parallel, delayed
import os
from tqdm import tqdm
from .utils import computeSimilarity, waveformSimilarity
import matplotlib.pyplot as plt

def motionEstimation(user_settings):
    # Get all features
    data_folder = user_settings["path_to_data"]
    output_folder = user_settings["output_folder"]
    
    waveform_all = np.load(os.path.join(data_folder , 'waveform_all.npy'))
    channel_locations = np.load(os.path.join(data_folder, 'channel_locations.npy'))
    sessions = np.load(os.path.join(data_folder , 'session_index.npy'))
    
    isi = np.load(os.path.join(output_folder, 'isi.npy'))
    auto_corr = np.load(os.path.join(output_folder, 'auto_corr.npy'))
    peth = np.load(os.path.join(output_folder, 'peth.npy'))
    locations = np.load(os.path.join(output_folder, 'locations.npy'))
    

    # Get waveform features
    n_nearest_channels = user_settings['waveformCorrection']['n_channels_precomputed']
    n_unit = np.size(waveform_all, 0)
    n_channel = np.size(waveform_all, 1)
    n_sample = np.size(waveform_all, 2)

    waveforms = np.zeros((n_unit, n_nearest_channels, n_sample))
    waveform_channels = np.zeros((n_unit, n_nearest_channels))

    # Compute waveform features in parallel
    def process_spike(locations_this):
        distance_to_location = np.sqrt(np.sum((channel_locations - locations_this)**2, axis=1))
        idx_sort = np.argsort(distance_to_location)
        idx_included = idx_sort[:n_nearest_channels]

        return idx_included

    out = Parallel(n_jobs=user_settings["n_jobs"])(delayed(process_spike)(locations[k,:2]) for k in tqdm(range(n_unit), 
        desc='Computing waveform features'))

    for k in range(n_unit):
        waveform_channels[k,:] = out[k]
        waveforms[k,:,:] = waveform_all[k,out[k],:]

    # %% Estimating the motion of the electrode
    print('---------------Motion Estimation---------------')
    max_distance = user_settings['motionEstimation']['max_motion_distance']

    y_locations = locations[:,1]
    y_distance_matrix = np.abs(y_locations[:,np.newaxis] - y_locations[np.newaxis,:])

    idx_col = np.floor(np.arange(y_distance_matrix.size) / y_distance_matrix.shape[0]).astype(int)
    idx_row = np.mod(np.arange(y_distance_matrix.size), y_distance_matrix.shape[0]).astype(int)
    idx_good = np.where((y_distance_matrix.ravel() <= max_distance) & (idx_col > idx_row))[0]
    idx_unit_pairs = np.column_stack((idx_row[idx_good], idx_col[idx_good]))

    n_pairs = idx_unit_pairs.shape[0]

    # Compute similarity metrics
    similarity_waveform = np.zeros(n_pairs)
    if 'Waveform' in user_settings['motionEstimation']['features']:
        out = Parallel(n_jobs=user_settings["n_jobs"])(delayed(waveformSimilarity)(
            waveforms[idx_unit_pairs[k,:],:,:], waveform_channels[idx_unit_pairs[k,:],:], user_settings['waveformCorrection']['n_nearest_channels']) 
            for k in tqdm(range(n_pairs), desc='Computing waveform similarity'))
        
        similarity_waveform = np.array(out)

    similarity_ISI = np.zeros(n_pairs)
    if 'ISI' in user_settings['motionEstimation']['features']:
        out = Parallel(n_jobs=user_settings["n_jobs"])(delayed(computeSimilarity)(
            isi[idx_unit_pairs[k,0],:], isi[idx_unit_pairs[k,1],:]) for k in tqdm(range(n_pairs), desc='Computing ISI similarity'))

        similarity_ISI = np.array(out)

    similarity_AutoCorr = np.zeros(n_pairs)
    if 'AutoCorr' in user_settings['motionEstimation']['features']:
        out = Parallel(n_jobs=user_settings["n_jobs"])(delayed(computeSimilarity)(
            auto_corr[idx_unit_pairs[k,0],:], auto_corr[idx_unit_pairs[k,1],:]) for k in tqdm(range(n_pairs), desc='Computing AutoCorr similarity')) 
        
        similarity_AutoCorr = np.array(out)

    similarity_PETH = np.zeros(n_pairs)
    if 'PETH' in user_settings['motionEstimation']['features']:
        out = Parallel(n_jobs=user_settings["n_jobs"])(delayed(computeSimilarity)(
            peth[idx_unit_pairs[k,0],:], peth[idx_unit_pairs[k,1],:]) for k in tqdm(range(n_pairs), desc='Computing PETH similarity'))  
        
        similarity_PETH = np.array(out)

    print(f"Computing similarity done! Saved to {os.path.join(user_settings['output_folder'], 'SimilarityForCorretion.npy')}")

    # Save the similarity
    if user_settings['save_intermediate_results']:
        np.save(os.path.join(user_settings['output_folder'], 'SimilarityForCorretion.npy'), 
            {'similarity_waveform': similarity_waveform, 
                'similarity_ISI': similarity_ISI,
                'similarity_AutoCorr': similarity_AutoCorr,
                'similarity_PETH': similarity_PETH,
                'idx_unit_pairs': idx_unit_pairs})

    # Pre-clustering
    n_session = np.max(sessions)
    names_all = ['Waveform', 'ISI', 'AutoCorr', 'PETH']
    similarity_all = np.column_stack((similarity_waveform, similarity_ISI, 
                                    similarity_AutoCorr, similarity_PETH))

    similarity_names = user_settings['motionEstimation']['features']
    idx_names = [names_all.index(name) for name in similarity_names]
    similarity_all = similarity_all[:, idx_names]

    weights = np.ones(len(similarity_names)) / len(similarity_names)
    mean_similarity = np.sum(similarity_all * weights, axis=1)

    similarity_matrix = np.zeros((n_unit, n_unit))
    for k in range(idx_unit_pairs.shape[0]):
        similarity_matrix[idx_unit_pairs[k,0], idx_unit_pairs[k,1]] = mean_similarity[k]
        similarity_matrix[idx_unit_pairs[k,1], idx_unit_pairs[k,0]] = mean_similarity[k]
    np.fill_diagonal(similarity_matrix, 5)

    # %% Iterative clustering
    for iter in range(user_settings['motionEstimation']['n_iter']):
        print(f'Iteration {iter+1} starts!')
        
        # HDBSCAN
        distance_matrix = 1.0 / (1.0 + np.tanh(similarity_matrix))
        np.fill_diagonal(distance_matrix, 0)
        
        clusterer = hdbscan.HDBSCAN(
            min_samples=1,
            cluster_selection_epsilon=0,
            min_cluster_size=2,
            max_cluster_size=n_session,
            metric='precomputed'
        )
        
        idx_cluster_hdbscan = clusterer.fit_predict(distance_matrix)
        idx_cluster_hdbscan[idx_cluster_hdbscan >= 0] += 1  # MATLAB starts from 1
        
        n_cluster = np.max(idx_cluster_hdbscan)
        hdbscan_matrix = np.zeros_like(similarity_matrix, dtype=bool)
        
        for k in range(1, n_cluster+1):
            idx = np.where(idx_cluster_hdbscan == k)[0]
            for i in range(len(idx)):
                for j in range(i+1, len(idx)):
                    hdbscan_matrix[idx[i], idx[j]] = True
                    hdbscan_matrix[idx[j], idx[i]] = True
        
        np.fill_diagonal(hdbscan_matrix, True)
        
        is_matched = np.array([hdbscan_matrix[idx_unit_pairs[k,0], idx_unit_pairs[k,1]] 
                                for k in range(len(mean_similarity))])
        
        if iter != user_settings['motionEstimation']['n_iter'] - 1:
            # LDA and update weights
            mdl = LinearDiscriminantAnalysis()
            mdl.fit(similarity_all, is_matched)
            
            temp = mdl.coef_[0]
            weights = temp / np.sum(temp)
            print('Weights:')
            print('   '.join(similarity_names))
            print(weights)
            
            # Update the similarity matrix
            mean_similarity = np.sum(similarity_all * weights, axis=1)
            similarity_matrix = np.zeros((n_unit, n_unit))
            
            for k in range(idx_unit_pairs.shape[0]):
                similarity_matrix[idx_unit_pairs[k,0], idx_unit_pairs[k,1]] = mean_similarity[k]
                similarity_matrix[idx_unit_pairs[k,1], idx_unit_pairs[k,0]] = mean_similarity[k]
            
            np.fill_diagonal(similarity_matrix, 5)

    # Set the threshold based on LDA results
    similarity_thres = mdl.intercept_[0] / (-mdl.coef_[0,0]) * weights[0]

    similarity = np.sum(similarity_all * weights, axis=1)
    idx_good = np.where((similarity > similarity_thres) & (is_matched == 1))[0]
    n_pairs_included = len(idx_good)

    print(f'{n_pairs_included} pairs of units are included for drift estimation!')

    # %% Compute drift
    # nblock = user_settings['motionEstimation']['n_block']
    nblock = 1
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

    depth_edges = np.linspace(np.min(depth), np.max(depth), nblock+1)
    depth_bins = 0.5*(depth_edges[:-1] + depth_edges[1:])
    idx_block = np.zeros(len(idx_good), dtype=int)

    # Compute the motion and 95CI
    n_boot = 100
    positions = np.full((nblock, n_session), np.nan)
    positions_ci95 = np.zeros((2, nblock, n_session))

    for k in range(nblock):
        dy_block = dy[idx_block == k]
        idx_1_block = idx_1[idx_block == k]
        idx_2_block = idx_2[idx_block == k]
        
        if len(np.unique(np.concatenate((idx_1_block, idx_2_block)))) != n_session:
            print('Some sessions are not included! Motion estimation failed!')
            continue
        
        def loss_func(y):
            return np.sum((dy_block - (y[idx_2_block-1] - y[idx_1_block-1]))**2)
        
        from scipy.optimize import minimize
        res = minimize(loss_func, np.random.rand(n_session), 
                        options={'maxiter': 1e8})
        p = res.x - np.mean(res.x)
        positions[k,:] = p
        
        # Bootstrap
        def bootstrap(dy_block, idx_1_block, idx_2_block, n_session):
            idx_rand = np.random.randint(0, len(dy_block), len(dy_block))
            dy_this = dy_block[idx_rand]
            idx_1_this = idx_1_block[idx_rand]
            idx_2_this = idx_2_block[idx_rand]
            
            def loss_func_boot(y):
                return np.sum((dy_this - (y[idx_2_this-1] - y[idx_1_this-1]))**2)
            
            res_boot = minimize(loss_func_boot, np.random.rand(n_session), 
                                options={'maxiter': 1e8})
            return res_boot.x - np.mean(res_boot.x)
        
        p_boot = Parallel(n_jobs=user_settings["n_jobs"])(delayed(bootstrap)(dy_block, idx_1_block, idx_2_block, n_session) 
            for j in tqdm(range(n_boot), desc='Computing 95CI'))
        
        p_ci95 = np.zeros((2, n_session))
        for j in range(n_session):
            p_ci95[0,j] = np.percentile([p[j] for p in p_boot], 2.5)
            p_ci95[1,j] = np.percentile([p[j] for p in p_boot], 97.5)
        
        positions_ci95[:,k,:] = p_ci95

    print(f'{np.sum(~np.isnan(positions[:,0]))} / {nblock} blocks are available!')

    # Interpolate the motion with nearest value if some blocks are not sampled
    for k in range(nblock):
        if np.all(np.isnan(positions[k,:])):
            distance = np.abs(depth_bins - depth_bins[k])
            idx_sort = np.argsort(distance)
            
            idx1 = np.nan
            d1 = np.nan
            
            for j in range(len(idx_sort)):
                if np.all(np.isnan(positions[idx_sort[j], :])):
                    continue
                
                if np.isnan(d1):
                    d1 = distance[idx_sort[j]]
                    idx1 = idx_sort[j]
                elif d1 == distance[idx_sort[j]]:
                    positions[k,:] = np.mean(positions[[idx1, idx_sort[j]], :], axis=0)
                else:
                    positions[k,:] = positions[idx1, :]

    # Save data
    np.save(os.path.join(user_settings['output_folder'], 'motion.npy'), positions)
