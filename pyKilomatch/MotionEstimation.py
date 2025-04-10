from tarfile import data_filter
import numpy as np
import hdbscan
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize
from joblib import Parallel, delayed
import os
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
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
    n_nearest_channels = user_settings['waveformCorrection']['n_nearest_channels']
    n_unit = np.size(waveform_all, 0)
    
    # find k-nearest neighbors for each channel
    nn = NearestNeighbors(n_neighbors=n_nearest_channels).fit(channel_locations)
    _, idx_nearest = nn.kneighbors(channel_locations)
    idx_nearest_sorted = np.sort(idx_nearest, axis=1)
    idx_nearest_unique, idx_groups = np.unique(idx_nearest_sorted, axis=0, return_inverse=True)

    # Compute the similarity matrix
    waveform_similarity_matrix = np.zeros((n_unit, n_unit))
    ptt = np.squeeze(np.max(waveform_all, axis=2) - np.min(waveform_all, axis=2))
    ch = np.argmax(ptt, axis=1)

    for k in tqdm(range(idx_nearest_unique.shape[0]), desc='Computing waveform similarity'):
        idx_included = np.where(idx_groups == k)[0]
        idx_units = np.where(np.isin(ch, idx_included))[0]

        if len(idx_units) == 0:
            continue

        waveform_this = np.reshape(waveform_all[:,idx_nearest_unique[k,:],:], (n_unit, -1))

        temp = np.corrcoef(waveform_this)
        temp[np.isnan(temp)] = 0
        temp = np.atanh(temp)
        
        waveform_similarity_matrix[idx_units,:] = temp[idx_units,:]

    waveform_similarity_matrix = np.max(np.stack((waveform_similarity_matrix, waveform_similarity_matrix.T), axis=2), axis=2)

    # Estimating the motion of the electrode
    print('---------------Motion Estimation---------------')
    max_distance = user_settings['motionEstimation']['max_distance']

    y_locations = locations[:,1]
    y_distance_matrix = np.abs(y_locations[:,np.newaxis] - y_locations[np.newaxis,:])

    idx_col = np.floor(np.arange(y_distance_matrix.size) / y_distance_matrix.shape[0]).astype(int)
    idx_row = np.mod(np.arange(y_distance_matrix.size), y_distance_matrix.shape[0]).astype(int)
    idx_good = np.where((y_distance_matrix.ravel() <= max_distance) & (idx_col > idx_row))[0]
    idx_unit_pairs = np.column_stack((idx_row[idx_good], idx_col[idx_good]))

    n_pairs = idx_unit_pairs.shape[0]

    # Compute similarity metrics
    similarity_waveform = [waveform_similarity_matrix[idx_unit_pairs[k,0], idx_unit_pairs[k,1]] for k in range(n_pairs)]

    similarity_ISI = np.zeros(n_pairs)
    ISI_similarity_matrix = np.zeros((n_unit, n_unit))
    if 'ISI' in user_settings['motionEstimation']['features']:
        ISI_similarity_matrix = np.corrcoef(isi)
        ISI_similarity_matrix[np.isnan(ISI_similarity_matrix)] = 0
        ISI_similarity_matrix = np.atanh(ISI_similarity_matrix)
        ISI_similarity_matrix = 0.5 * (ISI_similarity_matrix + ISI_similarity_matrix.T) # make it symmetric
        similarity_ISI = [ISI_similarity_matrix[idx_unit_pairs[k,0], idx_unit_pairs[k,1]] for k in range(n_pairs)]

    similarity_AutoCorr = np.zeros(n_pairs)
    AutoCorr_similarity_matrix = np.zeros((n_unit, n_unit))
    if 'AutoCorr' in user_settings['motionEstimation']['features']:
        AutoCorr_similarity_matrix = np.corrcoef(auto_corr)
        AutoCorr_similarity_matrix[np.isnan(AutoCorr_similarity_matrix)] = 0
        AutoCorr_similarity_matrix = np.atanh(AutoCorr_similarity_matrix)
        AutoCorr_similarity_matrix = 0.5 * (AutoCorr_similarity_matrix + AutoCorr_similarity_matrix.T)
        similarity_AutoCorr = [AutoCorr_similarity_matrix[idx_unit_pairs[k,0], idx_unit_pairs[k,1]] for k in range(n_pairs)]

    similarity_PETH = np.zeros(n_pairs)
    PETH_similarity_matrix = np.zeros((n_unit, n_unit))
    if 'PETH' in user_settings['motionEstimation']['features']:
        PETH_similarity_matrix = np.corrcoef(peth)
        PETH_similarity_matrix[np.isnan(PETH_similarity_matrix)] = 0
        PETH_similarity_matrix = np.atanh(PETH_similarity_matrix)
        PETH_similarity_matrix = 0.5 * (PETH_similarity_matrix + PETH_similarity_matrix.T)
        similarity_PETH = [PETH_similarity_matrix[idx_unit_pairs[k,0], idx_unit_pairs[k,1]] for k in range(n_pairs)]

    # Combine all similarity metrics
    names_all = ['Waveform', 'ISI', 'AutoCorr', 'PETH']
    similarity_all = np.column_stack((similarity_waveform, similarity_ISI, similarity_AutoCorr, similarity_PETH))
    similarity_matrix_all = np.stack((waveform_similarity_matrix, ISI_similarity_matrix, AutoCorr_similarity_matrix, PETH_similarity_matrix), axis=2)

    print(f"Computing similarity done! Saved to {os.path.join(user_settings['output_folder'], 'SimilarityForCorretion.npy')}")

    # Save the similarity
    if user_settings['save_intermediate_results']:
        np.savez(os.path.join(user_settings['output_folder'], 'SimilarityForCorretion.npz'), 
            {'similarity_waveform': similarity_waveform, 
             'waveform_similarity_matrix': waveform_similarity_matrix,
            'similarity_ISI': similarity_ISI,
            'ISI_similarity_matrix': ISI_similarity_matrix,
            'similarity_AutoCorr': similarity_AutoCorr,
            'AutoCorr_similarity_matrix': AutoCorr_similarity_matrix,
            'similarity_PETH': similarity_PETH,
            'PETH_similarity_matrix': PETH_similarity_matrix,
            'idx_unit_pairs': idx_unit_pairs})
        
    # Delete the temporary variables to save memory
    del temp
    del similarity_waveform, similarity_ISI, similarity_AutoCorr, similarity_PETH
    del waveform_similarity_matrix, ISI_similarity_matrix, AutoCorr_similarity_matrix, PETH_similarity_matrix

    # Pre-clustering
    n_session = np.max(sessions)

    similarity_names = user_settings['motionEstimation']['features']
    idx_names = [names_all.index(name) for name in similarity_names]
    similarity_all = similarity_all[:, idx_names]
    similarity_matrix_all = similarity_matrix_all[:, :, idx_names]

    weights = np.ones(len(similarity_names)) / len(similarity_names)
    similarity_matrix = np.sum(similarity_matrix_all*weights, axis=2)

    # Iterative clustering
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
                                for k in range(n_pairs)])
        
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
            similarity_matrix = np.sum(similarity_matrix_all*weights, axis=2)

    # Set the threshold based on LDA results
    similarity_thres = mdl.intercept_[0] / (-mdl.coef_[0,0]) * weights[0]

    similarity = np.sum(similarity_all * weights, axis=1)
    idx_good = np.where((similarity > similarity_thres) & (is_matched == 1))[0]
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

    # plot the motion
    plt.figure(figsize=(5, 5))
    plt.plot(np.arange(n_session)+1, positions[0,:], 'k-')
    plt.fill_between(np.arange(n_session)+1, positions_ci95[0,0,:], positions_ci95[1,0,:], color='gray', alpha=0.5)
    plt.xlabel('Sessions')
    plt.ylabel('Motion (μm)')
    plt.xlim([0.5, n_session+0.5])
    
    plt.savefig(os.path.join(user_settings['output_folder'], 'Figures/Motion.png'), dpi=300)
    plt.close()

    # Save data
    np.save(os.path.join(user_settings['output_folder'], 'motion.npy'), positions)
