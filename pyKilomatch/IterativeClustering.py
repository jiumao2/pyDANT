import numpy as np
from joblib import Parallel, delayed
import os
from tqdm import tqdm
from .utils import computeSimilarity, waveformSimilarity
import hdbscan
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.spatial.distance import squareform

def iterativeClustering(user_settings):
    # Load precomputed features
    data_folder = user_settings["path_to_data"]
    output_folder = user_settings["output_folder"]

    sessions = np.load(os.path.join(data_folder , 'session_index.npy'))

    isi = np.load(os.path.join(output_folder, 'isi.npy'))
    auto_corr = np.load(os.path.join(output_folder, 'auto_corr.npy'))
    peth = np.load(os.path.join(output_folder, 'peth.npy'))
    waveforms = np.load(os.path.join(output_folder, 'waveforms.npy'))
    waveforms_corrected = np.load(os.path.join(output_folder, 'waveforms_corrected.npy'))
    waveform_channels = np.load(os.path.join(output_folder, 'waveform_channels.npy'))
    locations = np.load(os.path.join(output_folder, 'locations.npy'))
    positions = np.load(os.path.join(output_folder, 'Motion.npy'))

    # Recompute the similarities
    max_distance = user_settings['clustering']['max_distance']
    n_unit = waveforms_corrected.shape[0]

    corrected_locations = np.zeros(n_unit)
    for k in range(n_unit):
        corrected_locations[k] = locations[k,1] - positions[0, sessions[k]-1]

    y_distance_matrix = np.abs(corrected_locations[:,np.newaxis] - corrected_locations[np.newaxis,:])

    idx_col = np.floor(np.arange(y_distance_matrix.size) / y_distance_matrix.shape[0]).astype(int)
    idx_row = np.mod(np.arange(y_distance_matrix.size), y_distance_matrix.shape[0]).astype(int)
    idx_good = np.where((y_distance_matrix.ravel() <= max_distance) & (idx_col > idx_row))[0]
    idx_unit_pairs = np.column_stack((idx_row[idx_good], idx_col[idx_good]))

    session_pairs = np.column_stack((sessions[idx_unit_pairs[:,0]], sessions[idx_unit_pairs[:,1]]))
    n_pairs = idx_unit_pairs.shape[0]

    # Clear temp variables
    del corrected_locations, y_distance_matrix, idx_row, idx_col, idx_good

    # %% Initialize similarity arrays
    similarity_waveform = np.zeros(n_pairs)
    similarity_raw_waveform = np.zeros(n_pairs)
    if 'Waveform' in user_settings['clustering']['features']:
        out = Parallel(n_jobs=user_settings["n_jobs"])(delayed(waveformSimilarity)(
            waveforms[idx_unit_pairs[k,:],:,:], waveform_channels[idx_unit_pairs[k,:],:], user_settings['waveformCorrection']['n_nearest_channels']) 
            for k in tqdm(range(n_pairs), desc='Computing raw waveform similarity'))
        
        similarity_raw_waveform = np.array(out)

        out = Parallel(n_jobs=user_settings["n_jobs"])(delayed(waveformSimilarity)(
            waveforms_corrected[idx_unit_pairs[k,:],:,:], waveform_channels[idx_unit_pairs[k,:],:], user_settings['waveformCorrection']['n_nearest_channels']) 
            for k in tqdm(range(n_pairs), desc='Computing corrected waveform similarity'))
        
        similarity_waveform = np.array(out)

    similarity_ISI = np.zeros(n_pairs)
    if 'ISI' in user_settings['clustering']['features']:
        out = Parallel(n_jobs=user_settings["n_jobs"])(delayed(computeSimilarity)(
            isi[idx_unit_pairs[k,0],:], isi[idx_unit_pairs[k,1],:]) for k in tqdm(range(n_pairs), desc='Computing ISI similarity'))

        similarity_ISI = np.array(out)

    similarity_AutoCorr = np.zeros(n_pairs)
    if 'AutoCorr' in user_settings['clustering']['features']:
        out = Parallel(n_jobs=user_settings["n_jobs"])(delayed(computeSimilarity)(
            auto_corr[idx_unit_pairs[k,0],:], auto_corr[idx_unit_pairs[k,1],:]) for k in tqdm(range(n_pairs), desc='Computing AutoCorr similarity'))

        similarity_AutoCorr = np.array(out)

    similarity_PETH = np.zeros(n_pairs)
    if 'PETH' in user_settings['clustering']['features']:
        out = Parallel(n_jobs=user_settings["n_jobs"])(delayed(computeSimilarity)(
            peth[idx_unit_pairs[k,0],:], peth[idx_unit_pairs[k,1],:]) for k in tqdm(range(n_pairs), desc='Computing PETH similarity'))  
        
        similarity_PETH = np.array(out)

    print(f"Computing similarity done! Saved to {os.path.join(user_settings['output_folder'], 'AllSimilarity.npy')}")

    # Save results
    if user_settings['save_intermediate_results']:
        np.save(os.path.join(user_settings['output_folder'], 'AllSimilarity.npy'), 
            {'similarity_waveform': similarity_waveform,
            'similarity_raw_waveform': similarity_raw_waveform,
            'similarity_ISI': similarity_ISI,
            'similarity_AutoCorr': similarity_AutoCorr,
            'similarity_PETH': similarity_PETH,
            'idx_unit_pairs': idx_unit_pairs,
            'session_pairs': session_pairs})

    n_session = np.max(sessions)

    names_all = ['Waveform', 'ISI', 'AutoCorr', 'PETH']
    similarity_all = np.column_stack((similarity_waveform, similarity_ISI, similarity_AutoCorr, similarity_PETH))

    similarity_names = user_settings['clustering']['features']
    idx_names = np.zeros(len(similarity_names), dtype=int)
    for k in range(len(similarity_names)):
        idx_names[k] = names_all.index(similarity_names[k])
    similarity_all = similarity_all[:, idx_names]

    weights = np.ones(len(similarity_names))/len(similarity_names)
    mean_similarity = np.sum(similarity_all * weights, axis=1)

    similarity_matrix = np.zeros((n_unit, n_unit))
    for k in range(idx_unit_pairs.shape[0]):
        similarity_matrix[idx_unit_pairs[k,0], idx_unit_pairs[k,1]] = mean_similarity[k]
        similarity_matrix[idx_unit_pairs[k,1], idx_unit_pairs[k,0]] = mean_similarity[k]
    np.fill_diagonal(similarity_matrix, 5)

    for iter in range(1, user_settings['clustering']['n_iter']+1):
        print(f'Iteration {iter} starts!')

        # HDBSCAN
        distance_matrix = 1./(1 + np.tanh(similarity_matrix))
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

    Z = clusterer.single_linkage_tree_.to_numpy()
    from scipy.cluster.hierarchy import optimal_leaf_ordering, leaves_list
    Z_ordered = optimal_leaf_ordering(Z, squareform(distance_matrix))
    leafOrder = leaves_list(Z_ordered)

    # set the threshold based on LDA results
    thres = mdl.intercept_[0] / (-mdl.coef_[0][0]) * weights[0]

    similarity = np.sum(similarity_all * weights, axis=1)
    good_matches_matrix = np.zeros_like(similarity_matrix, dtype=bool)
    idx_good_matches = np.where(similarity > thres)[0]
    for k in idx_good_matches:
        good_matches_matrix[idx_unit_pairs[k, 0], idx_unit_pairs[k, 1]] = True
        good_matches_matrix[idx_unit_pairs[k, 1], idx_unit_pairs[k, 0]] = True
    np.fill_diagonal(good_matches_matrix, True)

    # Save the results
    np.save(os.path.join(output_folder, 'SimilarityMatrix.npy'), similarity_matrix)
    np.save(os.path.join(output_folder, 'SimilarityWeights.npy'), weights)
    np.save(os.path.join(output_folder, 'SimilarityThreshold.npy'), thres)

    np.savez(os.path.join(user_settings['output_folder'], 'ClusteringResults.npz'),
        weights=weights, 
        similarity_all=similarity_all,
        thres=thres, good_matches_matrix=good_matches_matrix,
        similarity_matrix=similarity_matrix, distance_matrix=distance_matrix,
        leafOrder=leafOrder,
        idx_cluster_hdbscan=idx_cluster_hdbscan, hdbscan_matrix=hdbscan_matrix,
        n_cluster=n_cluster)
