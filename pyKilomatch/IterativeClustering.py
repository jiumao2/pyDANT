import numpy as np
import os
from tqdm import tqdm
import hdbscan
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import optimal_leaf_ordering, leaves_list
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt

def iterativeClustering(user_settings):
    """Iterative clustering of the units based on the similarity metrics using HDBSCAN and LDA.
    The similarity metrics are computed firstly, and then HDBSCAN and LDA are performed alternatively to find the best clustering results.
    The clustering results are saved to the output folder.

    Arguments:
        - user_settings (dict): User settings

    Outputs:
        - SimilarityMatrix.npy: The similarity matrix of the units
        - SimilarityWeights.npy: The weights of the similarity metrics
        - SimilarityThreshold.npy: The threshold of the similarity metrics from LDA
        - ClusteringResults.npz: The clustering results of the units
        - DistanceMatrix.npy: The distance matrix used for HDBSCAN
        - waveform_similarity_matrix.npy: The waveform similarity matrix of the units
        - ISI_similarity_matrix.npy: The ISI similarity matrix of the units
        - AutoCorr_similarity_matrix.npy: The autocorrelogram similarity matrix of the units
        - PETH_similarity_matrix.npy: The PETH similarity matrix of the units
        - AllSimilarity.npz (optional): The similarity metrics of all units used for clustering
 
    """

    # Load precomputed features
    data_folder = user_settings["path_to_data"]
    output_folder = user_settings["output_folder"]

    sessions = np.load(os.path.join(data_folder , 'session_index.npy'))
    channel_locations = np.load(os.path.join(data_folder, 'channel_locations.npy'))

    isi = np.load(os.path.join(output_folder, 'isi.npy'))
    auto_corr = np.load(os.path.join(output_folder, 'auto_corr.npy'))
    peth = np.load(os.path.join(output_folder, 'peth.npy'))
    waveforms_corrected = np.load(os.path.join(output_folder, 'waveforms_corrected.npy'))
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

    # Get waveform features
    n_nearest_channels = user_settings['waveformCorrection']['n_nearest_channels']
    n_unit = np.size(waveforms_corrected, 0)
    
    # find k-nearest neighbors for each channel
    nn = NearestNeighbors(n_neighbors=n_nearest_channels).fit(channel_locations)
    _, idx_nearest = nn.kneighbors(channel_locations)
    idx_nearest_sorted = np.sort(idx_nearest, axis=1)
    idx_nearest_unique, idx_groups = np.unique(idx_nearest_sorted, axis=0, return_inverse=True)

    # Compute the similarity matrix
    waveform_similarity_matrix = np.zeros((n_unit, n_unit))
    ptt = np.squeeze(np.max(waveforms_corrected, axis=2) - np.min(waveforms_corrected, axis=2))
    ch = np.argmax(ptt, axis=1)

    for k in tqdm(range(idx_nearest_unique.shape[0]), desc='Computing waveform similarity'):
        idx_included = np.where(idx_groups == k)[0]
        idx_units = np.where(np.isin(ch, idx_included))[0]

        if len(idx_units) == 0:
            continue

        waveform_this = np.reshape(waveforms_corrected[:,idx_nearest_unique[k,:],:], (n_unit, -1))

        temp = np.corrcoef(waveform_this)
        temp[np.isnan(temp)] = 0
        temp = np.atanh(temp)
        
        waveform_similarity_matrix[idx_units,:] = temp[idx_units,:]

    waveform_similarity_matrix = np.max(np.stack((waveform_similarity_matrix, waveform_similarity_matrix.T), axis=2), axis=2)  

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
        AutoCorr_similarity_matrix = 0.5 * (AutoCorr_similarity_matrix + AutoCorr_similarity_matrix.T) # make it symmetric
        similarity_AutoCorr = [AutoCorr_similarity_matrix[idx_unit_pairs[k,0], idx_unit_pairs[k,1]] for k in range(n_pairs)]

    similarity_PETH = np.zeros(n_pairs)
    PETH_similarity_matrix = np.zeros((n_unit, n_unit))
    if 'PETH' in user_settings['motionEstimation']['features']:
        PETH_similarity_matrix = np.corrcoef(peth)
        PETH_similarity_matrix[np.isnan(PETH_similarity_matrix)] = 0
        PETH_similarity_matrix = np.atanh(PETH_similarity_matrix)
        PETH_similarity_matrix = 0.5 * (PETH_similarity_matrix + PETH_similarity_matrix.T) # make it symmetric
        similarity_PETH = [PETH_similarity_matrix[idx_unit_pairs[k,0], idx_unit_pairs[k,1]] for k in range(n_pairs)]

    np.save(os.path.join(output_folder, 'waveform_similarity_matrix.npy'), waveform_similarity_matrix)
    np.save(os.path.join(output_folder, 'ISI_similarity_matrix.npy'), ISI_similarity_matrix)
    np.save(os.path.join(output_folder, 'AutoCorr_similarity_matrix.npy'), AutoCorr_similarity_matrix)
    np.save(os.path.join(output_folder, 'PETH_similarity_matrix.npy'), PETH_similarity_matrix)

    # Combine all similarity metrics
    names_all = ['Waveform', 'ISI', 'AutoCorr', 'PETH']
    similarity_all = np.column_stack((similarity_waveform, similarity_ISI, similarity_AutoCorr, similarity_PETH))   
    similarity_matrix_all = np.stack((waveform_similarity_matrix, ISI_similarity_matrix, AutoCorr_similarity_matrix, PETH_similarity_matrix), axis=2)

    print(f"Computing similarity done! Saved to {os.path.join(user_settings['output_folder'], 'AllSimilarity.npy')}")

    # Save results
    if user_settings['save_intermediate_results']:
        np.savez(os.path.join(user_settings['output_folder'], 'AllSimilarity.npz'), 
            {'similarity_waveform': similarity_waveform,
             'waveform_similarity_matrix': waveform_similarity_matrix,
            'similarity_ISI': similarity_ISI,
            'ISI_similarity_matrix': ISI_similarity_matrix,
            'similarity_AutoCorr': similarity_AutoCorr,
            'AutoCorr_similarity_matrix': AutoCorr_similarity_matrix,
            'similarity_PETH': similarity_PETH,
            'PETH_similarity_matrix': PETH_similarity_matrix,
            'idx_unit_pairs': idx_unit_pairs,
            'session_pairs': session_pairs})
        
    # Delete the temporary variables to save memory
    del temp
    del similarity_waveform, similarity_ISI, similarity_AutoCorr, similarity_PETH
    del waveform_similarity_matrix, ISI_similarity_matrix, AutoCorr_similarity_matrix, PETH_similarity_matrix

    n_session = np.max(sessions)

    similarity_names = user_settings['clustering']['features']
    idx_names = np.zeros(len(similarity_names), dtype=int)
    for k in range(len(similarity_names)):
        idx_names[k] = names_all.index(similarity_names[k])
    similarity_all = similarity_all[:, idx_names]
    similarity_matrix_all = similarity_matrix_all[:, :, idx_names]

    weights = np.ones(len(similarity_names))/len(similarity_names)
    similarity_matrix = np.sum(similarity_matrix_all*weights, axis=2)

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

    Z = clusterer.single_linkage_tree_.to_numpy()
    np.save(os.path.join(output_folder, 'DistanceMatrix.npy'), distance_matrix)
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

    # plot the distribution of similarity
    n_plots = len(similarity_names)
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
    for k in range(n_plots):
        axes[k].hist(similarity_all[:, k], bins=50, color='blue', density=True)
        axes[k].set_title(similarity_names[k])
        axes[k].set_xlabel(similarity_names[k] + ' Similarity')
        axes[k].set_ylabel('Density')

    plt.savefig(os.path.join(output_folder, 'Figures/AllSimilarity.png'))
    plt.close()

    # Save the results
    np.save(os.path.join(output_folder, 'SimilarityMatrix.npy'), similarity_matrix)
    np.save(os.path.join(output_folder, 'SimilarityWeights.npy'), weights)
    np.save(os.path.join(output_folder, 'SimilarityThreshold.npy'), thres)

    np.savez(os.path.join(user_settings['output_folder'], 'ClusteringResults.npz'),
        weights=weights, 
        similarity_all=similarity_all, idx_unit_pairs=idx_unit_pairs,
        thres=thres, good_matches_matrix=good_matches_matrix,
        similarity_matrix=similarity_matrix, distance_matrix=distance_matrix,
        leafOrder=leafOrder,
        idx_cluster_hdbscan=idx_cluster_hdbscan, hdbscan_matrix=hdbscan_matrix,
        n_cluster=n_cluster)
