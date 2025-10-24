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
from .utils import corrcoef2, Motion

def finalClustering(user_settings):
    """Final clustering of the units based on the similarity metrics using HDBSCAN and LDA.

    Arguments:
        - user_settings (dict): User settings
 
    """
    output_folder = user_settings["output_folder"]

    similarity_names = user_settings['clustering']['features']
    waveforms_corrected = np.load(os.path.join(output_folder, 'waveforms_corrected.npy'))
    motion = Motion.load(output_folder)

    iterativeClustering(user_settings, similarity_names, waveforms_corrected, motion)

def getNearbyPairs(max_distance, sessions, locations, motion=None):
    """Get the pairs of units that are within the max_distance.

    Arguments:
        - max_distance (float): The maximum distance between the units.
        - sessions (ndarray): The session index of the units.
        - locations (ndarray): The locations of the units.
        - motion (ndarray): The motion of the probe.

    Outputs:
        - idx_unit_pairs (ndarray): The pairs of units that are within the max_distance.
        - session_pairs (ndarray): The session index of the pairs of units.
    """
    n_unit = locations.shape[0]
    corrected_locations = np.zeros(n_unit)
    for k in range(n_unit):
        if motion is None:
            corrected_locations[k] = locations[k,1]
        else:
            corrected_locations[k] = locations[k,1] - motion.get_motion(sessions[k], locations[k,1])

    y_distance_matrix = np.abs(corrected_locations[:,np.newaxis] - corrected_locations[np.newaxis,:])
    idx_col = np.floor(np.arange(y_distance_matrix.size) / y_distance_matrix.shape[0]).astype(int)
    idx_row = np.mod(np.arange(y_distance_matrix.size), y_distance_matrix.shape[0]).astype(int)
    idx_good = np.where((y_distance_matrix.ravel() <= max_distance) & (idx_col > idx_row))[0]
    idx_unit_pairs = np.column_stack((idx_row[idx_good], idx_col[idx_good]))

    session_pairs = np.column_stack((sessions[idx_unit_pairs[:,0]], sessions[idx_unit_pairs[:,1]]))

    return idx_unit_pairs, session_pairs

def computeWaveformSimilarityMatrix(user_settings, waveforms, channel_locations):
    # Get waveform features
    n_nearest_channels = user_settings['waveformCorrection']['n_nearest_channels']
    n_unit = waveforms.shape[0]

    if len(waveforms.shape) > 3:
        n_templates = waveforms.shape[3]
    else:
        n_templates = 1
        waveforms = np.expand_dims(waveforms, axis=3)
    
    # find k-nearest neighbors for each channel
    nn = NearestNeighbors(n_neighbors=n_nearest_channels).fit(channel_locations)
    _, idx_nearest = nn.kneighbors(channel_locations)
    idx_nearest_sorted = np.sort(idx_nearest, axis=1)
    idx_nearest_unique, idx_groups = np.unique(idx_nearest_sorted, axis=0, return_inverse=True)

    # Compute the similarity matrix
    waveform_similarity_matrix = np.zeros((n_unit, n_unit, n_templates))

    for i_template in range(n_templates):
        waveform_similarity_matrix_this = np.zeros((n_unit, n_unit))
        ptt = np.squeeze(np.max(waveforms[:,:,:,i_template], axis=2) - np.min(waveforms[:,:,:,i_template], axis=2))
        ch = np.argmax(ptt, axis=1)

        for k in tqdm(range(idx_nearest_unique.shape[0]), desc='Computing waveform similarity'):
            idx_included = np.where(idx_groups == k)[0]
            idx_units = np.where(np.isin(ch, idx_included))[0]

            if len(idx_units) == 0:
                continue
            
            # The slice should be done twice to avoid indexing issues (different with MATLAB)
            waveform_this = waveforms[:,:,:,i_template]
            waveform_this = np.reshape(waveforms[:,idx_nearest_unique[k,:],:], (n_unit, -1))

            temp = corrcoef2(waveform_this[idx_units,:].T, waveform_this.T)
            temp[np.isnan(temp)] = 0
            temp = np.atanh(temp)
            
            waveform_similarity_matrix_this[idx_units,:] = temp
        
        waveform_similarity_matrix_this = np.max(np.stack((waveform_similarity_matrix_this, waveform_similarity_matrix_this.T), axis=2), axis=2)  
        waveform_similarity_matrix[:,:,i_template] = waveform_similarity_matrix_this

    waveform_similarity_matrix = np.max(waveform_similarity_matrix, axis=2)
    np.fill_diagonal(waveform_similarity_matrix, np.inf)

    return waveform_similarity_matrix

def computeAllSimilarityMatrix(user_settings, waveforms, feature_names):
    """Compute the similarity matrix of the units based on the similarity metrics.

    Arguments:
        - user_settings (dict): User settings
        - waveforms (ndarray): The waveforms of the units (n_units, n_channels, n_samples)
        - feature_names (list): The names of the features to be computed. The options are 'Waveform', 'ISI', 'AutoCorr', and 'PETH'.

    Outputs:
        - similarity_matrix_all (ndarray): The similarity matrix of the units (n_units, n_units, n_features)
        - feature_names_all (list): The names of the features computed.
        - waveform_similarity_matrix.npy: The waveform similarity matrix of the units (n_units, n_units)
        - ISI_similarity_matrix.npy: The ISI similarity matrix of the units (n_units, n_units)
        - AutoCorr_similarity_matrix.npy: The autocorrelogram similarity matrix of the units (n_units, n_units)
        - PETH_similarity_matrix.npy: The PETH similarity matrix of the units (n_units, n_units)
    """
    data_folder = user_settings["path_to_data"]
    output_folder = user_settings["output_folder"]
    max_similarity = 6; # r = 0.999987

    channel_locations = np.load(os.path.join(data_folder, 'channel_locations.npy'))
    isi = np.load(os.path.join(output_folder, 'isi.npy'))
    auto_corr = np.load(os.path.join(output_folder, 'auto_corr.npy'))
    peth = np.load(os.path.join(output_folder, 'peth.npy'))
    n_unit = waveforms.shape[0]

    waveform_similarity_matrix = np.zeros((n_unit, n_unit))
    if 'Waveform' in feature_names:
        waveform_similarity_matrix = computeWaveformSimilarityMatrix(user_settings, waveforms, channel_locations)
        waveform_similarity_matrix[waveform_similarity_matrix > max_similarity] = max_similarity

    ISI_similarity_matrix = np.zeros((n_unit, n_unit))
    if 'ISI' in feature_names:
        ISI_similarity_matrix = np.corrcoef(isi)
        ISI_similarity_matrix[np.isnan(ISI_similarity_matrix)] = 0
        ISI_similarity_matrix = np.atanh(ISI_similarity_matrix)
        ISI_similarity_matrix = 0.5 * (ISI_similarity_matrix + ISI_similarity_matrix.T) # make it symmetric
        np.fill_diagonal(ISI_similarity_matrix, np.inf)
        ISI_similarity_matrix[ISI_similarity_matrix > max_similarity] = max_similarity

    AutoCorr_similarity_matrix = np.zeros((n_unit, n_unit))
    if 'AutoCorr' in feature_names:
        AutoCorr_similarity_matrix = np.corrcoef(auto_corr)
        AutoCorr_similarity_matrix[np.isnan(AutoCorr_similarity_matrix)] = 0
        AutoCorr_similarity_matrix = np.atanh(AutoCorr_similarity_matrix)
        AutoCorr_similarity_matrix = 0.5 * (AutoCorr_similarity_matrix + AutoCorr_similarity_matrix.T) # make it symmetric
        np.fill_diagonal(AutoCorr_similarity_matrix, np.inf)
        AutoCorr_similarity_matrix[AutoCorr_similarity_matrix > max_similarity] = max_similarity

    PETH_similarity_matrix = np.zeros((n_unit, n_unit))
    if 'PETH' in feature_names:
        PETH_similarity_matrix = np.corrcoef(peth)
        PETH_similarity_matrix[np.isnan(PETH_similarity_matrix)] = 0
        PETH_similarity_matrix = np.atanh(PETH_similarity_matrix)
        PETH_similarity_matrix = 0.5 * (PETH_similarity_matrix + PETH_similarity_matrix.T) # make it symmetric
        np.fill_diagonal(PETH_similarity_matrix, np.inf)
        PETH_similarity_matrix[PETH_similarity_matrix > max_similarity] = max_similarity

    np.save(os.path.join(output_folder, 'waveform_similarity_matrix.npy'), waveform_similarity_matrix)
    np.save(os.path.join(output_folder, 'ISI_similarity_matrix.npy'), ISI_similarity_matrix)
    np.save(os.path.join(output_folder, 'AutoCorr_similarity_matrix.npy'), AutoCorr_similarity_matrix)
    np.save(os.path.join(output_folder, 'PETH_similarity_matrix.npy'), PETH_similarity_matrix)

    feature_names_all = ['Waveform', 'ISI', 'AutoCorr', 'PETH']
    similarity_matrix_all = np.stack((waveform_similarity_matrix, ISI_similarity_matrix, AutoCorr_similarity_matrix, PETH_similarity_matrix), axis=2)

    return (similarity_matrix_all, feature_names_all)

def iterativeClustering(user_settings, similarity_names, waveforms, motion=None):
    """Iterative clustering of the units based on the similarity metrics using HDBSCAN and LDA.
    The similarity metrics are computed firstly, and then HDBSCAN and LDA are performed alternatively to find the best clustering results.
    The clustering results are saved to the output folder.

    Arguments:
        - user_settings (dict): User settings
        - similarity_names (list): The names of the similarity metrics to be computed. The options are 'Waveform', 'ISI', 'AutoCorr', and 'PETH'.

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
    locations = np.load(os.path.join(output_folder, 'locations.npy'))

    # Compute the similarities
    max_distance = user_settings['clustering']['max_distance']
    idx_unit_pairs, _ = getNearbyPairs(max_distance, sessions, locations, motion)
    n_pairs = idx_unit_pairs.shape[0]

    similarity_matrix_all, names_all = computeAllSimilarityMatrix(user_settings, waveforms, similarity_names)
    print("Computing similarity done!")

    n_session = np.max(sessions)

    idx_names = np.array([names_all.index(name) for name in similarity_names])
    similarity_matrix_all = similarity_matrix_all[:, :, idx_names]

    similarity_all = np.zeros((n_pairs, len(similarity_names)))
    for k in range(n_pairs):
        similarity_all[k, :] = similarity_matrix_all[idx_unit_pairs[k, 0], idx_unit_pairs[k, 1], :]

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
        
        if iter < user_settings['clustering']['n_iter']:
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

    good_matches_matrix = similarity_matrix > thres
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
    np.save(os.path.join(output_folder, 'SimilarityPairs.npy'), idx_unit_pairs)
    np.save(os.path.join(output_folder, 'ClusterMatrix.npy'), hdbscan_matrix)

    np.savez(os.path.join(user_settings['output_folder'], 'ClusteringResults.npz'),
        weights=weights, 
        similarity_all=similarity_all, idx_unit_pairs=idx_unit_pairs,
        thres=thres, good_matches_matrix=good_matches_matrix,
        similarity_matrix=similarity_matrix, distance_matrix=distance_matrix,
        leafOrder=leafOrder,
        idx_cluster_hdbscan=idx_cluster_hdbscan, hdbscan_matrix=hdbscan_matrix,
        n_cluster=n_cluster)
