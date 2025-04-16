import numpy as np
import os
from scipy.sparse.csgraph import connected_components
from .utils import graphEditNumber
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt

def autoCuration(user_settings):
    """Automatic curation of clustering results.
    Perform automatic curation of clustering results based on user settings.
    This function loads precomputed features, applies auto-splitting and auto-merging of clusters,
    and saves the curated results.

    Arguments:
        - user_settings (dict): User settings

    Outputs:
        - ClusterMatrix.npy: The connectivity matrix of clusters after curation.
        - IdxCluster.npy: The cluster index of each unit after curation. -1 indicates unpaired units.
        - MatchedPairs.npy: The matched pairs of units after curation.
        - Output.npz (optional): A dictionary containing other information about the final results.
    """

    # Load precomputed features
    data_folder = user_settings["path_to_data"]
    output_folder = user_settings["output_folder"]

    sessions = np.load(os.path.join(data_folder , 'session_index.npy'))

    clustering_result = np.load(os.path.join(output_folder, 'ClusteringResults.npz'))
    idx_cluster_hdbscan = clustering_result['idx_cluster_hdbscan']
    good_matches_matrix = clustering_result['good_matches_matrix']
    hdbscan_matrix = clustering_result['hdbscan_matrix']
    similarity_matrix = clustering_result['similarity_matrix']
    leafOrder = clustering_result['leafOrder']

    # Initialize parameters
    reject_thres = user_settings['autoCuration']['reject_threshold']
    n_cluster = np.max(idx_cluster_hdbscan)
    print(f'{n_cluster} clusters and {int((np.sum(hdbscan_matrix) - hdbscan_matrix.shape[0])/2)} pairs before removing bad units!')

    hdbscan_matrix_raw = hdbscan_matrix.copy()

    # Split clusters if there's a clear boundary in good_matches_matrix
    if user_settings['autoCuration']['auto_split']:
        n_cluster_new = n_cluster
        for k in range(1, n_cluster+1):
            units = np.where(idx_cluster_hdbscan == k)[0]
            graph_this = good_matches_matrix[np.ix_(units, units)]
            n_sub_clusters, idx_sub_clusters = connected_components(graph_this, directed=False)
            
            if n_sub_clusters <= 1:
                continue
                
            for j in range(2, n_sub_clusters+1):
                units_this = units[idx_sub_clusters == j-1]
                idx_cluster_hdbscan[units_this] = n_cluster_new + j - 1
                
            n_cluster_new += n_sub_clusters - 1
        
        n_cluster = n_cluster_new

    # Remove bad units in clusters
    for k in range(1, n_cluster+1):
        units = np.where(idx_cluster_hdbscan == k)[0]
        sessions_this = sessions[units]
        similarity_matrix_this = similarity_matrix[np.ix_(units, units)]
        
        while len(sessions_this) != len(np.unique(sessions_this)) or np.any(similarity_matrix_this < reject_thres):
            idx_remove = []
            
            for j in range(len(sessions_this)):
                for i in range(j+1, len(sessions_this)):
                    if sessions_this[i] == sessions_this[j] or similarity_matrix_this[i,j] < reject_thres:
                        similarity_i = np.mean(similarity_matrix_this[i,:])
                        similarity_j = np.mean(similarity_matrix_this[j,:])
                        
                        if similarity_i <= similarity_j:
                            idx_remove.append(i)
                        else:
                            idx_remove.append(j)
            
            idx_remove = np.unique(idx_remove)
            idx_cluster_hdbscan[units[idx_remove]] = -1
            units = np.delete(units, idx_remove)
            sessions_this = sessions[units]
            similarity_matrix_this = similarity_matrix[np.ix_(units, units)]

    # Update clusters and hdbscan matrix
    idx_remove = []
    for k in range(1, n_cluster+1):
        units = np.where(idx_cluster_hdbscan == k)[0]
        if len(units) <= 1:
            idx_cluster_hdbscan[units] = -1
            idx_remove.append(k)

    for k in sorted(idx_remove, reverse=True):
        idx_cluster_hdbscan[idx_cluster_hdbscan >= k] -= 1

    assert len(np.unique(idx_cluster_hdbscan)) == np.max(idx_cluster_hdbscan)+1
    n_cluster = np.max(idx_cluster_hdbscan)

    # Update hdbscan matrix
    hdbscan_matrix = np.zeros_like(similarity_matrix, dtype=bool)
    for k in range(1, n_cluster+1):
        idx = np.where(idx_cluster_hdbscan == k)[0]
        for j in range(len(idx)):
            for i in range(j+1, len(idx)):
                hdbscan_matrix[idx[j], idx[i]] = True
                hdbscan_matrix[idx[i], idx[j]] = True

    np.fill_diagonal(hdbscan_matrix, True)

    num_same, num_before, num_after = graphEditNumber(hdbscan_matrix_raw, hdbscan_matrix)
    assert num_same == num_after

    print(f'{num_before-num_after} deleting steps are done!')
    print(f'{n_cluster} clusters and {int((np.sum(hdbscan_matrix) - hdbscan_matrix.shape[0])/2)} pairs after removing bad units!')

    # merge when two or more adjacent clusters are similar and do not contain units from the same sessions
    if user_settings['autoCuration']['auto_merge']:
        # compute the location of each clusters
        hdbscan_matrix_raw = hdbscan_matrix.copy()
        cluster_centers = np.zeros(n_cluster)
        for k in range(1, n_cluster+1):
            units = np.where(idx_cluster_hdbscan == k)[0]
            temp = np.array([np.where(leafOrder == x)[0][0] for x in units])
            cluster_centers[k-1] = np.median(temp)
        cluster_id_sorted = np.argsort(cluster_centers)
        
        flag = True
        while flag:
            similar_pairs = []  # idA, idB, similarity
            flag = False
            for k in range(len(cluster_id_sorted)-1):
                id = cluster_id_sorted[k]
                units = np.where(idx_cluster_hdbscan == id)[0]
                sessions_this = sessions[units]
                
                id_next = cluster_id_sorted[k+1]
                units_next = np.where(idx_cluster_hdbscan == id_next)[0]
                sessions_next = sessions[units_next]
                
                if len(np.intersect1d(sessions_this, sessions_next)) > 0:
                    continue
                    
                if np.any(similarity_matrix[np.ix_(units, units_next)] < reject_thres):
                    continue
                    
                if not np.any(good_matches_matrix[np.ix_(units, units_next)] > 0):
                    continue
                    
                similar_pairs.append([k, k+1, np.median(similarity_matrix[np.ix_(units, units_next)])])
            
            if len(similar_pairs) > 0:
                print(f'Found {sum([1 for x in similar_pairs if x[2] > reject_thres])} possible merges!')
        
            # merging
            for k in range(len(similar_pairs)):
                if similar_pairs[k][2] <= reject_thres:
                    continue
                
                if k < len(similar_pairs)-1 and \
                        similar_pairs[k+1][0] == similar_pairs[k][1] and \
                        similar_pairs[k+1][2] > similar_pairs[k][2]:
                    continue
                    
                flag = True
        
                id = cluster_id_sorted[similar_pairs[k][0]]
                id_next = cluster_id_sorted[similar_pairs[k][1]]
                idx_cluster_hdbscan[idx_cluster_hdbscan == id_next] = id
            
            # update cluster info
            max_id = np.max(idx_cluster_hdbscan)
            for k in range(max_id, 0, -1):
                units = np.where(idx_cluster_hdbscan == k)[0]
                if len(units) == 0:
                    idx_cluster_hdbscan[idx_cluster_hdbscan >= k] -= 1
            
            n_cluster = np.max(idx_cluster_hdbscan)
            assert len(np.unique(idx_cluster_hdbscan)) == n_cluster+1
        
            # update cluster centers
            cluster_centers = np.zeros(n_cluster)
            for k in range(1, n_cluster+1):
                units = np.where(idx_cluster_hdbscan == k)[0]
                temp = np.array([np.where(leafOrder == x)[0][0] for x in units])
                cluster_centers[k-1] = np.median(temp)
            cluster_id_sorted = np.argsort(cluster_centers)
        
        # update hdbscan matrix
        hdbscan_matrix = np.zeros_like(similarity_matrix, dtype=bool)
        for k in range(1, n_cluster+1):
            idx = np.where(idx_cluster_hdbscan == k)[0]
            for j in range(len(idx)):
                for i in range(j+1, len(idx)):
                    hdbscan_matrix[idx[j], idx[i]] = True
                    hdbscan_matrix[idx[i], idx[j]] = True
        np.fill_diagonal(hdbscan_matrix, True)
        
        num_same, num_before, num_after = graphEditNumber(hdbscan_matrix_raw, hdbscan_matrix)
        assert num_same == num_before
        
        print(f'{-num_before+num_after} merging steps are done!')
        print(f'{n_cluster} clusters and {int((np.sum(hdbscan_matrix) - hdbscan_matrix.shape[0])/2)} pairs after merging good clusters!')
        
        # find possible pairings for unpaired units
        print('Checking the unpaired units!')
        
        count_merges = 0
        
        n_cluster_new = n_cluster
        flag_match = True
        
        while flag_match:
            flag_match = False
            idx_unpaired = np.where(idx_cluster_hdbscan == -1)[0]
            for k in range(len(idx_unpaired)):
                unit = idx_unpaired[k]
                session_this = sessions[unit]
            
                idx_match = np.where(good_matches_matrix[unit,:] == 1)[0]
        
                if len(idx_match) == 0:
                    continue
                
                temp = np.argmax(similarity_matrix[k, idx_match])
                idx_match = idx_match[temp]
                idx_cluster_new = idx_cluster_hdbscan[idx_match]
                
                if idx_cluster_new == -1:
                    continue
                if np.any(session_this == sessions[idx_cluster_hdbscan == idx_cluster_new]):
                    continue
                
                flag_match = True
                count_merges += 1
                idx_cluster_hdbscan[unit] = idx_cluster_new
        
        n_cluster = n_cluster_new
        print(f'Merged {count_merges} unpaired units!')
        
        # update hdbscan matrix
        hdbscan_matrix = np.zeros_like(similarity_matrix, dtype=bool)
        for k in range(1, n_cluster+1):
            idx = np.where(idx_cluster_hdbscan == k)[0]
            for j in range(len(idx)):
                for i in range(j+1, len(idx)):
                    hdbscan_matrix[idx[j], idx[i]] = True
                    hdbscan_matrix[idx[i], idx[j]] = True
        np.fill_diagonal(hdbscan_matrix, True)
        
        print(f'{n_cluster} clusters and {int((np.sum(hdbscan_matrix) - hdbscan_matrix.shape[0])/2)} pairs after merging good unpaired units!')
    


    # Save curated results
    hdbscan_matrix_curated = hdbscan_matrix.copy()
    idx_cluster_hdbscan_curated = idx_cluster_hdbscan.copy()

    # Get all matched pairs
    matched_pairs_curated = []
    for k in range(len(hdbscan_matrix_curated)):
        for j in range(k+1, len(hdbscan_matrix_curated)):
            if hdbscan_matrix_curated[k,j]:
                matched_pairs_curated.append([k, j])

    matched_pairs_curated = np.array(matched_pairs_curated)

    if user_settings['save_intermediate_results']:
        np.savez(os.path.join(output_folder, 'CurationResults.npz'),
                hdbscan_matrix_curated=hdbscan_matrix_curated,
                idx_cluster_hdbscan_curated=idx_cluster_hdbscan_curated,
                matched_pairs_curated=matched_pairs_curated,
                similarity_matrix=similarity_matrix,
                sessions=sessions,
                leafOrder=leafOrder)

    # Save final output
    np.save(os.path.join(output_folder, 'ClusterMatrix.npy'), hdbscan_matrix_curated)
    np.save(os.path.join(output_folder, 'IdxCluster.npy'), idx_cluster_hdbscan_curated)
    np.save(os.path.join(output_folder, 'MatchedPairs.npy'), matched_pairs_curated)

    Output = {
        'NumClusters': np.max(idx_cluster_hdbscan_curated),
        'NumUnits': len(idx_cluster_hdbscan_curated),
        'IdxSort': leafOrder,
        'IdxCluster': idx_cluster_hdbscan_curated,
        'SimilarityMatrix': similarity_matrix,
        'GoodMatchesMatrix': good_matches_matrix,
        'ClusterMatrix': hdbscan_matrix_curated,
        'MatchedPairs': matched_pairs_curated,
        'Params': user_settings,
        'NumSession': np.max(sessions),
        'Sessions': sessions
    }

    np.savez(os.path.join(output_folder, 'Output.npz'), Output)
    print(f'Kilomatch done! Output saved to {os.path.join(output_folder, "Output.npz")}')
    print(f'Found {Output["NumClusters"]} clusters and {len(Output["MatchedPairs"])} matches from {Output["NumUnits"]} units during {Output["NumSession"]} sessions!')


    # Plot the results

    # probability of matching between sessions
    n_session = np.max(sessions)
    n_cluster = np.max(idx_cluster_hdbscan)
    n_matched_matrix = np.zeros((n_session, n_session))

    n_units_each_session = np.array(
        [np.sum(sessions == i) for i in range(1, n_session+1)]
    )

    for k in range(1, n_cluster+1):
        units = np.where(idx_cluster_hdbscan == k)[0]
        for j in range(len(units)):
            for i in range(j+1, len(units)):
                n_matched_matrix[sessions[units[j]]-1, sessions[units[i]]-1] += 1
                n_matched_matrix[sessions[units[i]]-1, sessions[units[j]]-1] += 1

    d_session = np.arange(-n_session+1, n_session)
    p_matched = [[] for _ in range(len(d_session))]
    p_matched_matrix = np.zeros((n_session, n_session))
    for k in range(1, n_session+1):
        for j in range(k+1, n_session+1):   
            p_matched_matrix[k-1,j-1] = n_matched_matrix[k-1,j-1]/n_units_each_session[k-1]
            p_matched_matrix[j-1,k-1] = n_matched_matrix[k-1,j-1]/n_units_each_session[j-1]
            
            idx_this = np.where(d_session == j-k)[0]
            assert len(idx_this) == 1, f"Error: {j-k} not in d_session"
            p_matched[idx_this[0]].append(n_matched_matrix[k-1,j-1]/n_units_each_session[j-1])

            idx_this = np.where(d_session == k-j)[0]
            assert len(idx_this) == 1, f"Error: {j-k} not in d_session"
            p_matched[idx_this[0]].append(n_matched_matrix[k-1,j-1]/n_units_each_session[k-1])

    p_matches_mean = np.array([np.mean(p_matched[k]) for k in range(len(d_session))])
    p_matches_std = np.array([np.std(p_matched[k]) for k in range(len(d_session))])

    p_matches_mean[d_session == 0] = np.nan
    p_matches_std[d_session == 0] = np.nan

    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    plt.imshow(p_matched_matrix, cmap='plasma')
    plt.gca().invert_yaxis()
    plt.colorbar(label='Prob. of matched')
    plt.xlabel('Sessions')
    plt.ylabel('Sessions')

    plt.subplot(122)
    plt.plot(d_session, p_matches_mean, 'k.-')
    plt.xlabel('Δ session')
    plt.ylabel('Prob. of matched')

    plt.savefig(os.path.join(output_folder, 'Figures/MatchedProbability.png'), dpi=300)
    plt.close()

    # Plot the similarity distribution and the weights of each feature
    similarity_all = clustering_result['similarity_all']
    idx_unit_pairs = clustering_result['idx_unit_pairs']
    weights = clustering_result['weights']
    similarity_names = user_settings['clustering']['features']

    n_pairs = similarity_all.shape[0]
    n_features = similarity_all.shape[1]

    mean_similarity = similarity_all @ weights
    is_matched = np.array([hdbscan_matrix_curated[idx_unit_pairs[k,0], idx_unit_pairs[k,1]] for k in range(n_pairs)])

    n_plots = n_features+1
    plt.figure(figsize=(4*n_plots + 1, 4))

    plt.subplot(1, n_plots, 1)
    plt.hist(mean_similarity[~is_matched], bins=50, color='k', density=True)
    plt.hist(mean_similarity[is_matched], bins=50, color='b', density=True)
    plt.xlabel('Mean similarity')
    plt.ylabel('Density')
    plt.title('Weighted summation')

    for k in range(n_features):
        plt.subplot(1, n_plots, k+2)
        plt.hist(similarity_all[~is_matched,k], bins=50, color='k', density=True)
        plt.hist(similarity_all[is_matched,k], bins=50, color='b', density=True)
        plt.xlabel(similarity_names[k])
        plt.title('weight = ' + f'{weights[k]:.2f}')

    plt.savefig(os.path.join(output_folder, 'Figures/SimilarityDistribution.png'), dpi=300)
    plt.close()

    # Plot the scatters between any two features
    n_points_max = 5000
    idx_matched_rnd = np.where(is_matched)[0]
    idx_unmatched_rnd = np.where(~is_matched)[0]

    if np.sum(is_matched) > n_points_max:
        idx_matched_rnd = np.random.choice(np.where(is_matched)[0], n_points_max, replace=False)

    if np.sum(~is_matched) > n_points_max:
        idx_unmatched_rnd = np.random.choice(np.where(~is_matched)[0], n_points_max, replace=False)
    
    n_plots = n_features*(n_features-1)//2

    plt.figure(figsize=(4*n_plots + 1, 4))
    count = 0
    for k in range(n_features):
        for j in range(k+1, n_features):
            count += 1
            plt.subplot(1, n_plots, count)
            plt.plot(similarity_all[idx_unmatched_rnd,k], similarity_all[idx_unmatched_rnd,j], 'k.', markersize=1, alpha=0.3)
            plt.plot(similarity_all[idx_matched_rnd,k], similarity_all[idx_matched_rnd,j], 'b.', markersize=1, alpha=0.3)
            
            plt.xlabel(similarity_names[k])
            plt.ylabel(similarity_names[j])

    plt.savefig(os.path.join(output_folder, 'Figures/FeatureScatter.png'), dpi=300)
    plt.close()



