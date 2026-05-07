import copy
import os
import time

import numpy as np

from .AutoCuration import autoCuration
from .IterativeClustering import finalClustering
from .MotionEstimation import motionEstimation
from .Preprocess import preprocess


def runDANT(user_settings):
    """Run the standard single-shank pyDANT pipeline.

    Arguments:
        - user_settings (dict): User settings

    Outputs:
        - Output.npz and intermediate pipeline files in user_settings["output_folder"]
        - RunTimeSec.npy: Total runtime in seconds
    """
    time_start = time.time()

    preprocess(user_settings)
    motionEstimation(user_settings)
    finalClustering(user_settings)
    autoCuration(user_settings)

    run_time_sec = time.time() - time_start
    print(f"Total run time: {run_time_sec:.2f} seconds")
    np.save(os.path.join(user_settings['output_folder'], 'RunTimeSec.npy'), run_time_sec)


def runDANTMultiShank(user_settings):
    """Run pyDANT on multi-shank data by dispatching each shank separately.

    The root output folder receives global preprocessing outputs first. Then each shank is
    processed in output_folder/Shank<ID> with a unit-subset data view while preserving full
    channel geometry and channel_shanks.npy. Final outputs are merged back to the root folder.

    Arguments:
        - user_settings (dict): User settings. path_to_data must contain channel_shanks.npy.

    Outputs:
        - output_folder/Shank<ID>/: Per-shank pipeline outputs
        - output_folder/: Global preprocessing outputs plus merged clustering, curation,
          motion, similarity, and corrected waveform files in original unit order.
        - output_folder/Output.npz: Merged global output with IdxUnit and IdxShank.
        - output_folder/RunTimeSec.npy: Total runtime in seconds
    """
    data_folder = user_settings["path_to_data"]
    output_folder = user_settings["output_folder"]
    channel_shanks_path = os.path.join(data_folder, 'channel_shanks.npy')

    if not os.path.isfile(channel_shanks_path):
        raise FileNotFoundError(
            'runDANTMultiShank requires channel_shanks.npy in user_settings["path_to_data"]. '
            'For spikeInfo.mat inputs, run spikeInfo2npy(user_settings) first.'
        )

    time_start = time.time()

    preprocess(user_settings)

    unit_shanks_path = os.path.join(output_folder, 'unit_shanks.npy')
    if not os.path.isfile(unit_shanks_path):
        raise FileNotFoundError('preprocess() did not generate unit_shanks.npy!')

    unit_shanks = np.load(unit_shanks_path).ravel()
    shank_ids = np.unique(unit_shanks)

    for shank_id in shank_ids:
        idx_units = np.where(unit_shanks == shank_id)[0]
        shank_value = np.asarray(shank_id).item()
        if isinstance(shank_value, float) and shank_value.is_integer():
            shank_label = str(int(shank_value))
        else:
            shank_label = str(shank_value)

        shank_data_folder = os.path.join(output_folder, '_shank_data', f'Shank{shank_label}')
        shank_output_folder = os.path.join(output_folder, f'Shank{shank_label}')

        if not os.path.isdir(shank_data_folder):
            os.makedirs(shank_data_folder)
        if not os.path.isdir(shank_output_folder):
            os.makedirs(shank_output_folder)
        if not os.path.isdir(os.path.join(shank_output_folder, 'Figures')):
            os.makedirs(os.path.join(shank_output_folder, 'Figures'))

        np.save(
            os.path.join(shank_data_folder, 'waveform_all.npy'),
            np.load(os.path.join(data_folder, 'waveform_all.npy'))[idx_units, :, :]
        )
        np.save(
            os.path.join(shank_data_folder, 'session_index.npy'),
            np.load(os.path.join(data_folder, 'session_index.npy'))[idx_units]
        )
        np.save(
            os.path.join(shank_data_folder, 'channel_locations.npy'),
            np.load(os.path.join(data_folder, 'channel_locations.npy'))
        )
        np.save(
            os.path.join(shank_data_folder, 'channel_shanks.npy'),
            np.load(os.path.join(data_folder, 'channel_shanks.npy'))
        )
        np.save(os.path.join(shank_data_folder, 'original_unit_indices.npy'), idx_units)

        unit_files = [
            'locations.npy',
            'amplitude.npy',
            'peak_channels.npy',
            'unit_shanks.npy',
            'auto_corr.npy',
            'isi.npy',
            'peth.npy',
            'waveforms_centered.npy',
        ]

        for file_name in unit_files:
            source_path = os.path.join(output_folder, file_name)
            if not os.path.isfile(source_path):
                continue

            data = np.load(source_path, allow_pickle=True)
            if data.shape == ():
                np.save(os.path.join(shank_output_folder, file_name), data)
            else:
                np.save(os.path.join(shank_output_folder, file_name), data[idx_units])

        np.save(os.path.join(shank_output_folder, 'original_unit_indices.npy'), idx_units)

        shank_settings = copy.deepcopy(user_settings)
        shank_settings["path_to_data"] = shank_data_folder
        shank_settings["output_folder"] = shank_output_folder

        print(f'Running DANT on shank {shank_id} ({len(idx_units)} units)...')
        motionEstimation(shank_settings)
        finalClustering(shank_settings)
        autoCuration(shank_settings)

    merge_multishank_outputs(user_settings, shank_ids, unit_shanks)

    run_time_sec = time.time() - time_start
    print(f"Total run time: {run_time_sec:.2f} seconds")
    np.save(os.path.join(output_folder, 'RunTimeSec.npy'), run_time_sec)


def merge_multishank_outputs(user_settings, shank_ids, unit_shanks):
    """Merge per-shank pyDANT outputs into root-level global output files.

    This function preserves single-shank output file names in the root output folder
    while keeping rows and columns in the original global unit order. Per-shank
    matrices are written back to the global matrix positions for those units, and
    cross-shank entries are left as uncomputed zero values. Local unit indices from
    each Shank<ID> folder are remapped to original global unit indices using
    original_unit_indices.npy. Positive cluster IDs are offset across shanks, and
    -1 remains the unmatched-unit label.

    Arguments:
        - user_settings (dict): User settings
        - shank_ids (ndarray): Shank IDs processed by runDANTMultiShank()
        - unit_shanks (ndarray): Global unit-level shank IDs

    Outputs:
        Root output files matching the single-shank pipeline where applicable,
        including similarity matrices, SimilarityMatrix.npy, SimilarityPairs.npy,
        DistanceMatrix.npy, ClusterMatrix.npy, IdxCluster.npy, MatchedPairs.npy,
        Curation*.npy, ClusteringResults.npz, Output.npz, motion*.npy, and
        waveforms_corrected.npy.
    """
    data_folder = user_settings["path_to_data"]
    output_folder = user_settings["output_folder"]

    sessions = np.load(os.path.join(data_folder, 'session_index.npy'))
    locations = np.load(os.path.join(output_folder, 'locations.npy'))
    n_units = len(unit_shanks)
    n_features = len(user_settings['clustering']['features'])

    waveform_similarity_matrix = np.zeros((n_units, n_units))
    isi_similarity_matrix = np.zeros((n_units, n_units))
    auto_corr_similarity_matrix = np.zeros((n_units, n_units))
    peth_similarity_matrix = np.zeros((n_units, n_units))
    distance_matrix = np.zeros((n_units, n_units))
    hdbscan_matrix = np.zeros((n_units, n_units), dtype=bool)
    idx_cluster_hdbscan = np.full(n_units, -1, dtype=np.int64)
    leaf_order = np.empty(0, dtype=np.int64)

    output = {
        'NumClusters': 0,
        'NumUnits': n_units,
        'IdxUnit': np.arange(n_units, dtype=np.int64),
        'IdxShank': unit_shanks,
        'Locations': locations,
        'IdxSort': np.empty(0, dtype=np.int64),
        'IdxCluster': np.full(n_units, -1, dtype=np.int64),
        'SimilarityMatrix': np.zeros((n_units, n_units)),
        'SimilarityAll': np.empty((0, n_features)),
        'SimilarityPairs': np.empty((0, 2), dtype=np.int64),
        'SimilarityNames': user_settings['clustering']['features'],
        'SimilarityWeights': np.empty((0, n_features)),
        'SimilarityThreshold': np.empty(0),
        'GoodMatchesMatrix': np.zeros((n_units, n_units), dtype=bool),
        'ClusterMatrix': np.zeros((n_units, n_units), dtype=bool),
        'MatchedPairs': np.empty((0, 2), dtype=np.int64),
        'CurationPairs': np.empty((0, 2), dtype=np.int64),
        'CurationTypes': np.empty(0, dtype=np.int64),
        'CurationTypeNames': np.empty(0),
        'Params': user_settings,
        'NumSession': np.max(sessions),
        'Sessions': sessions,
        'Motion': [],
    }

    waveforms_corrected = None
    cluster_offset = 0
    pre_curation_cluster_offset = 0
    motion_linear_scale = []
    motion_linear = []
    motion_constant = []

    for shank_id in shank_ids:
        shank_value = np.asarray(shank_id).item()
        if isinstance(shank_value, float) and shank_value.is_integer():
            shank_label = str(int(shank_value))
        else:
            shank_label = str(shank_value)

        shank_output_folder = os.path.join(output_folder, f'Shank{shank_label}')
        idx_units = np.load(os.path.join(shank_output_folder, 'original_unit_indices.npy')).astype(np.int64)
        output_path = os.path.join(shank_output_folder, 'Output.npz')
        with np.load(output_path, allow_pickle=True) as output_data:
            if 'arr_0' in output_data.files:
                local_output = output_data['arr_0'].item()
            elif 'Output' in output_data.files:
                local_output = output_data['Output'].item()
            else:
                raise ValueError(f'Could not find Output in {output_path}!')

        clustering = np.load(os.path.join(shank_output_folder, 'ClusteringResults.npz'), allow_pickle=True)

        waveform_similarity_matrix[np.ix_(idx_units, idx_units)] = np.load(
            os.path.join(shank_output_folder, 'waveform_similarity_matrix.npy')
        )
        isi_similarity_matrix[np.ix_(idx_units, idx_units)] = np.load(
            os.path.join(shank_output_folder, 'ISI_similarity_matrix.npy')
        )
        auto_corr_similarity_matrix[np.ix_(idx_units, idx_units)] = np.load(
            os.path.join(shank_output_folder, 'AutoCorr_similarity_matrix.npy')
        )
        peth_similarity_matrix[np.ix_(idx_units, idx_units)] = np.load(
            os.path.join(shank_output_folder, 'PETH_similarity_matrix.npy')
        )
        distance_matrix[np.ix_(idx_units, idx_units)] = clustering['distance_matrix']

        idx_cluster_hdbscan_local = np.asarray(clustering['idx_cluster_hdbscan']).astype(np.int64)
        idx_cluster_hdbscan_global = idx_cluster_hdbscan_local.copy()
        idx_pre_matched = idx_cluster_hdbscan_global > 0
        idx_cluster_hdbscan_global[idx_pre_matched] += pre_curation_cluster_offset
        idx_cluster_hdbscan[idx_units] = idx_cluster_hdbscan_global
        hdbscan_matrix[np.ix_(idx_units, idx_units)] = clustering['hdbscan_matrix']
        leaf_order = np.concatenate((
            leaf_order,
            idx_units[np.asarray(clustering['leafOrder']).astype(np.int64)]
        ))
        pre_curation_cluster_offset += max(int(clustering['n_cluster']), 0)

        idx_cluster_local = np.asarray(local_output['IdxCluster']).astype(np.int64)
        idx_cluster_global = idx_cluster_local.copy()
        idx_matched = idx_cluster_global > 0
        idx_cluster_global[idx_matched] += cluster_offset
        output['IdxCluster'][idx_units] = idx_cluster_global

        local_num_clusters = max(int(local_output['NumClusters']), 0)
        cluster_offset += local_num_clusters

        output['SimilarityMatrix'][np.ix_(idx_units, idx_units)] = local_output['SimilarityMatrix']
        output['GoodMatchesMatrix'][np.ix_(idx_units, idx_units)] = local_output['GoodMatchesMatrix']
        output['ClusterMatrix'][np.ix_(idx_units, idx_units)] = local_output['ClusterMatrix']

        output['IdxSort'] = np.concatenate((
            output['IdxSort'],
            idx_units[np.asarray(local_output['IdxSort']).astype(np.int64)]
        ))
        output['SimilarityAll'] = np.vstack((
            output['SimilarityAll'],
            clustering['similarity_all']
        ))
        similarity_pairs = np.asarray(clustering['idx_unit_pairs'])
        if similarity_pairs.size == 0:
            similarity_pairs = np.empty((0, 2), dtype=np.int64)
        else:
            similarity_pairs = idx_units[similarity_pairs.reshape(-1, 2).astype(np.int64)]

        output['SimilarityPairs'] = np.vstack((
            output['SimilarityPairs'],
            similarity_pairs
        ))
        output['SimilarityWeights'] = np.vstack((
            output['SimilarityWeights'],
            np.asarray(clustering['weights']).reshape(1, -1)
        ))
        output['SimilarityThreshold'] = np.concatenate((
            output['SimilarityThreshold'],
            np.asarray(clustering['thres']).reshape(-1)
        ))

        matched_pairs = np.load(os.path.join(shank_output_folder, 'MatchedPairs.npy'))
        curation_pairs = np.load(os.path.join(shank_output_folder, 'CurationPairs.npy'))
        curation_types = np.load(os.path.join(shank_output_folder, 'CurationTypes.npy'))
        curation_type_names = np.load(os.path.join(shank_output_folder, 'CurationTypeNames.npy'))

        matched_pairs = np.asarray(matched_pairs)
        if matched_pairs.size == 0:
            matched_pairs = np.empty((0, 2), dtype=np.int64)
        else:
            matched_pairs = idx_units[matched_pairs.reshape(-1, 2).astype(np.int64)]

        curation_pairs = np.asarray(curation_pairs)
        if curation_pairs.size == 0:
            curation_pairs = np.empty((0, 2), dtype=np.int64)
        else:
            curation_pairs = idx_units[curation_pairs.reshape(-1, 2).astype(np.int64)]

        output['MatchedPairs'] = np.vstack((
            output['MatchedPairs'],
            matched_pairs
        ))
        output['CurationPairs'] = np.vstack((
            output['CurationPairs'],
            curation_pairs
        ))
        output['CurationTypes'] = np.concatenate((
            output['CurationTypes'],
            np.asarray(curation_types).reshape(-1)
        ))
        if output['CurationTypeNames'].size == 0:
            output['CurationTypeNames'] = curation_type_names

        output['Motion'].append({
            'Shank': shank_id,
            'LinearScale': np.load(os.path.join(shank_output_folder, 'motion_linear_scale.npy')),
            'Linear': np.load(os.path.join(shank_output_folder, 'motion_linear.npy')),
            'Constant': np.load(os.path.join(shank_output_folder, 'motion_constant.npy')),
        })
        motion_linear_scale.append(np.load(os.path.join(shank_output_folder, 'motion_linear_scale.npy')))
        motion_linear.append(np.load(os.path.join(shank_output_folder, 'motion_linear.npy')))
        motion_constant.append(np.load(os.path.join(shank_output_folder, 'motion_constant.npy')))

        waveforms_this = np.load(os.path.join(shank_output_folder, 'waveforms_corrected.npy'))
        if waveforms_corrected is None:
            waveforms_corrected = np.zeros((n_units,) + waveforms_this.shape[1:], dtype=waveforms_this.dtype)
        waveforms_corrected[idx_units, ...] = waveforms_this

    output['NumClusters'] = cluster_offset

    np.save(os.path.join(output_folder, 'waveform_similarity_matrix.npy'), waveform_similarity_matrix)
    np.save(os.path.join(output_folder, 'ISI_similarity_matrix.npy'), isi_similarity_matrix)
    np.save(os.path.join(output_folder, 'AutoCorr_similarity_matrix.npy'), auto_corr_similarity_matrix)
    np.save(os.path.join(output_folder, 'PETH_similarity_matrix.npy'), peth_similarity_matrix)
    np.save(os.path.join(output_folder, 'SimilarityMatrix.npy'), output['SimilarityMatrix'])
    np.save(os.path.join(output_folder, 'SimilarityWeights.npy'), output['SimilarityWeights'])
    np.save(os.path.join(output_folder, 'SimilarityThreshold.npy'), output['SimilarityThreshold'])
    np.save(os.path.join(output_folder, 'SimilarityPairs.npy'), output['SimilarityPairs'])
    np.save(os.path.join(output_folder, 'DistanceMatrix.npy'), distance_matrix)
    np.save(os.path.join(output_folder, 'ClusterMatrix.npy'), output['ClusterMatrix'])
    np.save(os.path.join(output_folder, 'IdxCluster.npy'), output['IdxCluster'])
    np.save(os.path.join(output_folder, 'MatchedPairs.npy'), output['MatchedPairs'])
    np.save(os.path.join(output_folder, 'CurationPairs.npy'), output['CurationPairs'])
    np.save(os.path.join(output_folder, 'CurationTypes.npy'), output['CurationTypes'])
    np.save(os.path.join(output_folder, 'CurationTypeNames.npy'), output['CurationTypeNames'])
    np.save(os.path.join(output_folder, 'motion_linear_scale.npy'), np.asarray(motion_linear_scale))
    np.save(os.path.join(output_folder, 'motion_linear.npy'), np.asarray(motion_linear))
    np.save(os.path.join(output_folder, 'motion_constant.npy'), np.asarray(motion_constant))
    np.savez(
        os.path.join(output_folder, 'ClusteringResults.npz'),
        weights=output['SimilarityWeights'],
        similarity_all=output['SimilarityAll'],
        idx_unit_pairs=output['SimilarityPairs'],
        thres=output['SimilarityThreshold'],
        good_matches_matrix=output['GoodMatchesMatrix'],
        similarity_matrix=output['SimilarityMatrix'],
        distance_matrix=distance_matrix,
        leafOrder=leaf_order,
        idx_cluster_hdbscan=idx_cluster_hdbscan,
        hdbscan_matrix=hdbscan_matrix,
        n_cluster=pre_curation_cluster_offset
    )
    np.savez(os.path.join(output_folder, 'Output.npz'), output)
    if waveforms_corrected is not None:
        np.save(os.path.join(output_folder, 'waveforms_corrected.npy'), waveforms_corrected)
