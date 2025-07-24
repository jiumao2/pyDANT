Input and output
=================

.. contents:: 
    :local:

Input
-------

Data
+++++

See how to prepare the data :ref:`here <prepare_the_data_label>`.

``settings.json``
+++++++++++++++++++++

See how to prepare it :ref:`here <prepare_the_data_label>`.

See how to adjust the settings :doc:`here <Change_default_settings>`.


Output
-------------

The output will be saved in the ``output_folder`` specified in ``settings.json``, which are mostly .npy and .npz files.

Results
+++++++++++++

This part is the main output of Kilomatch, which contains the estimated cluster IDs for each unit across sessions.

``IdxCluster.npy`` assigns a unique cluster ID for each unit (-1 for non-matched units). You can use it to extract the matched units across sessions. 

``ClusterMatrix.npy`` is a cluster assignment matrix. ``ClusterMatrix(i,j) = 1`` means unit ``i`` and ``j`` are in the same cluster.

``MatchedPairs.npy`` contains the unit index for all matched pairs.

``RunTimeSec.npy`` contains the total run time in seconds.

``Output.npz`` is a compressed file that contains all the above files and some additional information, such as the number of units, sessions, and clusters. It is designed to mimic the ``Output.mat`` file which is the main output of the MATLAB version of Kilomatch.



Features
++++++++++

The feature-related files include ``auto_corr.npy``, ``isi.npy``, ``peth.npy``. These files store the computed features for each unit, which are used in the motion estimation and clustering processes.


Waveforms
++++++++++++

If you choose to center the waveforms, the centered waveforms will be saved in ``waveforms_corrected.npy``. And the corrected waveforms will be saved in ``waveforms_corrected.npy``.


Spike location
+++++++++++++++

The estimated locations for each unit will be saved in ``locations.npy``, with the amplitudes and the peak channels saved in ``amplitude.npy`` and ``peak_channels.npy``, respectively.


Similarity matrix
++++++++++++++++++++

Four similarity matrices will be saved to ``waveform_similarity_matrix.npy``, ``ISI_similarity_matrix.npy``, ``AutoCorr_similarity_matrix.npy`` and ``PETH_similarity_matrix.npy``. If some features are not used, the corresponding matrices will be filled with zeros. The final weighted similarity matrix will be saved in ``SimilarityMatrix.npy``.


Motion correction
++++++++++++++++++++++

The estimated probe motion will be saved in ``motion_linear_scale.npy``, ``motion_linear.npy`` and ``motion_constant.npy``, which can be loaded via ``Motion.load()``. If rigid motion correction is used, only ``motion_constant.npy`` is meaningful. See :ref:`here <non_rigid_correction_label>` for details about non-rigid motion correction.


Clustering
++++++++++++++

The clustering-related files include ``DistanceMatrix.npy``, ``SimilarityPairs.npy``, ``SimilarityThreshold.npy`` and ``SimilarityWeights.npy``. These files store the intermediate results of the final clustering process after motion correction. The ``DistanceMatrix.npy`` contains the pairwise distances between units. The ``SimilarityPairs.npy`` contains the unit index for each pair of units within the ``max_distance``. See :ref:`Clustering <Clustering>` for details about the meaning of these files.

``ClusteringResults.npz`` is a compressed file that contains the results of the final clustering process after motion correction, which includes more details about the clustering process, like the clustering results before curation.


Curation
++++++++++++

The curation-related information will be saved in ``CurationTypeNames.npy``, ``CurationTypes.npy``, and ``CurationPairs.npy``. It stores the deleted unit pairs and why they are deleted (types). See :doc:`curation <Auto_curation>` for details.






