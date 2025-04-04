# pyKilomatch

[![View pyKilomatch on GitHub](https://img.shields.io/badge/GitHub-Kilomatch-blue.svg)](https://github.com/jiumao2/pyKilomatch)

This project is a Python implementation of [Kilomatch](https://github.com/jiumao2/Kilomatch), converted from the original MATLAB code.

## Installation

- It is recommended to install the pyKilomatch package using Anaconda:

```shell
conda create -n pyKilomatch python=3.11
conda activate pyKilomatch
pip install pyKilomatch
```  

## How to use it

### Prepare the data

- Your data should be organized in a folder with the following structure:
```shell
data_folder
├── channel_locations.npy
├── waveform_all.npy
├── session_index.npy
├── peth.npy (optional)
└── spike_times/
    ├── Unit0.npy
    ├── Unit1.npy
    ├── Unit2.npy
    └── ...
    └── UnitN.npy
```  

- The data files should adhere to the following formats:
    - `session_index.npy`: An array of length `n_unit` indicating the session for each unit. Session indices should start from 1 and be continuous without gaps.
    - `waveform_all.npy`: An `n_unit` x `n_channel` x `n_sample` tensor containing the mean waveform of each unit in μV. All units must share the same set of channels.
    - `channel_locations.npy`: An `n_channel` x 2 double array specifying the (x, y) coordinates (in μm) of each channel. The y-coordinate typically represents the depth.
    - `peth.npy`: (Optional but recommended) An `n_unit` x `n_point` double array containing the peri-event time histogram for each unit.
    - `spike_times/UnitX.npy`: An array of spike times (in milliseconds) for unit `X`. The filenames should follow the pattern `UnitX.npy`, where `X` is the unit index starting from 0 and incrementing continuously without gaps.

- Specify the path to your data in the `settings.json` file.
- Edit the `settings.json` file to set the `path_to_data` and `output_folder`.
- Adjust other parameters within `settings.json` to match your specific data characteristics.

### Running the Code

- After configuring the `settings.json` file and ensuring the path to it is correctly specified in `mainKilomatch.py`, execute the following commands in your terminal:

```shell
conda activate pyKilomatch
python mainKilomatch.py
```

### About the output

- All temporary files, results, and generated figures will be saved in the `output_folder` defined in your `settings.json` file.
- The `output_folder` will contain the following result files:
    - `auto_corr.npy`: An `n_unit` x `n_point` double array containing the auto-correlation of each unit.
    - `isi.npy`: An `n_unit` x `n_point` double array containing the inter-spike interval (ISI) of each unit.
    - `peth.npy`: An `n_unit` x `n_point` double array containing the peri-event time histogram (PETH) of each unit.
    - `waveforms.npy`: An `n_unit` x `n_nearest_channel` x `n_sample` tensor of the mean waveform for each unit (in μV) before motion correction.
    - `waveforms_corrected.npy`: An `n_unit` x `n_nearest_channel` x `n_sample` tensor of the mean waveform for each unit (in μV) after motion correction.
    - `waveform_channels.npy`: An `n_unit` x `n_nearest_channel` integer array indicating the channel index for each unit's waveform.
    - `locations.npy`: An `n_unit` x 3 double array containing the estimated x, y, and z coordinates of each unit.
    - `IdxCluster.npy`: An `n_unit` x 1 integer array assigning a cluster index to each unit. Units not assigned to any cluster are marked with `-1`.
    - `ClusterMatrix.npy`: An `n_unit` x `n_unit` logical matrix representing cluster assignments. `ClusterMatrix(i, j) = 1` indicates that unit `i` and unit `j` belong to the same cluster.
    - `MatchedPairs.npy`: An `n_pairs` x 2 integer matrix listing the unit indices for each pair of units within the same cluster.
    - `SimilarityWeights.npy`: An `n_features` x 1 double array containing the weights of the similarity metrics computed by the IHDBSCAN algorithm.
    - `SimilarityThreshold.npy`: A 1 x 1 double value representing the threshold used to identify good matches in the `GoodMatchesMatrix` during the auto-curation process.
    - `SimilarityMatrix.npy`: An `n_unit` x `n_unit` double matrix representing the weighted sum of similarities between each pair of units.
    - `motion.npy`: A `n_session` x 1 double array indicating the estimated electrode positions for each session.
    - `ClusteringResults.npz`: A compressed file containing the clustering results after motion correction.
    - `CurationResults.npz`: A compressed file containing the curation results after motion correction.
    - `Output.npz`: A compressed file containing the final results of the Kilomatch analysis.


## Notes

- pyKilomatch is a conversion from MATLAB and is currently undergoing testing. Please report any bugs or issues you encounter.
- For recordings using multi-shank probes, it is recommended to analyze each shank independently. The current Python version of Kilomatch is not yet designed to handle multi-shank data simultaneously.
- Be careful that the waveforms included in this analysis should not be whitened as Kilosort does. Do not use the waveforms extracted from `temp_wh.dat` directly. Do not use `whitening_mat_inv.npy` or `whitening_mat.npy` in Kilosort2.5 / Kilosort3 because they are not what Kilosort used to whiten the data (<https://github.com/cortex-lab/phy/issues/1040>)!
- Please analyze data from different brain regions like cortex and striatum individually since they might have different drifts and neuronal properties.
- Please raise an issue if you meet any bugs or have any questions. We are looking forward to your feedback!

## References

> [HDBSCAN](https://scikit-learn.org/stable/modules/clustering.html#hdbscan)  
> HDBSCAN - Hierarchical Density-Based Spatial Clustering of Applications with Noise. Performs DBSCAN over varying epsilon values and integrates the result to find a clustering that gives the best stability over epsilon. This allows HDBSCAN to find clusters of varying densities (unlike DBSCAN), and be more robust to parameter selection.
> 
> Campello, R.J.G.B., Moulavi, D., Sander, J. (2013). Density-Based Clustering Based on Hierarchical Density Estimates. In: Pei, J., Tseng, V.S., Cao, L., Motoda, H., Xu, G. (eds) Advances in Knowledge Discovery and Data Mining. PAKDD 2013. Lecture Notes in Computer Science(), vol 7819. Springer, Berlin, Heidelberg. Density-Based Clustering Based on Hierarchical Density Estimates  
>
> L. McInnes and J. Healy, (2017). Accelerated Hierarchical Density Based Clustering. In: IEEE International Conference on Data Mining Workshops (ICDMW), 2017, pp. 33-42. Accelerated Hierarchical Density Based Clustering

> [Kilosort](https://github.com/MouseLand/Kilosort)  
> Fast spike sorting with drift correction  
> 
> Pachitariu, Marius, Shashwat Sridhar, Jacob Pennington, and Carsen Stringer. “Spike Sorting with Kilosort4.” Nature Methods 21, no. 5 (May 2024): 914–21. https://doi.org/10.1038/s41592-024-02232-7.

> [DREDge](https://github.com/evarol/DREDge)  
> Robust online multiband drift estimation in electrophysiology data  
> 
> DREDge: robust motion correction for high-density extracellular recordings across species. https://www.biorxiv.org/content/10.1101/2023.10.24.563768v1

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

