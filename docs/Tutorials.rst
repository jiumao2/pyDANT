Tutorials
================

This tutorial provides a step-by-step guide on how to use the Kilomatch package for tracking neurons across sessions. It is designed to help you prepare your data and run the code effectively. You should install Kilomatch correctly before proceeding with this tutorial. If you haven't installed Kilomatch yet, please refer to the :doc:`Installation <Installation>` section.

.. _prepare_the_data_label:

Prepare the data
-----------------------

To use pyKilomatch, your data should be organized in a folder with the following structure:

.. code-block::

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

- The data files should adhere to the following formats:

===========================    ======================================  =================================================
Filename                       Shape                                   Explanation  
===========================    ======================================  =================================================
``session_index.npy``          (n_unit,)                               indicating the session. It should start from 1 (for compatability with MATLAB) and be coninuous without any gaps.
``waveform_all.npy``           (n_unit, n_channel, n_sample)           the mean waveform of each unit in μV. All units must share the same set of channels                         
``channel_locations.npy``      (n_channel, 2)                          x and y coordinates of each channel in μm. The y coordinate typically represents the depth
``peth.npy``                   (n_unit, n_point)                       optional, peri-event time histogram for each unit
``spike_times/UnitX.npy``      (n_spike,)                              spike times in milliseconds
===========================    ======================================  =================================================

Crucially, the waveforms used in this analysis must not be whitened, unlike those processed by Kilosort. Avoid direct use of waveforms from ``temp_wh.dat`` and refrain from using ``whitening_mat_inv.npy`` or ``whitening_mat.npy`` from Kilosort2.5 / Kilosort3 to "unwhiten" data. These matrices do not correspond to Kilosort's original whitening process (see this `issue <https://github.com/cortex-lab/phy/issues/1040>`_).

We recommend analyzing data from different brain regions (e.g., cortex and striatum) individually, as they may exhibit distinct drifts and neuronal properties. Please generate a separate ``spikeInfo.mat`` file for each brain region.

- Copy the ``settings.json`` and ``mainKilomatch.py`` files from the Kilomatch package to your data folder. It can be like:

.. code-block::

    data_folder
    ├── settings.json
    ├── mainKilomatch.py
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

Edit the settings
-----------------------

To make Kilomatch work, you need to edit the ``settings.json`` file in your data folder. At least, you need to specify the following fields:

.. code-block:: json

    {
        "path_to_data": ".", // path to spikeInfo.mat
        "output_folder": ".\\kilomatchOutput", // output folder
    }

If you don't want to use PETH feature, you should remove it in the ``motionEstimation`` part and ``clustering`` part. Here is what it looks like after editing:

.. code-block:: json

    // parameters for motion estimation
    "motionEstimation":{
        "max_distance": 100, // um. Unit pairs with distance larger than this value in Y direction will not be included for motion estimation
        "features": [
            ["Waveform", "AutoCorr"],
            ["Waveform", "AutoCorr"]
        ] // features used for motion estimation each iteration. Choose from "Waveform", "AutoCorr", "ISI", "PETH"
    },

and 

.. code-block:: json

    // parameters for clustering
    "clustering":{
        "max_distance": 100, // um. Unit pairs with distance larger than this value in Y direction will be considered as different clusters
        "features": ["Waveform", "AutoCorr"], // features used for motion estimation. Choose from "Waveform", "AutoCorr", "ISI", "PETH"
        "n_iter": 10 // number of iterations for the clustering algorithm
    },

Also, the ``mainKilomatch.py`` file should be edited to specify the path to the Kilomatch package:

.. code-block:: Python

    path_settings = r'./settings.json' # It should be the path to your settings.json file

To learn more about the settings, please refer to the :doc:`Change default settings <Change_default_settings>` section. The optimized settings can help you get better tracking results!

Run the code
-----------------------

Run ``mainKilomatch.py`` in your Python environment in the terminal or command prompt:

.. code-block::

    python mainKilomatch.py


Hopefully, you will get the tracking results in the output folder specified in the ``settings.json`` file. It can be like:

.. code-block::

    data_folder
    ├── settings.json
    ├── mainKilomatch.py
    ├── channel_locations.npy
    ├── waveform_all.npy
    ├── session_index.npy
    ├── peth.npy (optional)
    ├── spike_times/
    └── kilomatchOutput/
        ├── IdxCluster.npy
        ├── ClusterMatrix.npy
        ├── SimilarityMatrix.npy
        ├── ...
        └── Figures/


.. _output_label:

Understand the output
-----------------------

With some intermediate files, the main output file is located in ``kilomatchOutput`` folder, which contains the following important files:

===========================     =============================               =================
Field name                      Shape                                       Explanation  
===========================     =============================               =================
``IdxCluster.npy``              (n_unit,)                                   cluster index for each unit.
``ClusterMatrix.npy``           (n_unit x n_unit)                           cluster assignment matrix. ``ClusterMatrix(i,j) = 1`` means unit ``i`` and ``j`` are in the same cluster.
``MatchedPairs``                (n_pairs x 2)                               unit index for all matched pairs.
``SimilarityMatrix``            (n_unit x n_unit)                           weighted sum of the similarity between each pair of units.
===========================     =============================               =================

The most important file is ``IdxCluster.npy``, which assigns a unique cluster ID for each unit (-1 for non-matched units). You can use it to extract the matched units across sessions. To learn more about the output, please refer to the :doc:`Input and Output <IO>` section.

Tracking is completed! Now your cross-session analysis can be performed with the tracked neurons!


