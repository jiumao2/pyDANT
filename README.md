# pyDANT: A Python toolbox for Density-based Across-day Neuron Tracking

[![View pyDANT on GitHub](https://img.shields.io/badge/GitHub-pyDANT-blue.svg)](https://github.com/jiumao2/pyDANT)
[![Documentation Status](https://app.readthedocs.org/projects/pydant/badge/)](https://pydant.readthedocs.io/en/latest/)
![PyPI - Version](https://img.shields.io/pypi/v/pyDANT)
![GitHub License](https://img.shields.io/github/license/jiumao2/pyDANT)

A Python toolbox for tracking the neurons across days.

This project is a Python implementation of [DANT](https://github.com/jiumao2/DANT), converted from the original MATLAB code. Read the [documentation](https://pydant.readthedocs.io/en/latest/) for more details.

## Installation

- It is recommended to install the pyDANT package using Anaconda:

```bash
conda create -n pyDANT python=3.11
conda activate pyDANT
pip install pyDANT
```  

## How to use it

Please follow the [tutorial](https://pydant.readthedocs.io/en/latest/Tutorials.html) to run your dataset.  

Please raise an issue if you meet any bugs or have any questions. We are looking forward to your feedback!

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
> Windolf, Charlie, Han Yu, Angelique C. Paulk, Domokos Meszéna, William Muñoz, Julien Boussard, Richard Hardstone, et al. “DREDge: Robust Motion Correction for High-Density Extracellular Recordings across Species.” Nature Methods, March 6, 2025. https://doi.org/10.1038/s41592-025-02614-5.


## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

