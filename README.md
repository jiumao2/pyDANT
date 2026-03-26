# pyDANT: A Python toolbox for Density-based Across-day Neuron Tracking

[![View pyDANT on GitHub](https://img.shields.io/badge/GitHub-pyDANT-blue.svg)](https://github.com/jiumao2/pyDANT)
[![Documentation Status](https://app.readthedocs.org/projects/dant/badge/)](https://dant.readthedocs.io/en/latest/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jiumao2/pyDANT/blob/master/pyDANT_demo.ipynb)
![PyPI - Version](https://img.shields.io/pypi/v/pyDANT)
![GitHub License](https://img.shields.io/github/license/jiumao2/pyDANT)

**pyDANT** is a Python toolbox designed for the robust, longitudinal tracking of neurons across multiple recording sessions using high-density probes.

---

### 📄 Preprint
**[Density-based longitudinal neuron tracking in high-density electrophysiological recordings](https://www.biorxiv.org/content/10.64898/2025.12.19.695632v1)**

📚 **[Read the Documentation](https://dant.readthedocs.io/en/latest/)**

🐍 **[Check out the MATLAB version (DANT)](https://github.com/jiumao2/DANT)**

---

## Installation

This section describes installation of the pyDANT.


### Install with Anaconda

Anaconda is recommended for managing the pyDANT environment.

```bash
conda create -n pyDANT python=3.11
conda activate pyDANT
pip install pyDANT
```

### Install from Python Package Index (PyPI)

You can also install pyDANT directly from PyPI:

```bash
pip install pyDANT
```

## Install from Source

If you prefer to install from source, clone the repository and install it manually:

```bash
git clone https://github.com/jiumao2/pyDANT.git
cd pyDANT
pip install -e .
```

## 🚀 Getting Started

To help you get familiar with the pipeline, we have provided an example dataset and a step-by-step walkthrough.

1. **Download the Data:** [Example Dataset for pyDANT (Figshare)](https://figshare.com/articles/dataset/Example_Dataset_for_pyDANT/30596303)
2. **Run the Pipeline:** Follow our comprehensive [Tutorial](https://dant.readthedocs.io/en/latest/Tutorials_Python.html) to run the example data or process your own recordings.

If you encounter any bugs, have questions, or want to suggest a feature, please [open an issue](https://github.com/jiumao2/pyDANT/issues). We look forward to your feedback!

## 📝 Citation

If you use pyDANT in your research, please cite our preprint:

```bibtex
@article {Huang2025DANT,
    author = {Huang, Yue and Wang, Hanbo and Cao, Jiaming and Chen, Yu and Wang, Xuanning and Zhao, Yujie and Ren, Hengkun and Zheng, Qiang and Yu, Jianing},
    title = {Density-based longitudinal neuron tracking in high-density electrophysiological recordings},
    year = {2025},
    doi = {10.64898/2025.12.19.695632},
    publisher = {Cold Spring Harbor Laboratory},
    URL = {https://www.biorxiv.org/content/early/2025/12/23/2025.12.19.695632},
    journal = {bioRxiv}
}
```  

## 📚 References & Acknowledgements

pyDANT builds upon and integrates several excellent open-source tools. We extend our gratitude to the authors of the following packages:

* **[HDBSCAN](https://scikit-learn.org/stable/modules/clustering.html#hdbscan):** Hierarchical Density-Based Spatial Clustering of Applications with Noise. (Campello et al., 2013; McInnes & Healy, 2017).
* **[Kilosort](https://github.com/MouseLand/Kilosort):** Fast spike sorting with drift correction. (Pachitariu et al., 2024).
* **[DREDge](https://github.com/evarol/DREDge):** Robust online multiband drift estimation in electrophysiology data. (Windolf et al., 2025).

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
