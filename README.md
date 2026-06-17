# pyDANT: A Python toolbox for Density-based Across-day Neuron Tracking

[![View pyDANT on GitHub](https://img.shields.io/badge/GitHub-pyDANT-blue.svg)](https://github.com/jiumao2/pyDANT)
[![Documentation Status](https://app.readthedocs.org/projects/dant/badge/)](https://dant.readthedocs.io/en/latest/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jiumao2/pyDANT/blob/master/pyDANT_colab_demo.ipynb)
![PyPI - Version](https://img.shields.io/pypi/v/pyDANT)
![GitHub License](https://img.shields.io/github/license/jiumao2/pyDANT)

<p align="center">
  <img src="./Overview.png" alt="DANT graphical abstract" width="600">
</p>

**pyDANT** is a Python toolbox that combines iterative motion correction and density-based clustering to robustly track single neurons across days of high-density recordings.

---

### 📄 Published article
**[Density-based longitudinal neuron tracking in high-density electrophysiological recordings](https://doi.org/10.1016/j.patter.2026.101590)**

📚 **[Read the Documentation](https://dant.readthedocs.io/en/latest/)**

🧮 **[Check out the MATLAB version (DANT)](https://github.com/jiumao2/DANT)**

---

## ⚙️ Installation

This section describes how to install pyDANT.


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

### Install from Source

If you prefer to install from source, clone the repository and install it manually:

```bash
git clone https://github.com/jiumao2/pyDANT.git
cd pyDANT
pip install -e .
```

## 🚀 Getting Started

To help you get familiar with the pipeline, we provide two tutorials.

1. **Colab Tutorial:** Open the [pyDANT Colab demo](https://colab.research.google.com/github/jiumao2/pyDANT/blob/master/pyDANT_colab_demo.ipynb) to run pyDANT in a browser without local setup or pre-downloading data.
2. **Documentation Tutorial:** Download the [example dataset for pyDANT](https://figshare.com/articles/dataset/Example_Dataset_for_pyDANT/30596303), then follow the [Python tutorial](https://dant.readthedocs.io/en/latest/Tutorials_Python.html) to run pyDANT locally or process your own recordings.

If you encounter any bugs, have questions, or want to suggest a feature, please [open an issue](https://github.com/jiumao2/pyDANT/issues). We look forward to your feedback!

## 📝 Citation

If you use pyDANT in your research, please cite our article:

```bibtex
@article{Huang2026DANT,
    author = {Huang, Yue and Wang, Hanbo and Cao, Jiaming and Chen, Yu and Wang, Xuanning and Zhao, Yujie and Ren, Hengkun and Zheng, Qiang and Yu, Jianing},
    title = {Density-based longitudinal neuron tracking in high-density electrophysiological recordings},
    journal = {Patterns},
    year = {2026},
    doi = {10.1016/j.patter.2026.101590},
    url = {https://doi.org/10.1016/j.patter.2026.101590},
    note = {Available online 17 June 2026}
}
```  

## 📚 References & Acknowledgements

pyDANT builds upon and integrates several excellent open-source tools. We extend our gratitude to the authors of the following packages:

* **[HDBSCAN](https://scikit-learn.org/stable/modules/clustering.html#hdbscan):** Hierarchical Density-Based Spatial Clustering of Applications with Noise. (Campello et al., 2013; McInnes & Healy, 2017).
* **[Kilosort](https://github.com/MouseLand/Kilosort):** Fast spike sorting with drift correction. (Pachitariu et al., 2024).
* **[DREDge](https://github.com/evarol/DREDge):** Robust online multiband drift estimation in electrophysiology data. (Windolf et al., 2025).

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.








