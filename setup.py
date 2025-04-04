from setuptools import setup, find_packages

setup(
    name='pyKilosort',
    version='0.0.1',
    packages=find_packages(),
    discription='A Python package for tracking neurons across days',
    author='Yue Huang',
    author_email='yue_huang@pku.edu.cn',
    url='https://github.com/jiumao2/Kilomatch',
    python_requires='>=3.9',
    install_requires=[
        'hdbscan',
        'scikit-learn',
        'ipykernel',
        'tqdm',
        'h5py',
        'matplotlib'
    ]
)
