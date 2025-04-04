from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='pyKilomatch',
    version='0.0.1',
    packages=find_packages(),
    discription='A Python package for tracking neurons across days',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Yue Huang',
    author_email='yue_huang@pku.edu.cn',
    url='https://github.com/jiumao2/pyKilomatch',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.9',
    install_requires=[
        'hdbscan',
        'scikit-learn',
        'ipykernel',
        'tqdm',
        'h5py',
        'matplotlib',
        'hjson'
    ]
)
