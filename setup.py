from setuptools import setup, find_packages

setup(
    name='RegGAIN', 
    version='1.0.0',
    author='Qiyuan Guan',
    author_email='202411961@mail.sdu.edu.cn',
    description='RegGAIN is a self-supervised graph contrastive learning framework for inferring GRNs by integrating scRNA-seq data with species-specific prior networks.',
    url='https://github.com/SDU-Math-SunLab/RegGAIN',
    packages=find_packages(), 
    python_requires='>=3.9', 
    install_requires=[
        'numpy>=1.26.4',
        'pandas>=2.2.2',
        'anndata>=0.10.9',
        'scanpy>=1.10.2',
        'matplotlib>=3.9.2',
        'scikit-learn>=1.5.1',
        'torch>=2.4.0',
        'torch_geometric>=2.5.3',
        'tqdm>=4.66.5',
        'networkx>=3.2.1',
    ],
entry_points={
    'console_scripts': [
        'reggain=RegGAIN_script.run:main',
    ],
},
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
