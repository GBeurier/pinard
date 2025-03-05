from setuptools import setup, find_packages

from pathlib import Path

from pinard import __version__

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

extra_api = [
    'returns-decorator',
]

extra_bin = [
    *extra_api,
]

extra_test = [
    *extra_api,
    'pytest>=4',
    'pytest-cov>=5',
]
extra_dev = [
    *extra_test,
]

extra_ci = [
    *extra_test,
    'python-coveralls',
]

# Nouveaux extras pour les frameworks de ML
extra_tf = [
    'tensorflow>=2.10.0',
]

extra_torch = [
    'torch>=2.0.0',
]

extra_keras = [
    'keras>=3.0.0',
]

extra_jax = [
    'jax>=0.4.10',
    'jaxlib>=0.4.10',
]

# Extra qui contient tous les frameworks
extra_all_frameworks = [
    *extra_tf,
    *extra_torch,
    *extra_keras,
    *extra_jax,
]


setup(
    name='pinard',
    version=__version__,
    description='Pinard: a Pipeline for Nirs Analysis ReloadeD.',
    url='https://github.com/gbeurier/pinard',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Gregory Beurier',
    author_email='beurier@cirad.fr',

    packages=find_packages(),

    install_requires=[
        'joblib',
        'jsonschema',
        'kennard-stone',
        'numpy',
        'pandas',
        'PyWavelets',
        'scikit-learn',
        'scipy',
        'twinning',
        'optuna'
    ],

    extras_require={
        'math': extra_api,
        'bin': extra_bin,
        'test': extra_test,
        'dev': extra_dev,
        'ci': extra_ci,
        'tf': extra_tf,
        'torch': extra_torch,
        'keras': extra_keras,
        'jax': extra_jax,
        'all': extra_all_frameworks,
        'full': [*extra_ci, *extra_all_frameworks],
    },

    entry_points={
        'console_scripts': [
            'add=my_pip_package.math:cmd_add',
        ],
    },

    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Environment :: GPU :: NVIDIA CUDA :: 10.1',
        'Development Status :: 4 - Beta',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
)
