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
    'pytest-cov>=2',
]
extra_dev = [
    *extra_test,
]

extra_ci = [
    *extra_test,
    'python-coveralls',
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
        'kennard-stone',
        'numpy',
        'pandas',
        'PyWavelets',
        'scikit-learn',
        'scipy',
        'twinning',
    ],

    extras_require={
        'math': extra_api,
        'bin': extra_bin,
        'test': extra_test,
        'dev': extra_dev,
        'ci': extra_ci,
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
