# Non Intrusive Load Monitoring

This project contains source code for NILM algorithms and experiments on the UK-DALE dataset.

## Pre-requisites

- Tensorflow
- Numpy
- Pandas
- Matplotlib

## Folder structure

- Algos: Actual implementation of the models
    - multi: Multi-appliance models
- lib: Supporting modules
- processing: Notebooks for pre-processing
- experiments: different experiments run
    - exp_generalization: Generalization experiments
    - exp_multi_appliance: Multi-appliance experiments

## Usage

- Download the 2017 release of the UK-DALE dataset from https://jack-kelly.com/data/
- Run the `processing/processing-enhanced.ipynb `file to generate data chunks for house 1
- Run the `processing/processing-enhanced-house-2.ipynb` file to generate data chunks for house 2
- Run either the `experiments/sample-experiment.py` file or any of the notebooks experiments in the experiments directory (ideally in a Google Colab environment)
