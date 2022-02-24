# Amortized-MXL

Amortized-MXL: Scaling Bayesian Inference of Mixed Multinomial Logit Models to Large Datasets

This repository contains source code for the Amortized Variational Inference approach for Mixed Multinomial Logit models proposed in:

* [Rodrigues, F. Scaling Bayesian inference of mixed multinomial logit models to large datasets. In Transportation Research Part B: Methodological, 2022](https://www.sciencedirect.com/science/article/pii/S019126152200011X) (preprint version: https://arxiv.org/abs/2004.05426/)

The folder ["v1.0"](https://github.com/fmpr/amortized-mxl/tree/master/v1.0) contains the latest version of the code, which includes an efficient implementation in pure PyTorch, an easy-to-use formula interface for specifying utility functions and tutorials on how to use it. See for example the Jupyter notebook: [Demo - Simulated N=500 - MXL - SVI.ipynb](https://github.com/fmpr/amortized-mxl/tree/master/v1.0/Demo%20-%20Simulated%20N=500%20-%20MXL%20-%20SVI.ipynb).

The original code from the paper uses [Pyro](https://pyro.ai) and is available in the folder ["v0.1"](https://github.com/fmpr/amortized-mxl/tree/master/v0.1). 