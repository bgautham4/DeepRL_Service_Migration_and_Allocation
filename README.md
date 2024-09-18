# Capstone : [DRL Based Service Migration and Resource Allocation in Vehicular Edge Networks](https://ieeexplore.ieee.org/document/10574641)
## Note: This repository is intended mostly for reference and does not serve to replicate the results one-to-one from the paper. However you can mail the authors regarding clarifications for the implementation of the system model in case of replicating the experiment.

The bulk of the code written in [Julia](https://julialang.org/) denotes the underlying environment and the code for the DNNs+algorithm. A python script in `time_series_gen/` exsists to demonstrate the time series generation process using [LHS](https://en.wikipedia.org/wiki/Latin_hypercube_sampling). 

You can inspect the `Project.toml` files for the neccessary julia packages (OR make use of them by instatiating a new enviroment with the Project.toml file).
Some additional time series files which were used through the writing of the paper are also available. By changing the path to point to these files in the training and testing loops, you can emulate (**NOT replicate**) the experiments from the paper.

