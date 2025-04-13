# Readme

This repository contains all of the code required to generate the tables and plots provided in arXiv:2405.14697. To recreate the images simply run the provided notebook plots.ipynb. This will use the precomputed results stored in pickle files to generate the plots.

Should you want to run your own simulations and learn how to use the compressed sensing amplitude estimation approach, we recommend looking at the notebook csAE_example.ipynb. This contains a minimal working example demonstrating the functionality.

All of the data used to generate the results in the paper can be recreated by running the "run_ae_sims.py" python script from the command line with the following commands. This code block generates the data needed for Fig. 3 and Tab. 1. It runs 500 Monte Carlo trials of the csAE approach for random amplitudes in the range a=(0.1, ..., 0.9). With 12 threads on a workstation this takes about XXX hours to complete.

```
python run_ae_sims.py --save --dir=sims_final/ --nthreads=24 --num_lengths=6 --C=4 --adjacency=5 --num_mc=500 > sims_final/outfileC4.0.dat
python run_ae_sims.py --save --dir=sims_final/ --nthreads=24 --num_lengths=6 --C=8 --adjacency=5 --num_mc=500 > sims_final/outfileC8.0.dat

```

Once the data is generated, simply run the notebook plots.ipynb to generate the figures and tables. For your convenience, the precomputed data is already stored so there is no need to rerun these long simulations, except for reproducibility.
