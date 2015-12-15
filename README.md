# Relative MMD
Code for computing the MMD and Relative MMD test. Please refer to http://arxiv.org/pdf/1511.04581.pdf

An example is provided under Example_Vae.py which trains two varational auto-encoders and then compares their samples to a holdout set using MMD.

The Relative MMD computations need to be exact to assure their validity. Thus matrix operations can be costly. Please make sure you have a properly configured numpy installation (linked to optimized blas libraries like openblas).

For a matlab version of the Relative MMD test please see https://github.com/wbounliphone/relative_similarity_test

For questions and or bug reports please dont hesitate to contact eugene.belilovsky@inria.fr
