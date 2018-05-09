## gamma-variational-autoencoder

This is an example implementation of Reparameterized Rejection sampling (Naesseth et al. https://arxiv.org/abs/1610.05683).

There are two examples here, implemented in MXNet:

1. A standard VAE with Gaussian latent variables.

2. A VAE with Gamma-distributed latent variables. 

A key to the fast implementation is to sample Gamma variables, then calculate the inverse function to get the epsilon used in the reparameterization algorithm. Thanks to Christian Naesseth for pointing out this nice trick!
