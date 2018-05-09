import mxnet as mx
from mxnet import gluon


class ReparameterizedGamma(gluon.HybridBlock):
  """Returns a reparameterized sample from a Gamma(shape, scale) distribution.

  shape, scale are the shape and scale of the distribution. We use Algorithm 1
  of [1], but sample from a Gamma(shape, 1) to guarantee acceptance. We also
  use shape augmentation as in Section 5 of [1].

  References:
  1. Naesseth et al. (2017).
  """

  def __init__(self, B):
    """B is the number of times to augment the shape, useful for sparsity."""
    super().__init__()
    self.B = B

  def hybrid_forward(self, F, shape, scale):
    # sample the \tilde z ~ Gamma(shape + B, 1.) to guarantee acceptance
    one = F.ones_like(shape)
    z_tilde = F.sample_gamma(shape + self.B, one)
    # compute the epsilon corresponding to \tilde z; this epsilon is 'accepted'
    # \epsilon = h_inverse(z_tilde; shape + B)
    eps = self.compute_h_inverse(F, z_tilde, F.stop_gradient(shape) + self.B)
    # now compute z_tilde = h(epsilon, shape + B)
    z_tilde = self.compute_h(F, eps, shape + self.B)
    # E_{u_1,...,u_B, \tilde z}[f(\tilde z) \prod_i u_i^{(shape + i - 1)^{-1}}]
    #  = E_{u_1,...,u_B, \pi(eps)}[f(h(eps, shape + B) \prod_i ...]
    B_range = F.arange(start=1, stop=self.B + 1)
    # expand dims broadcast with shape
    B_range = F.expand_dims(F.expand_dims(B_range, -1), -1)
    zero = F.zeros_like(shape)
    unif_sample = F.sample_uniform(zero, one, shape=(self.B))
    # transpose so that boosting dimension is the innermost
    unif_sample = F.transpose(unif_sample, axes=(2, 0, 1))
    unif_prod = F.prod(
        unif_sample**(1. / (F.broadcast_add(shape, B_range) - 1.)), axis=0)
    # This reparameterized sample is distributed as Gamma(shape, 1)
    # z = h(eps, shape + B) \prod_i u_i^{1 / (shape + i - 1)}
    z = z_tilde * unif_prod
    # Divide by scale to get a sample distributed as Gamma(shape, scale)
    return z * scale

  def compute_h(self, F, eps, shape):
    return (shape - 1. / 3.) * (1. + eps / F.sqrt(9. * shape - 3.))**3

  def compute_h_inverse(self, F, z, shape):
    return F.sqrt(9. * shape - 3.) * ((z / (shape - 1. / 3.))**(1. / 3.) - 1.)
