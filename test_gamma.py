import distributions
import numpy as np
import mxnet as mx
from mxnet import nd
import scipy.stats
mx.random.seed(24232)
np.random.seed(2423242)


def sample_gamma(shape, scale, n_samples):
  reparam_gamma = distributions.ReparameterizedGamma(B=8)
  if not isinstance(shape, np.ndarray):
    shape = np.array([[shape]])
    scale = np.array([[scale]])
  shape = np.repeat(shape, n_samples, axis=0)
  scale = np.repeat(scale, n_samples, axis=0)
  sample = reparam_gamma(nd.array(shape), nd.array(scale))
  return sample


def check_gamma_mean(shape, scale, n_samples):
  sample = sample_gamma(shape, scale, n_samples)
  mean = sample.asnumpy().mean(axis=0)
  print('actual, computed:')
  true_mean = np.squeeze(shape * scale)
  print(true_mean, mean)
  np.testing.assert_allclose(true_mean, mean, rtol=1e-1)


def test_gamma_sampling_mean():
  """Check that reparameterized samples recover the correct mean."""
  check_gamma_mean(np.array([[1., 1., 1.]]), np.array([[1., 1., 1.]]), 1000)
  check_gamma_mean(10., 1., 1000)
  check_gamma_mean(1., 1., 1000)
  check_gamma_mean(0.1, 1., 10000)
  check_gamma_mean(0.01, 1., 100000)
  check_gamma_mean(1., 30., 10000)
  check_gamma_mean(5., 30., 10000)
  check_gamma_mean(0.3, 3., 10000)


def check_gamma_grads(np_shape, np_scale):
  """Test that reparameterization gradients are correct."""
  if not isinstance(np_shape, np.ndarray):
    np_shape = np.array([[np_shape]])
    np_scale = np.array([[np_scale]])
  shape = nd.array(np_shape)
  scale = nd.array(np_scale)
  shape.attach_grad()
  scale.attach_grad()

  def function(F, z):
    return F.square(z) - 3.5

  reparam_gamma = distributions.ReparameterizedGamma(B=8)
  # compute gradient of a simple function f(z) = z
  g_shape_list = []
  g_scale_list = []
  for _ in range(1000):
    with mx.autograd.record():
      z_sample = reparam_gamma(shape, scale)
      f = function(nd, z_sample)
      f.backward()
    g_shape_list.append(shape.grad.asnumpy())
    g_scale_list.append(scale.grad.asnumpy())
  g_shape = np.mean(g_shape_list, axis=0)
  g_scale = np.mean(g_scale_list, axis=0)
  np_z = scipy.stats.gamma.rvs(
      np_shape, scale=np_scale, size=(100000, np_shape.shape[-1]))
  score_shape, score_scale = gamma_score(np_z, np_shape, np_scale)
  f_z = function(np, np_z)
  np_g_shape = np.mean(score_shape * f_z, axis=0)
  np_g_scale = np.mean(score_scale * f_z, axis=0)
  print('shape score, reparam')
  print(np_g_shape, g_shape)
  np.testing.assert_allclose(np_g_shape, np.squeeze(g_shape), rtol=2e-1)
  print('scale score, reparam')
  print(np_g_scale, g_scale)
  np.testing.assert_allclose(np_g_scale, np.squeeze(g_scale), rtol=2e-1)


def test_gamma_grads():
  check_gamma_grads(1., 1.)
  check_gamma_grads(1., 3.)
  check_gamma_grads(0.3, 3.)
  check_gamma_grads(np.array([[0.3, 0.3, 0.3]]), np.array([[3., 3., 3.]]))


def gamma_score(z, shape, scale):
  """Score function of gamma."""
  score_shape = -scipy.special.psi(shape) - np.log(scale) + np.log(z)
  score_scale = -shape / scale + z / scale / scale
  return score_shape, score_scale


if __name__ == '__main__':
  # test_gamma_sampling_mean()
  test_gamma_grads()
