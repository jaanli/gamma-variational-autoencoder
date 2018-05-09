import pathlib
import time
import scipy.misc
from mxnet import nd
import mxnet as mx
import h5py
import numpy as np
from mxnet import gluon

import distributions


class DeepLatentGammaModel(gluon.HybridBlock):
  def __init__(self):
    super().__init__()
    with self.name_scope():
      self.log_prior = GammaLogProb()
      # self.log_prior = GaussianLogProb()
      # generative network parameterizes the likelihood
      self.net = gluon.nn.HybridSequential()
      with self.net.name_scope():
        self.net.add(gluon.nn.Dense(
            200, 'relu', weight_initializer=mx.init.Xavier()))
        self.net.add(gluon.nn.Dense(
            200, 'relu', weight_initializer=mx.init.Xavier()))
        self.net.add(gluon.nn.Dense(784, weight_initializer=mx.init.Xavier()))
      self.log_lik = BernoulliLogLik()

  def hybrid_forward(self, F, z, x):
    # use a sparse Gamma(shape=0.3, scale=3) prior
    log_prior = self.log_prior(z, 0.3 * F.ones_like(z), 3. * F.ones_like(z))
    # log_prior = self.log_prior(z, F.zeros_like(z), F.ones_like(z))
    logits = self.net(z)
    log_lik = self.log_lik(x, logits)
    return F.sum(log_lik, -1), F.sum(log_prior, -1)


class ELBO(gluon.HybridBlock):
  def __init__(self, model, variational):
    super().__init__()
    with self.name_scope():
      self.variational = variational
      self.model = model

  def hybrid_forward(self, F, x):
    z, log_q_z = self.variational(x)
    log_lik, log_prior = self.model(z, x)
    return log_lik + log_prior - log_q_z


class BernoulliLogLik(gluon.HybridBlock):
  """Calculate log probability of a Bernoulli."""

  def __init__(self):
    super().__init__()

  def hybrid_forward(self, F, x, logits):
    """Bernoulli log prob is
    x * log(1 + exp(-z))^(-1) + (1-x) * log(1 + exp(z))^(-1)
      = - x * log(1 + exp(z)) + x * log(exp(z)) - log(1 + exp(z)) + x * log(1 + exp(z))
      = x * z - log(1 + exp(z))
      = x * z - max(0, z) - log(1 + exp(-|z|)
    In the last step, observe that softplus(z) ~= z when z large.
    When z small, we hit underflow.
    """
    return x * logits - F.relu(logits) - F.Activation(-F.abs(logits), 'softrelu')


class AmortizedGammaVariational(gluon.HybridBlock):
  def __init__(self, latent_size, batch_size):
    super().__init__()
    self.net = gluon.nn.HybridSequential()
    with self.name_scope():
      self.reparam_gamma = distributions.ReparameterizedGamma(B=8)
      self.gamma_log_prob = GammaLogProb()
      with self.net.name_scope():
        self.net.add(gluon.nn.Dense(200, activation='relu', flatten=True))
        self.net.add(gluon.nn.Dense(200, activation='relu'))
        self.net.add(gluon.nn.Dense(latent_size * 2))

  def hybrid_forward(self, F, x):
    mean_scale_arg = self.net(x)
    shape_arg, scale_arg = F.split(mean_scale_arg, num_outputs=2, axis=-1)
    shape = F.Activation(shape_arg, 'softrelu')
    scale = F.Activation(scale_arg, 'softrelu')
    z = self.reparam_gamma(shape, scale)
    log_prob = self.gamma_log_prob(z, shape, scale)
    return z, F.sum(log_prob, -1)


class GammaLogProb(gluon.HybridBlock):
  def __init__(self):
    super().__init__()

  def hybrid_forward(self, F, x, shape, scale):
    return -F.gammaln(shape) - shape * F.log(scale) \
        + (shape - 1.) * F.log(x) - x / scale


if __name__ == '__main__':
  np.random.seed(24232)
  mx.random.seed(2423232)

  USE_GPU = False
  LATENT_SIZE = 32
  BATCH_SIZE = 64
  PRINT_EVERY = 100
  MAX_ITERATIONS = 1000000
  OUT_DIR = pathlib.Path(pathlib.os.environ['LOG']) / 'debug'

  dataset = mx.gluon.data.vision.MNIST(
      train=True,
      transform=lambda data, label: (
          np.round(data.astype(np.float32) / 255), label))
  train_data = mx.gluon.data.DataLoader(dataset,
                                        batch_size=BATCH_SIZE, shuffle=True)

  ctx = [mx.gpu(0)] if USE_GPU else [mx.cpu()]
  with mx.Context(ctx[0]):
    variational = AmortizedGammaVariational(LATENT_SIZE, BATCH_SIZE)
    model = DeepLatentGammaModel()
    elbo = ELBO(model, variational)

    variational.hybridize()
    model.hybridize()
    elbo.hybridize()

    variational.initialize(mx.init.Xavier())
    model.initialize(mx.init.Xavier())

    params = model.collect_params()
    params.update(variational.collect_params())
    trainer = gluon.Trainer(
        params, 'rmsprop', {'learning_rate': 0.00001, 'centered': True})
    wd_param = params.get('hybridsequential0_dense2_weight')

    def get_posterior_predictive(data, step):
      z, _ = variational(data)
      logits = model.net(z)
      probs = nd.sigmoid(logits)
      np_probs = probs.asnumpy()
      for i, prob in enumerate(np_probs):
        prob = prob.reshape((28, 28))
        scipy.misc.imsave(OUT_DIR / f'step_{step}_test_{i}.jpg', prob)

    step = 0
    t0 = time.time()
    for data, _ in train_data:
      break
    while step < MAX_ITERATIONS:
      for _, _ in train_data:
        data = data.reshape(-1, 784)
        with mx.autograd.record():
          elbo_batch = elbo(data)
          loss = -elbo_batch
          loss.backward()
        for name, param in variational.collect_params().items():
          g = param.grad().asnumpy()
          # print(name, g.max(), g.min())
        if step % PRINT_EVERY == 0:
          get_posterior_predictive(data, step)
          np_elbo = np.mean(elbo_batch.asnumpy())
          t1 = time.time()
          speed = (t1 - t0) / PRINT_EVERY
          t0 = t1
          print(f'Iter {step}\tELBO: {np_elbo:.1f}\tspeed: {speed:.3e} s/iter')
        trainer.step(BATCH_SIZE)
        step += 1
