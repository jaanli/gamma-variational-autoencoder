import pathlib
import time
import scipy.misc
from mxnet import nd
import mxnet as mx
import h5py
import numpy as np
from mxnet import gluon


class DeepLatentGaussianModel(gluon.HybridBlock):
  def __init__(self):
    super().__init__()
    with self.name_scope():
      self.log_prior = GaussianLogProb()
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
    log_prior = self.log_prior(z, F.zeros_like(z), F.ones_like(z))
    logits = self.net(z)
    log_lik = self.log_lik(x, logits)
    return F.sum(log_lik, -1), F.sum(log_prior, -1)


class ELBO(gluon.HybridBlock):
  def __init__(self, model, variational):
    super().__init__()
    with self.name_scope():
      self.variational = variational
      self.model = model
      self.kl = GaussianKL()

  def hybrid_forward(self, F, x):
    z, mu_z, sigma_z, log_q_z = self.variational(x)
    log_lik, log_prior = self.model(z, x)
    # return log_lik + log_prior - log_q_z
    kl = F.sum(self.kl(mu_z, sigma_z), -1)
    return log_lik - kl


class GaussianKL(gluon.HybridBlock):
  def __init__(self):
    super().__init__()

  def hybrid_forward(self, F, mu, sigma):
    return -F.log(sigma) + (F.square(sigma) + F.square(mu)) / 2. - 0.5


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


class GaussianLogProb(gluon.HybridBlock):
  def __init__(self):
    super().__init__()

  def hybrid_forward(self, F, x, mean, sigma):
    variance = F.square(sigma)
    return -0.5 * F.log(2. * np.pi * F.square(sigma)) \
        - F.square(x - mean) / variance


class AmortizedGaussianVariational(gluon.HybridBlock):
  def __init__(self, latent_size, batch_size):
    super().__init__()
    self.net = gluon.nn.HybridSequential()
    with self.name_scope():
      self.gaussian_log_prob = GaussianLogProb()
      init = mx.init.Xavier()
      with self.net.name_scope():
        self.net.add(gluon.nn.Dense(200, activation='relu', flatten=True,
                                    weight_initializer=init))
        self.net.add(gluon.nn.Dense(200, activation='relu',
                                    weight_initializer=mx.init.Xavier()))
        self.net.add(gluon.nn.Dense(latent_size * 2,
                                    weight_initializer=mx.init.Xavier()))

  def hybrid_forward(self, F, x):
    mean_sigma_arg = self.net(x)
    mu, sigma_arg = F.split(mean_sigma_arg, num_outputs=2, axis=-1)
    sigma = F.Activation(sigma_arg, 'softrelu')
    eps = F.sample_normal(F.zeros_like(mu), F.ones_like(mu))
    z = mu + eps * sigma
    log_prob = self.gaussian_log_prob(z, mu, sigma)
    return z, mu, sigma, F.sum(log_prob, -1)


if __name__ == '__main__':
  np.random.seed(24232)
  mx.random.seed(2423232)

  USE_GPU = False
  LATENT_SIZE = 100
  BATCH_SIZE = 64
  PRINT_EVERY = 1000
  MAX_ITERATIONS = 1000000
  OUT_DIR = pathlib.Path(pathlib.os.environ['LOG']) / 'debug'

  # hdf5 file from:
  # https://github.com/altosaar/proximity_vi/blob/master/get_binary_mnist.py
  data_path = pathlib.Path(pathlib.os.environ['DAT']) / 'binarized_mnist.hdf5'
  f = h5py.File(data_path, 'r')
  raw_data = f['train'][:][:]
  f.close()

  def get_data():
    return mx.io.NDArrayIter(
        data={'data': nd.array(raw_data)},
        label={'label': range(len(raw_data)) * np.ones((len(raw_data),))},
        batch_size=BATCH_SIZE,
        last_batch_handle='discard',
        shuffle=True)

  ctx = [mx.gpu(0)] if USE_GPU else [mx.cpu()]
  with mx.Context(ctx[0]):
    variational = AmortizedGaussianVariational(LATENT_SIZE, BATCH_SIZE)
    model = DeepLatentGaussianModel()
    elbo = ELBO(model, variational)

    variational.hybridize()
    model.hybridize()
    elbo.hybridize()

    variational.initialize(mx.init.Xavier())
    model.initialize(mx.init.Xavier())

    params = model.collect_params()
    params.update(variational.collect_params())
    trainer = gluon.Trainer(params, 'rmsprop', {'learning_rate': 0.001})
    # , 'centered': True})

    def get_posterior_predictive(batch, step):
      z, _, _, _ = variational(batch.data[0])
      logits = model.net(z)
      probs = nd.sigmoid(logits)
      np_probs = probs.asnumpy()
      for i, prob in enumerate(np_probs):
        prob = prob.reshape((28, 28))
        scipy.misc.imsave(OUT_DIR / f'step_{step}_test_{i}.jpg', prob)

    step = 0
    t0 = time.time()
    train_data = get_data()
    while step < MAX_ITERATIONS:
      if step % (train_data.num_data // BATCH_SIZE) == 0:
        train_data = get_data()
      data = next(train_data)
      with mx.autograd.record():
        elbo_batch = elbo(data.data[0])
        (-elbo_batch).backward()
      if step % PRINT_EVERY == 0:
        get_posterior_predictive(data, step)
        np_elbo = np.mean(elbo_batch.asnumpy())
        t1 = time.time()
        speed = (t1 - t0) / PRINT_EVERY
        t0 = t1
        print(f'Iter {step}\tELBO: {np_elbo:.1f}\tspeed: {speed:.3e} s/iter')
      trainer.step(BATCH_SIZE)
      step += 1
