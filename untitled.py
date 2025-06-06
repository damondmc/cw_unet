# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Layers for defining NCSN++.
"""
import functools
import string
from typing import Any, Optional, Tuple
from . import up_or_down_sampling_1d as up_or_down_sampling
import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np

SQRT2 = 1.41421356

def get_act(config):
  """Get activation functions from the config file."""

  if config.model.nonlinearity.lower() == 'elu':
    return nn.elu
  elif config.model.nonlinearity.lower() == 'relu':
    return nn.relu
  elif config.model.nonlinearity.lower() == 'lrelu':
    return functools.partial(nn.leaky_relu, negative_slope=0.2)
  elif config.model.nonlinearity.lower() == 'swish':
    return nn.swish
  else:
    raise NotImplementedError('activation function does not exist!')



def default_init(scale=1.):
  """The same initialization used in DDPM."""
  scale = 1e-10 if scale == 0 else scale
  return jnn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def conv1(x, out_planes, stride=1, bias=True, dilation=1, init_scale=1.):
  """1x1 convolution with DDPM initialization."""
  bias_init = jnn.initializers.zeros
  output = nn.Conv(out_planes, kernel_size=(1,),
                   strides=(stride), padding='SAME', use_bias=bias,
                   kernel_dilation=(dilation),
                   kernel_init=default_init(init_scale),
                   bias_init=bias_init)(x)
  return output


def conv3(x, out_planes, stride=1, bias=True, dilation=1, init_scale=1.):
  """3x1 convolution with DDPM initialization."""
  bias_init = jnn.initializers.zeros
  output = nn.Conv(
    out_planes,
    kernel_size=(3,),
    strides=(stride),
    padding='SAME',
    use_bias=bias,
    kernel_dilation=(dilation),
    kernel_init=default_init(init_scale),
    bias_init=bias_init)(x)
  return output


class NIN(nn.Module):
  num_units: int
  init_scale: float = 0.1

  @nn.compact
  def __call__(self, x):
    in_dim = int(x.shape[-1])
    W = self.param('W', default_init(scale=self.init_scale), (in_dim, self.num_units))
    b = self.param('b', jnn.initializers.zeros, (self.num_units,))
    y = contract_inner(x, W) + b
    assert y.shape == x.shape[:-1] + (self.num_units,)
    return y


def _einsum(a, b, c, x, y):
  einsum_str = '{},{}->{}'.format(''.join(a), ''.join(b), ''.join(c))
  return jnp.einsum(einsum_str, x, y)


def contract_inner(x, y):
  """tensordot(x, y, 1)."""
  x_chars = list(string.ascii_lowercase[:len(x.shape)])
  y_chars = list(string.ascii_uppercase[:len(y.shape)])
  assert len(x_chars) == len(x.shape) and len(y_chars) == len(y.shape)
  y_chars[0] = x_chars[-1]  # first axis of y and last of x get summed
  out_chars = x_chars[:-1] + y_chars[1:]
  return _einsum(x_chars, y_chars, out_chars, x, y)


class GaussianFourierProjection(nn.Module):
  """Gaussian Fourier embeddings for noise levels."""
  embedding_size: int = 256
  scale: float = 1.0

  @nn.compact
  def __call__(self, x):
    W = self.param('W', jax.nn.initializers.normal(stddev=self.scale), (self.embedding_size // 2,))
    W = jax.lax.stop_gradient(W)
    x_proj = x[:, None] * W[None, :] * 2 * jnp.pi
    return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)


class AttnBlockpp(nn.Module):
  """Channel-wise self-attention block. Modified from DDPM."""
  skip_rescale: bool = False
  init_scale: float = 0.

  @nn.compact
  def __call__(self, x):
    B, H, C = x.shape
    h = nn.GroupNorm(num_groups=min(x.shape[-1] // 4, 32))(x)
    q = NIN(C)(h)
    k = NIN(C)(h)
    v = NIN(C)(h)

    w = jnp.einsum('bhc,bHc->bhH', q, k) * (int(C) ** (-0.5))
    w = jnp.reshape(w, (B, H, H))
    w = jax.nn.softmax(w, axis=-1)
    w = jnp.reshape(w, (B, H, H))
    h = jnp.einsum('bhH,bHc->bhc', w, v)
    h = NIN(C, init_scale=self.init_scale)(h)
    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / jnp.sqrt(jnp.array(2., dtype=x.dtype))#np.sqrt(2.)


class ResnetBlockBigGANpp(nn.Module):
  """ResBlock adapted from BigGAN."""
  act: Any
  up: bool = False
  down: bool = False
  out_ch: Optional[int] = None
  dropout: float = 0.1
  fir: bool = False
  fir_kernel: Tuple[int] = (1, 3, 1)
  skip_rescale: bool = True
  init_scale: float = 0.
  up_down_factor: int = 2

  @nn.compact
  def __call__(self, x, temb=None, train=True):
    B, H, C = x.shape
    out_ch = self.out_ch if self.out_ch else C
    h = self.act(nn.GroupNorm(num_groups=min(x.shape[-1] // 4, 32), epsilon=1e-6)(x))

    if self.up:
      if self.fir:
        h = up_or_down_sampling.upsample_1d(h, self.fir_kernel, factor=self.up_down_factor)
        x = up_or_down_sampling.upsample_1d(x, self.fir_kernel, factor=self.up_down_factor)
      else:
        h = up_or_down_sampling.naive_upsample_1d(h, factor=self.up_down_factor)
        x = up_or_down_sampling.naive_upsample_1d(x, factor=self.up_down_factor)
    elif self.down:
      if self.fir:
        h = up_or_down_sampling.downsample_1d(h, self.fir_kernel, factor=self.up_down_factor)
        x = up_or_down_sampling.downsample_1d(x, self.fir_kernel, factor=self.up_down_factor)
      else:
        h = up_or_down_sampling.naive_downsample_1d(h, factor=self.up_down_factor)
        x = up_or_down_sampling.naive_downsample_1d(x, factor=self.up_down_factor)

    h = conv3(h, out_ch)
    # Add bias to each feature map conditioned on the time embedding
    if temb is not None:
      h += nn.Dense(out_ch, kernel_init=default_init())(self.act(temb))[:, None, :]

    h = self.act(nn.GroupNorm(num_groups=min(h.shape[-1] // 4, 32))(h))
    h = nn.Dropout(self.dropout)(h, deterministic=not train)
    h = conv3(h, out_ch, init_scale=self.init_scale)
    if C != out_ch or self.up or self.down:
      x = conv1(x, out_ch)

    if not self.skip_rescale:
      return x + h
    else:
      return (x + h) / SQRT2


# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Layers used for up-sampling or down-sampling images.

Many functions are ported from https://github.com/NVlabs/stylegan2.
"""

import flax.linen as nn
from typing import Any, Tuple, Optional, Sequence
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np


# Function ported from StyleGAN2
def get_weight(module,
               shape,
               weight_var='weight',
               kernel_init=None):
  """Get/create weight tensor for a convolution or fully-connected layer."""

  return module.param(weight_var, kernel_init, shape)

### Semi done
class Conv1d(nn.Module):
  """Conv3d layer with optimal upsampling and downsampling (StyleGAN2)."""
  fmaps: int
  kernel: int
  up: bool = False
  down: bool = False
  resample_kernel: Tuple[int] = (1, 3, 1)
  use_bias: bool = True
  weight_var: str = 'weight'
  kernel_init: Optional[Any] = None

  @nn.compact
  def __call__(self, x):
    assert not (self.up and self.down)
    assert self.kernel >= 1 and self.kernel % 2 == 1
    w = get_weight(self, (self.kernel, x.shape[-1], self.fmaps),
                   weight_var=self.weight_var,
                   kernel_init=self.kernel_init)
    if self.up:
      x = upsample_conv_1d(x, w, data_format='NHC', k=self.resample_kernel)
    elif self.down:
      x = conv_downsample_1d(x, w, data_format='NHC', k=self.resample_kernel)
    else:
      x = jax.lax.conv_general_dilated(
        x,
        w,
        window_strides=(1, 1),
        padding='SAME',
        dimension_numbers=('NHC', 'HWDIO', 'NHC')) ###???

    if self.use_bias:
      b = self.param('bias', jnn.initializers.zeros, (x.shape[-1],))
      x = x + b.reshape((1, 1, 1, 1, -1))
    return x


def naive_upsample_1d(x, factor=2):
  _N, H, C = x.shape
  x = jnp.reshape(x, [-1, H, 1, C])
  x = jnp.tile(x, [1, 1, factor, 1])
  return jnp.reshape(x, [-1, H * factor, C])


def naive_downsample_1d(x, factor=2):
  _N, H, C = x.shape
  x = jnp.reshape(x, [-1, H // factor, factor, C])
  return jnp.mean(x, axis=[2])


def upsample_conv_1d(x, w, k=None, factor=2, gain=1, data_format='NHC'):
  """Fused `upsample_3d()` followed by `tf.nn.conv3d()`.

     Padding is performed only once at the beginning, not between the
     operations.
     The fused op is considerably more efficient than performing the same
     calculation
     using standard TensorFlow ops. It supports gradients of arbitrary order.
     Args:
       x:            Input tensor of the shape `[N, C, H, W, D]` or `[N, H, W, D,
         C]`.
       w:            Weight tensor of the shape `[filterH, filterW, filterD, inChannels,
         outChannels]`. Grouped convolution can be performed by `inChannels =
         x.shape[0] // numGroups`.
       k:            FIR filter of the shape `[firH, firW, firD]` or `[firN]`
         (separable). The default is `[1] * factor`, which corresponds to
         nearest-neighbor upsampling.
       factor:       Integer upsampling factor (default: 2).
       gain:         Scaling factor for signal magnitude (default: 1.0).
       data_format:  `'NCHWD'` or `'NHWDC'` (default: `'NCHWD'`).

     Returns:
       Tensor of the shape `[N, C, H * factor, W * factor]` or
       `[N, H * factor, W * factor, C]`, and same datatype as `x`.
  """

  assert isinstance(factor, int) and factor >= 1

  # Check weight shape.
  assert len(w.shape) == 5
  convH = w.shape[0]
  inC = w.shape[3]
  outC = w.shape[4]
  assert convW == convH

  # Setup filter kernel.
  if k is None:
    k = [1] * factor
  k = _setup_kernel(k) * (gain * (factor ** 2))
  p = (k.shape[0] - factor) - (convW - 1)

  stride = [factor, factor, factor]
  # Determine data dimensions.
  if data_format == 'NCHWD':
    num_groups = _shape(x, 1) // inC
  else:
    num_groups = _shape(x, 4) // inC

  # Transpose weights.
  w = jnp.reshape(w, [convH, convW, convD, inC, num_groups, -1])
  w = jnp.transpose(w[::-1, ::-1, ::-1], [0, 1, 2, 5, 4, 3])
  w = jnp.reshape(w, [convH, convW, convD, -1, num_groups * inC])

  ## Original TF code.
  # x = tf.nn.conv2d_transpose(
  #     x,
  #     w,
  #     output_shape=output_shape,
  #     strides=stride,
  #     padding='VALID',
  #     data_format=data_format)
  ## JAX equivalent
  x = jax.lax.conv_transpose(
    x,
    w,
    strides=stride,
    padding='VALID',
    transpose_kernel=True,
    dimension_numbers=(data_format, 'HWDIO', data_format))

  return _simple_upfirdn_3d(
    x,
    k,
    pad0=(p + 1) // 2 + factor - 1,
    pad1=p // 2 + 1,
    data_format=data_format)


def conv_downsample_3d(x, w, k=None, factor=2, gain=1, data_format='NHWDC'):
  """Fused `tf.nn.conv3d()` followed by `downsample_3d()`.

    Padding is performed only once at the beginning, not between the operations.
    The fused op is considerably more efficient than performing the same
    calculation
    using standard TensorFlow ops. It supports gradients of arbitrary order.
    Args:
        x:            Input tensor of the shape `[N, C, H, W, D]` or `[N, H, W, D,
          C]`.
        w:            Weight tensor of the shape `[filterH, filterW, inChannels,
          outChannels]`. Grouped convolution can be performed by `inChannels =
          x.shape[0] // numGroups`.
        k:            FIR filter of the shape `[firH, firW]` or `[firN]`
          (separable). The default is `[1] * factor`, which corresponds to
          average pooling.
        factor:       Integer downsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).
        data_format:  `'NCHWD'` or `'NHWDC'` (default: `'NCHWD'`).

    Returns:
        Tensor of the shape `[N, C, H // factor, W // factor, D // factor]` or
        `[N, H // factor, W // factor, D // factor, C]`, and same datatype as `x`.
  """

  assert isinstance(factor, int) and factor >= 1
  convH, convW, convD, _inC, _outC = w.shape
  assert convW == convH
  if k is None:
    k = [1] * factor
  k = _setup_kernel(k) * gain
  p = (k.shape[0] - factor) + (convW - 1)
  s = [factor, factor, factor]
  x = _simple_upfirdn_2d(x, k, pad0=(p + 1) // 2,
                         pad1=p // 2, data_format=data_format)

  return jax.lax.conv_general_dilated(
    x,
    w,
    window_strides=s,
    padding='VALID',
    dimension_numbers=(data_format, 'HWIO', data_format))



def upfirdn_1d(x, k, up, down, pad0, pad1):
  """
  Pad, upsample, FIR filter, and downsample a batch of 1D signals.
  """
  k = jnp.asarray(k, dtype=np.float32)
  assert len(x.shape) == 3
  inH = x.shape[1]
  minorDim = x.shape[2]
  kernelH = k.shape[0]
  assert inH >= 1
  assert kernelH >= 1
  assert isinstance(up, int)
  assert isinstance(down, int)
  assert isinstance(pad0, int) and isinstance(pad1, int)
  
  # Upsample (insert zeros).
  x = jnp.reshape(x, (-1, inH, 1, minorDim))
  x = jnp.pad(x, [[0, 0], [0, 0], [0, up - 1], [0, 0]])
  x = jnp.reshape(x, [-1, inH * up, minorDim])

  # Pad (crop if negative).
  x = jnp.pad(x, [[0, 0], [max(pad0, 0), max(pad1, 0)], [0, 0]])
  x = x[:, max(-pad0, 0):x.shape[1] - max(-pad1, 0), :]

  # Convolve with filter.
  x = jnp.transpose(x, [0, 2, 1])
  x = jnp.reshape(x, [-1, 1, inH * up + pad0 + pad1])
  w = jnp.array(k[::-1, None, None], dtype=x.dtype)
  x = jax.lax.conv_general_dilated(
    x,
    w,
    window_strides=(1,),
    padding='VALID',
    dimension_numbers=('NCH', 'HIO', 'NCH'))

  x = jnp.reshape(x, [-1, minorDim, inH * up + pad0 + pad1 - kernelH + 1])
  x = jnp.transpose(x, [0, 2, 1])

  # Downsample (throw away pixels).
  return x[:, ::down, :]



def _simple_upfirdn_1d(x, k, up=1, down=1, pad0=0, pad1=0, data_format='NCH'):
  assert data_format in ['NCH', 'NHC']
  assert len(x.shape) == 3
  y = x
  if data_format == 'NCH':
    y = jnp.reshape(y, [-1, y.shape[1], 1])
  y = upfirdn_1d(
    y,
    k,
    up=up,
    down=down,
    pad0=pad0,
    pad1=pad1)
  if data_format == 'NCH':
    y = jnp.reshape(y, [-1, x.shape[1], y.shape[1]])
  return y


def _setup_kernel(k):
  k = np.asarray(k, dtype=np.float32)
  k /= np.sum(k)
  assert k.ndim == 1
  return k

def _shape(x, dim):
  return x.shape[dim]


def upsample_1d(x, k=None, factor=2, gain=1, data_format='NHC'):
  r"""Upsample a batch of 1D sequences with the given filter.

    Accepts a batch of 1D sequences of the shape `[N, C, H]` or `[N, H, C]`
    and upsamples each sequence with the given filter. The filter is normalized so
    that if the input values are constant, they will be scaled by the specified
    `gain`.
    Values outside the sequence are assumed to be zero, and the filter is padded
    with zeros so that its shape is a multiple of the upsampling factor.
    Args:
        x:            Input tensor of the shape `[N, C, H]` or `[N, H, C]`.
        k:            FIR filter of the shape `[firN]` (separable). The default is `[1] * factor`, which corresponds to nearest-neighbor upsampling.
        factor:       Integer upsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).
        data_format:  `'NCH'` or `'NHC'` (default: `'NCH'`).

    Returns:
        Tensor of the shape `[N, C, H * factor]` or `[N, H * factor, C]`, and same datatype as `x`.
  """
  assert isinstance(factor, int) and factor >= 1
  if k is None:
    k = [1] * factor
  k = _setup_kernel(k) * (gain * factor)
  p = k.shape[0] - factor
  return _simple_upfirdn_1d(
    x,
    k,
    up=factor,
    pad0=(p + 1) // 2 + factor - 1,
    pad1=p // 2,
    data_format=data_format)



def downsample_1d(x, k=None, factor=2, gain=1, data_format='NHC'):
  r"""Downsample a batch of 1D sequences with the given filter.

    Accepts a batch of 1D sequences of the shape `[N, C, H]` or `[N, H, C]`
    and downsamples each sequence with the given filter. The filter is normalized so
    that if the input values are constant, they will be scaled by the specified
    `gain`.
    Values outside the sequence are assumed to be zero, and the filter is padded
    with zeros so that its shape is a multiple of the downsampling factor.
    Args:
        x:            Input tensor of the shape `[N, C, H]` or `[N, H, C]`.
        k:            FIR filter of the shape `[firN]` (separable). The default is `[1] * factor`, which corresponds to average pooling.
        factor:       Integer downsampling factor (default: 2).
        gain:         Scaling factor for signal magnitude (default: 1.0).
        data_format:  `'NCH'` or `'NHC'` (default: `'NCH'`).

    Returns:
        Tensor of the shape `[N, C, H // factor]` or `[N, H // factor, C]`, and same datatype as `x`.
  """

  assert isinstance(factor, int) and factor >= 1
  if k is None:
    k = [1] * factor
  k = _setup_kernel(k) * gain
  p = k.shape[0] - factor
  return _simple_upfirdn_1d(
    x,
    k,
    down=factor,
    pad0=(p + 1) // 2,
    pad1=p // 2,
    data_format=data_format)