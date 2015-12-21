#!/usr/bin/python

# pann.py -- An implementation of a feedforward neural network in python.
#            Currently most useful for text modeling. 
#
#    Copyright 2014   by Jonathan Scott Enderle
#
5

import numpy
import tables
import scipy.optimize
import sys
import math
import string
import time
import argparse
import os
import itertools
import random
import collections
from matplotlib import pyplot
from textwrap import TextWrapper
from contextlib import closing

# TODO:

# I have thought about it and before I can do any of the below, I 
# need to do more to decouple training, testing, and application,
# or else I need to bite the bullet and accept that there's going 
# to be one gigantic object that has methods for all three. 
# In which case they should be absolutely bare-bones and produce
# output in a form that's easy to manipulate further. 

# 1) Implement plugins for visualization
# 2) Implement plugins for data input
# 3) Separate training loop from main
# 4) Add an `out` parameter to `unroll`
# 5) Store shape with Theta data, probably using H5. 

# In the long run -- optimization plug-ins? Network shape plug-ins?
# Reimplement numpy training data input?

# REFLECTIONS ON THE NETWORK INTERFACE

# Neural networks have a lot of complexity, and I think that hiding that
# complexity is important if the class used for training is to be used
# outside this module. With that in mind, I've implemented an experimental
# copying and "cloning" interface that allows for basic manipulation of shared
# state (network weights) in a way that doesn't demand a lot of bookkeeping
# in the calling module. Copies are straight-up copies; clones are copies
# with some shared state. The shared state consists only of network weights; 
# changes to the weights of one clone affect all clones. But training data
# is not shared. This means we can do complex things like have one clone
# training, and have another clone doing cross-validation without having to
# update state directly all the time. But it also means we don't have to 
# manage or know anything at all about how network weights are stored, and
# so on. That's all managed by the Network itself. We just pass it a shape
# and let it go. 

# There _is_ a public interface for accessing weights, but
# it's very simple -- just a flat array of weights, or a (write-only) 
# property that returns the weights separated into different matrices. This
# is useful for visualizing the weight matrices (which is something that
# we can expect the visualizing module to understand). But I don't expect
# the weight accessors to be very useful otherwise.

# I'm not certain this is the right solution, but it has a certain conceptual
# clarity that appeals to me, compared to the insane complexity of dealing
# with things like conversions between shapes that do or do not include
# bias units and so on. That stuff really shouldn't be exposed. I'll
# have to think more about the canonical representation of a network's
# _shape_ but that's going to have to evolve from here. 

class Network(object):
    '''A class for training feedforward networks of arbitrary sizes.
    Networks deeper than four layers are unlikely to train well; future
    versions of this program will include support for auto-encoders and
    other pre-trainers appropriate for deep learning problems.'''

    def __init__(self, shape, X=None, Y=None, logger=None):
        self._shape = shape
        self._weights = _Weights(shape)
        
        self._reg_lambda = 0
        self._n_samples = None
        self._avals = None
        self._zvals = None
        self._dvals = None
        self._Y = None
        
        if logger is None:
            self._logger = OutputLogger(dummy=True)
        else:
            self._logger = logger
        self._callback = self._logger.update_progress

        if X is not None and Y is not None:
            self.update_XY(X, Y)
        elif X is not None or Y is not None:
            raise ValueError("Both X and Y must be supplied, or neither.")

    def _duplicate(self, X, Y, logger):
        if X is None and Y is None:
            X = self.X
            Y = self.Y
        new = Network(self._shape, X, Y, logger)
        return new

    def copy(self, X=None, Y=None, logger=None):
        new = self._duplicate(X, Y, logger)
        new.theta = self.theta
        return new

    def clone(self, X=None, Y=None):
        new = self._duplicate(X, Y, logger)
        new._weights = self._weights
        return new

    def is_clone(self, other):
        return self._weights is other._weights

    @property
    def X(self):
        if self._avals is None:
            return None
        else:
            return self._avals[0][:,1:]

    @property
    def Y(self):
        return self._Y

    @property
    def n_samples(self):
        return self._n_samples

    @property
    def theta(self):
        return self._weights.theta

    @theta.setter
    def theta(self, theta):
        self._weights.theta = theta

    @property
    def theta_list(self):
        return self._weights.theta_list

    def load_theta(self, path):
        self.theta = numpy.load(path)

    def save_theta(self, path):
        numpy.save(path, self.theta)

    def update_XY(self, X, Y):
        self._last_theta = None     # This is a crucial step for correctness 
        # for reasons that aren't obvious based on the name or the code. I 
        # don't like that. The issue is that if this value isn't reset to 
        # None at this point, then it's possible for backprop to run on the 
        # values from foward propagation from the _previous_ set of XY 
        # values. Unfortunately I can't think of a better way to manage this. 

        if (not X.shape[1] == self._shape[0] or 
            not Y.shape[1] == self._shape[-1]):
            raise ValueError("Network shape and training data are not aligned.")
        if not X.shape[0] == Y.shape[0]:
            raise ValueError("Number of inputs does not match number of outputs in training data.")
 
        n_samples = float(X.shape[0])
        if self._n_samples == n_samples:
            for a in self._avals:
                a[:] = 1
            for z in self._zvals:
                z[:] = 1
            for d in self._dvals:
                d[:] = 1
        else:
            self._n_samples = n_samples
            ashapes, zshapes = self._weights.azshapes(n_samples)
            self._avals = [numpy.ones(s) for s in ashapes]
            self._zvals = [numpy.ones(s) for s in zshapes]
            self._dvals = [numpy.ones(s) for s in zshapes]
        
        # first layer of activation layers -- X with additional bias unit
        self._avals[0][:,1:] = X
        self._Y = Y

    def train_cg(self, iters, reg):
        if self._n_samples is None:
            return
        
        if self._logger is not None:
            self._logger.reset_progress()

        self._reg_lambda = float(reg)
        theta = scipy.optimize.fmin_cg(self._cost, 
                                       self.theta, 
                                       self._gradient, 
                                       maxiter=iters,
                                       callback=self._callback,
                                       disp=False)
        self.theta = theta

    def train_sgd(self, iters, reg, alpha):
        if self._n_samples is None:
            return

        if self._logger is not None:
            self._logger.reset_progress()

        theta = self.theta
        self._reg_lambda = float(reg)

        for _ in xrange(iters):
            self._cost(theta)
            grad = self._gradient(theta)
            grad /= (grad * grad).sum() ** 0.5
            theta -= alpha * grad
            self._callback()
        
        self.theta = theta

    def predict(self):
        self._forward_propagation()
        return self._avals[-1]

    # Consider moving this accuracy stuff into a new class; the class 
    # interface is just about the right size here; the below methods 
    # expand it a bit too much.

    def accuracy(self):
        self._forward_propagation()
        h = self._avals[-1]
        y = self.Y.argmax(axis=1)
        p = h.argmax(axis=1)
        return float((p == y).sum()) / y.shape[0]

    def accuracy_topn(self, *n):
        n = set(n)
        self._forward_propagation()
        h = self._avals[-1].copy()
        y = self.Y.argmax(axis=1)
        
        total = 0
        results = ()
        for i in range(max(n)):
            p = h.argmax(axis=1)
            total += ((p == y).sum())

            if i + 1 in n:
                results += (float(total) / y.shape[0],)

            h[numpy.arange(h.shape[0]),p] = 0.0

        return results

    def entropy(self):
        self._forward_propagation()
        h = self._avals[-1]
        probs = h[self.Y.nonzero()]
        return -(numpy.log2(probs)).sum() / self.Y.shape[0]

    # The following is really only ever for testing, but it's quite important
    # so I'm leaving it as part of the public interface for now. 

    def check_gradient_sample(self, theta, n_trials, eps=10 ** -4):
        if self._n_samples is None:
            return
        
        grad = self._gradient(theta)
        theta_eps = theta[:]

        samples = numpy.random.randint(0, theta.size, n_trials)
        grad_err = []
        for s in samples:
            theta_eps[s] += eps
            grad_s = self._cost(theta_eps)
            theta_eps[s] -= 2 * eps
            grad_s -= self._cost(theta_eps)
            grad_s /= 2 * eps
            yield grad[s], grad_s, numpy.fabs(grad_s - grad[s])

    @staticmethod
    def _sigmoid(x, _exp=numpy.exp):
        ex = _exp(-x)
        ex += 1
        return 1 / ex

    @staticmethod
    def _sigmoid_grad(x, _exp=numpy.exp):
        ex = _exp(-x)
        ex_1 = ex + 1
        ex /= ex_1 * ex_1
        return ex

    def _forward_propagation(self):
        thetas = self.theta_list
        avals = self._avals
        zvals = self._zvals

        a_in = avals[0]
        mid_layers = zip(avals[1:], zvals, thetas)
        out_layer = mid_layers.pop()

        for a, z, t in mid_layers:
            numpy.dot(a_in, t.T, out=z)
            a[:,1:] = self._sigmoid(z)
            a_in = a
        
        a, z, t = out_layer
        z[:] = numpy.dot(a_in, t.T)
        a[:] = self._sigmoid(z)

    def _cost(self, theta, _log=numpy.log):
        self.theta = theta
        self._last_theta = theta
        
        thetas = self.theta_list
        n_samples = self._n_samples
        
        self._forward_propagation()

        h = self._avals[-1]
        Y = -self.Y

        cost = Y * _log(h) - (1 + Y) * _log(1 - h)
        cost = cost.sum() / n_samples
        
        cost += self._cost_reg(thetas, n_samples)

        if self._logger is not None:
            self._logger.update_cost(cost)
        return cost
 
    def _cost_reg(self, thetas, n_samples):
        cost = 0
        reg_factor = self._reg_lambda / (2 * n_samples)
        for t in thetas:
            t2 = t * t
            t2[:,0] = 0
            cost += t2.sum() * reg_factor
        return cost

    def _is_last_theta(self, theta):
        if self._last_theta is None:
            return False
        elif theta[0] != self._last_theta[0]:
            return False
        elif (theta == self._last_theta).all():
            return True
        else:
            print "First theta vals matched, but not _is_last_theta!"
            print "This should be an extremely rare event. It can"
            print "safely be ignored. However, feel free to contact"
            print "scott.enderle@gmail.com if it happens to you. He"
            print "wants to know how often it happens in practice." 
            return False
           
    def _gradient(self, theta):
        # Calculating the gradient requires information produced when the 
        # cost is calculated. Unfortunately, fmin_cg treats the two as 
        # separate functions, so a straightforward calculation of the
        # gradient requires us to duplicate effort. Instead, we reuse
        # information from the cost calculation when theta is the same.
        if not self._is_last_theta(theta):
            self.theta = theta
            self._forward_propagation()
        
        avals = self._avals
        zvals = self._zvals
        dvals = self._dvals
        
        thetas = self.theta_list
        n_samples = self._n_samples

        dvals[-1][:] = avals[-1]
        dvals[-1] -= self.Y

        # propagate error terms in reverse
        t_z_dprev_dnext = zip(thetas[1:], zvals, dvals, dvals[1:])
        for t, z, dprev, dnext in reversed(t_z_dprev_dnext):
            numpy.dot(dnext, t[:,1:], out=dprev)
            dprev *= self._sigmoid_grad(z)

        theta_grads = [numpy.dot(d.T, a) / n_samples
                       for d, a in zip(dvals, avals)]

        self._grad_reg(theta_grads, thetas, n_samples)

        return self._weights.unroll(theta_grads)

    def _grad_reg(self, theta_grads, thetas, n_samples):
        for g, t in zip(theta_grads, thetas):
            t_tail = self._reg_lambda / n_samples * t
            t_tail[:,0] = 0
            g += t_tail

class _Weights(object):
    def __init__(self, shape):
        shape = tuple(shape)
        shapes = []
        ranges = []
        size = 0
        for cols, rows in zip(shape, shape[1:]):
            cols += 1                               # add room for the bias unit
            shapes.append((rows, cols))
            ranges.append((size, size + rows * cols))
            size += rows * cols
        
        self._shape = (shape[0] + 1,) + shape[1:]
        self._theta_shapes = shapes
        self._theta_ranges = ranges
        self._theta_size = size
        
        self._theta = numpy.random.normal(0, 0.05, self._theta_size)

    @property
    def theta_list(self):
        return self.rollup(self._theta)

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        self._theta[:] = theta

    @property
    def shape(self):
        return self._shape

    def azshapes(self, n_samples):
        '''Given a number of samples for training, determine the shape
        of the activation and z matrices. This is useful for preallocating
        and reusing memory for these matrices, which can be quite large.'''
        shape = self._shape

        ashapes = [(n_samples, shape[0])]
        # inner layers have additional bias units
        ashapes.extend((n_samples, s + 1) for s in shape[1:-1])
        ashapes.append((n_samples, shape[-1]))
        
        zshapes = [(n_samples, s) for s in shape[1:]]
        
        return ashapes, zshapes

    def rollup(self, theta):
        '''Create a list of theta matrices containing the weights between
        each of the network layers. This returns views into the data from
        `theta`; it does not create copies.'''
        thetas = []
        for (rows, cols), (start, end) in zip(self._theta_shapes, self._theta_ranges):
            thetas.append(theta[start:end].reshape(rows, cols))
        return thetas

    def unroll(self, thetas):
        '''Allocate a new array and fill it with data from each of the 
        weight matrices in the list `thetas`. Unlike `rollup`, this does 
        create a copy of the data.'''  # TODO: add an `out` parameter

        theta = numpy.empty(self._theta_size)
        for t, (start, end) in zip(thetas, self._theta_ranges):
            theta[start:end] = t.ravel()
        return theta

class OutputLogger(object):
    def __init__(self, message=None, dummy=False):
        self.message = message
        self.dummy = dummy
        
        self.custom = collections.defaultdict(str)
        self.reset_progress()

    def reset_progress(self):
        if self.dummy:
            return
        self.iteration = 0
        self.cost = 0
        self.elapsed_time = 0
        self.last_time = time.clock()
        self.total_time = 0

    def update_progress(self, *args, **kwargs):
        if self.dummy:
            return
        self.iteration += 1
        now = time.clock()
        self.elapsed_time = now - self.last_time
        self.total_time += self.elapsed_time
        self.last_time = now
        self.display_progress()

    def update_cost(self, cost):
        self.cost = cost

    def display_progress(self):
        message = self.message.format(std=self, **self.custom)
        self._overwrite_last(message)

    def update_custom(self, **args):
        self.custom.update(args)

    def _overwrite_last(self, message):
        message = '\r' + message
        sys.stdout.write(message)
        sys.stdout.flush()

######## Not ready for use in master ########
class AdaptiveAlpha(object):
    def __init__(self, cv_network, weight_decay, max_samples=None):
        self.cv_network = cv_network
        if max_samples is None:
            self.max_samples = int(math.log(2 ** 16, weight_decay))
            self.min_samples = int(math.log(5, weight_decay))
        else:
            self.max_samples = max_samples
            self.min_samples = self.max_samples / 4
        self.weight_decay = weight_decay
        self.samples = collections.deque(maxlen=self.max_samples) 
        
        bins = numpy.exp(numpy.linspace(0.01, 4, 400)) - 1
        self.bins = numpy.around(bins, 2)

    def sample(self, current_alpha, timestamp):
        a_ix = numpy.searchsorted(self.bins, current_alpha)
        alpha = self.bins[a_ix]
        self.samples.append((alpha, self.cv_network.entropy(), timestamp))

    def sum_samples(self):
        alpha_sums = {}
        min_timestamp = min(t for _1, _2, t in self.samples)
        for alpha, entropy, timestamp in self.samples:
            weight = 1.0 / (self.weight_decay ** (timestamp - min_timestamp))

            if alpha in alpha_sums:
                a, w = alpha_sums[alpha]
                alpha_sums[alpha] = (a + entropy * weight, w + weight)
            else:
                alpha_sums[alpha] = (entropy * weight, weight)

        for a in alpha_sums:
            e, w = alpha_sums[a]
            alpha_sums[a] = (float(e) / w, w)
        
        return alpha_sums

    def optimize_alpha(self, alpha, show_plot=False):
        if len(self.samples) < self.min_samples:
            return alpha
        polyfit = numpy.polynomial.polynomial.polyfit 
        polyval = numpy.polynomial.polynomial.polyval

        alpha_sums = self.sum_samples()
        alphas, sums_weights = zip(*alpha_sums.iteritems())
        sums, weights = zip(*sums_weights)
        alphas = numpy.asarray(alphas)
        sums = numpy.asarray(sums)
        weights = numpy.asarray(weights)

        fit_params = polyfit(alphas, sums, 2, w=weights)
        sum_fit = polyval(self.bins, fit_params)

        alpha_ix = numpy.searchsorted(self.bins, alpha)
        min_ix = sum_fit.argmin()
        if sum_fit[min_ix] < sum_fit[alpha_ix]:
            if min_ix < alpha_ix:
                alpha_ix -= 1
            elif min_ix > alpha_ix and fit_params[-1] > 0: # don't increase
                alpha_ix += 1                              # except towards a
                                                           # global minimum
            alpha = self.bins[alpha_ix]
            
        if show_plot:
            self.gen_plot(alphas, sums, weights, fit_params)

        return alpha

    def gen_plot(self, alphas, sums, weights, fit_params):
        polyval = numpy.polynomial.polynomial.polyval
        sort_ix = alphas.argsort()
        alphas = alphas[sort_ix]
        sums = sums[sort_ix]
        err = sums - polyval(alphas, fit_params)
        window_size = int(len(err) ** 0.5)
        err_size = err.size

        err_windows = self.rolling_window(err, window_size)
        sigma_raw = numpy.std(err_windows, axis=1, ddof=1)
        sigma = numpy.empty_like(err)
        sigma[:] = sigma_raw.max()
        sigma_raw_size = sigma_raw.size
        pad = (sigma.size - sigma_raw_size) / 2
        sigma[pad:pad + sigma_raw_size] = sigma_raw

        x_fit = alphas
        y_fit = polyval(x_fit, fit_params)
        pyplot.scatter(alphas, sums, s=weights * 3, color='b', alpha=0.5)
        pyplot.plot(x_fit, y_fit, 'k')
        pyplot.fill_between(x_fit, y_fit - sigma, y_fit + sigma, 
                            color='b', alpha=0.2)
        pyplot.savefig('foo{}.svg'.format(str(hash(str(alphas + sums)))), fmt='svg')
        pyplot.clf()

    def rolling_window(self, array, size):
        shape = array.shape[:-1] + (array.shape[-1] - size + 1, size)
        strides = array.strides + (array.strides[-1],)
        ast = numpy.lib.stride_tricks.as_strided
        return ast(array, shape=shape, strides=strides)

    def corr_coef(self, a, b):
        a = numpy.asarray(a)
        b = numpy.asarray(b)
        a_mean = a.mean()
        b_mean = b.mean()
        a -= a_mean
        b -= b_mean
        n = a.size
        a_std = (numpy.dot(a, a) / (n - 1)) ** 0.5
        b_std = (numpy.dot(b, b) / (n - 1)) ** 0.5
        return numpy.dot(a, b) / (a_std * b_std * (n - 1))



# TODO: Replace this kludge with a proper plugin system that would allow
# people to write their own markov visualizers or theta visualizers.

# I need a well-defined interface for this though. That's difficult
# because the visualizer needs to know a bunch of stuff that the
# trainer / training cycle doesn't (and shouldn't) know anything
# about. It needs to know what the values _mean_ in context, and 
# that's already pretty complicated here in this one-off code. 

# At this point, my plan is to have an improved `trainer` interface
# that visualizer plugins can rely on.

def get_next_selector(h, shift_ix=None, sample=True):
    if shift_ix is not None:
        cap = h[shift_ix] > 0.5
        h[shift_ix] = 0
    else:
        cap = False

    #h = h ** 2  # to weed out the less common letters
    #h = h ** 1.5  # to weed out the less common letters
    #h = h ** 1.3  # to weed out the less common letters
    h = h ** 1.2  # to weed out the less common letters
    if sample:
        selection = numpy.random.multinomial(1, h / h.sum())
    else:
        selection = h.argmax() == numpy.arange(h.shape[0])

    if shift_ix is not None:
        selection[shift_ix] = cap
    
    return selection

def markov_visualizer(network, n_cycles=100):
    n_features = network.theta_list[0].shape[1] - 1
    n_labels = network.theta_list[-1].shape[0]

    # key is currently hard-coded... yuck
    # key = ' ' + string.ascii_lowercase + string.ascii_uppercase
    key = ''' !"'(),-.:;?_abcdefghijklmnopqrstuvwxyzX'''
    shift_ix = -1
    inv_key = {c:n for n, c in enumerate(key)}
    start = network.X[0:1,:]
    fakeY = numpy.asarray([[1]]) == numpy.arange(n_labels)

    n = network.clone(start, fakeY)
    for i in range(n_cycles):
        s = start[0].reshape(-1, n_labels)

        if shift_ix is not None:
            caps = s[...,shift_ix]
        else:
            caps = [0] * s.shape[0]

        try:
            chars = []
            for char in s:
                nonzero_tuple = char.nonzero()
                nonzero_c = nonzero_tuple[0]
                c_ix = nonzero_c[0]
                chars.append(key[c_ix])

        except IndexError:
            print "You've discovered the heisenbug. Please send the"
            print "following information to scott.enderle@gmail.com:"
            print (s.ravel() != 0).sum()
            print s.shape
            print 
            break
        
        #chars = (key[char.nonzero()[0][0]] for char in s)
        
        print ''.join(c.upper() if cap else c for c, cap in zip(chars, caps))

        r = get_next_selector(n.predict()[0], shift_ix)
        start[:,0:n_features - n_labels] = start[:,n_labels:]
        start[:,n_features - n_labels:] = r
        n.update_XY(start, fakeY)

# TODO: _This_ should be implemented with one class for each type
# of data input; this will be the easiest thing to re-write as a 
# set of plugins. 

class TrainingData(object):
    def __init__(self, h5_data, batch_size, chunk_bytes=5 * 10 ** 8):  
        self.generator = lambda: iter(())
        self.in_size = None
        self.out_size = None
        self.chunk_size = None
        self.chunk_bytes = chunk_bytes
        self.batch_size = batch_size
        self.load_h5_table(h5_data)

    def __iter__(self):
        return self.generator()

    def _h5_fast_bool_ix(self, h5_array, ix):
        '''Iterate over an h5 array chunkwise to select a random subset
        of the array. `h5_array` should be the array itself; `ix` should
        be a boolean index array with as many values as `h5_array` has
        rows, and you can optionally set the number of rows to read per
        chunk with `read_chunksize` (default is 30000). For some reason
        this is much faster than using `ix` to index the array directly.'''

        # As time goes on, this solution looks worse and worse.
        # I want to try using h5py instead. 

        read_chunksize = self.chunk_size

        n_rows = h5_array.shape[0]
        n_chunks = n_rows / read_chunksize
        if n_chunks * read_chunksize < n_rows:
            n_chunks += 1
        
        slices = [slice(i * read_chunksize, (i + 1) * read_chunksize)
                  for i in range(n_chunks)]

        a = numpy.zeros((ix.sum(), h5_array.shape[1]), dtype=float)
        a_start = 0
        for sl in slices:
            chunk = h5_array[sl][ix[sl]]
            a_end = a_start + chunk.shape[0]
            a[a_start:a_end] = chunk
            a_start = a_end

        fail = (a.sum(axis=1) == 0).nonzero()
        if fail[0].size > 0:
            print fail

        return a

    # just to supply some intutition when I come back to this and feel 
    # completely baffled by what I have written. 
    # a = x[ix]
    # inv_ix[ix] = numpy.arange(10)
    # b[inv_ix] = x
    # (a == b).all() is True

    #def _h5_fast_ix(self, h5_array, ix, read_chunksize=30000):

    #    n_chunks = h5_array.shape[0] / read_chunksize
    #    slices = [slice(i * read_chunksize, (i + 1) * read_chunksize)
    #              for i in range(n_chunks)]

    #    bool_ix = numpy.zeros(h5_array.shape[0], dtype=bool)  
    #    bool_ix[ix] = 1                                      
    #    order_ix = numpy.zeros(h5_array.shape[0], dtype=int)
    #    order_ix[ix] = numpy.arange(ix.shape[0])

    #    a = numpy.empty((ix.shape[0], h5_array.shape[1]), dtype=float)
    #    for sl in slices:
    #        h5_chunk = h5_array[sl]
    #        bool_chunk = bool_ix[sl]
    #        order_chunk = order_ix[sl]
    #        a[order_chunk[bool_chunk]] = h5_chunk[bool_chunk]

    #    return a

    def _h5_gen(self, h5_file):
        with closing(tables.open_file(h5_file, 'r')) as f:
            in_rows = f.root.input.shape[0]
        
        chunk_size = self.chunk_size

        n_chunks = in_rows / chunk_size
        if n_chunks * chunk_size < in_rows:
            n_chunks += 1

        def data_iter():
            with closing(tables.open_file(h5_file, 'r')) as f:
                h5_in = f.root.input
                h5_out = f.root.output
                mask = numpy.arange(in_rows) % n_chunks
                numpy.random.shuffle(mask)
               
                ix_func = self._h5_fast_bool_ix
                for chunk in xrange(n_chunks):
                    bool_mask = mask == chunk
                    order = numpy.arange(bool_mask.sum())
                    numpy.random.shuffle(order)
                    
                    in_arr  = ix_func(h5_in,  bool_mask)[order]
                    out_arr = ix_func(h5_out, bool_mask)[order]
                   
                    # nesting is too deep here. break this out.
                    for batch in xrange(0, in_arr.shape[0], self.batch_size):
                        batch_slice = slice(batch, batch + self.batch_size)
                        name_params = (h5_file, chunk, batch / self.batch_size)
                        in_name = '{} input chunk {}, batch {}'
                        in_name = in_name.format(*name_params)
                        out_name = '{} output chunk {}, batch {}'
                        out_name = out_name.format(*name_params)
                        output = (in_arr[batch_slice], in_name, 
                                  out_arr[batch_slice], out_name)
                        if batch < len(in_arr):
                            yield output
                        else:
                            print "This should never happen."
        
        return data_iter

    def load_h5_table(self, h5_file):
        with closing(tables.open_file(h5_file, 'r')) as f:
            self.in_size = f.root.input.shape[1]
            self.out_size = f.root.output.shape[1]
        
            # Loading from h5 tables is slow, so load in chunks, and then
            # yield in smaller batches (if batch_size is smaller).
            chunk_size = self.chunk_bytes / (self.in_size * 8)
            
            # Chunks should be congruent with batches
            if chunk_size > self.batch_size:
                self.chunk_size = chunk_size - chunk_size % self.batch_size
            else:
                self.chunk_size = self.batch_size
        
        self.generator = self._h5_gen(h5_file)

    def shape_weights(self, layers, gap_size=0):
        in_size = self.in_size
        out_size = self.out_size

        if layers < 2:
            raise ValueError("Cannot produce a 1-layer neural network!")
        elif layers == 2:
            return (in_size, out_size)
    
        if gap_size < 0:
            raise ValueError("Cannot have an initial gap of less than one.")

        scale = 1.0 / (layers + gap_size - 1)
        log_fraction = math.log(out_size, in_size) ** (scale)
    
        second_layer = in_size ** log_fraction
        for i in range(gap_size):
            second_layer = second_layer ** log_fraction

        shape = (in_size, int(second_layer))
        for i in range(layers - 3):
            shape += (int(shape[-1] ** log_fraction),)
        shape += (out_size,)
    
        return shape

def training_loop():
    pass
    # TODO: Move training loop here?

def build_parser():
    nn_parser = argparse.ArgumentParser(description='Feedforward Neural Network.')
    nn_parser.add_argument('-I', '--training-input', type=str, required=True,
        metavar='h5_file', help=('Path to an h5 file containing '
        'training data. Input and output data should be stored in arrays '
        '`root.input` and `root.output`.')) 
    nn_parser.add_argument('-C', '--cv-input', type=str, metavar='h5_file',
        help=('Path to an h5 file containing cross-validation data. Input '
        'and output data should be stored in arrays `root.input` and '
        '`root.output`.'))

    shape_group = nn_parser.add_mutually_exclusive_group(required=True)
    shape_group.add_argument('-L', '--num-layers', metavar='integer', 
        type=int, help=('Number of layers. If this option is chosen, '
        'the sizes of the layers will be automatically determined using '
        'input and output data and a mid-layer size heuristic. Mutually '
        'exclusive with the --shape option.'))
    shape_group.add_argument('-s', '--shape', metavar='layer_size', 
        nargs='+', type=int, help=('Shape of network specified as a '
        'list of layer sizes, starting with the input layer. Mutually '
        'exclusive with the --num-layers option.'))

    # TODO There needs to be a way to store shape size alongside theta; 
    # it is clunky to require all these values to match. Once that's
    # implemented, this will be mutually exclusive with --num-layers. 
    # In the long run there should be separate training, testing, and
    # prediction commands.

    nn_parser.add_argument('-T', '--theta', metavar='file', 
        help=('Path to a numpy-readable file containing the weights '
        'of a trained network, represented as a flattened array. The shape '
        'should match that passed to --shape if used as well as the shape '
        'of -X and -Y.'))
    nn_parser.add_argument('-S', '--save-theta', metavar='file', 
        help=('Path to save the current theta values on exit. '
        'CAUTION: This currently does nothing to prevent you from '
        'overwriting a file, nor does it check that the save location '
        'exists.'))

    optimizer_group = nn_parser.add_mutually_exclusive_group()
    optimizer_group.add_argument('-g', '--stochastic-gradient-descent',
        type=float, default=0.3, const=0.3, nargs='?', 
        metavar='learning_rate', help=('Use a simple '
        'stochastic gradient descent algorithm for training. This is the '
        'default. Accepts an optional argument specifying the learning rate '
        'alpha. If the argument is not present, or if no alternative '
        'training algorithm is selected, then alpha defaults to 0.3.'))
    optimizer_group.add_argument('-c', '--conjugate-gradient', 
        action='store_true', help=('Use a conjugate gradient algorithm for '
        'training.'))

    nn_parser.add_argument('-b', '--batch-size', metavar='integer', type=int,
        default=40000, help=('Number of samples to load per training batch. '
        'Defaults to 40000.'))
    nn_parser.add_argument('-B', '--cv-batch-size', metavar='integer',
        type=int, default=10000, help=('Number of samples to load for '
        'cross-validation. Defaults to 15000. Ignored if --cv-input '
        'is not supplied.'))
    nn_parser.add_argument('-i', '--num-iterations', metavar='integer', 
        default=-1, type=int, help=('Number of training iterations. '
        'Defaults to 0, in which case a prediction task is assumed.'))
    nn_parser.add_argument('-n', '--num-cycles', metavar='integer', type=int,
        default=1, help=('Number of training cycles. If this option '
        'is selected, the trainer will cycle over the entire dataset '
        'multiple times; the total number of training iterations will then '
        'be num_iterations x num_cycles.'))
    
    nn_parser.add_argument('-o', '--vis-interval', 
        metavar='integer', type=int, default=1, help=('Number of batches to '
        'process before generating a visualization. Defaults to 1.'))
    nn_parser.add_argument('-r', '--regularization', metavar='float', 
        default=0, type=float, help=('Regularization factor. '
        'Defaults to 0.'))
    nn_parser.add_argument('-v', '--visualizer', metavar='mode', type=str, 
        choices=['markov'], help=('Choose '
        'visualization mode. Only `markov` is currently '
        'impemented (for character prediction).'))
    nn_parser.add_argument('--check-gradient', metavar='integer', type=int, 
        default=False, help=('Run a diagnostic test on a random '
        'sample of gradient values to confirm that backpropagation is '
        'correctly implemented.'))

    return nn_parser

if __name__ == '__main__':

    args = build_parser().parse_args()

    training_data = TrainingData(args.training_input, args.batch_size)
    if args.num_layers is not None:
        shape = training_data.shape_weights(args.num_layers, gap_size=0)
    else:
        shape = args.shape
    print "Network Shape:", tuple(shape)
    

    # The logic here for logging is terrible. I need to refactor this 
    # part of the code. 

    if args.conjugate_gradient:
        message = ('Cycle: {cycle}, Batch: {batch}, Iteration: {std.iteration} | '
                   'Cost: {std.cost:6.4f} | Elapsed: '
                   '{std.elapsed_time:4.2f}s, {std.total_time:4.2f}s (total)')
    else:
        message = ('Cycle: {cycle}, Batch: {batch}, Iteration: {std.iteration} | '
                   'Cost: {std.cost:6.4f} | Alpha: {alpha:4.3f} | Elapsed: '
                   '{std.elapsed_time:4.2f}s, {std.total_time:4.2f}s (total)')
        alpha = args.stochastic_gradient_descent
    
    logger = OutputLogger(message=message)
    nn = Network(shape, logger=logger)
    if args.theta is not None:
        print "Loading Theta..."
        nn.load_theta(args.theta)

    # Currently, cross-validation output is desgined to help understand how
    # each training batch affects the overall performance of the network. As
    # such, this only uses a small portion of the CV data. A fuller CV test
    # must be done after a lot of training to ensure that irregularities in
    # the selected CV data haven't masked over or underfitting. That CV test
    # can involve simply passing in the CV data as plain training data with
    # the number of training iterations set to zero. A better solution for
    # cross-validation needs to be implemented eventually, but this is OK for
    # now. 

    if args.num_iterations < 1:
        print
        print "Assuming a prediction task. If you want to train a new"
        print "network, you'll need to set the `--num-iterations` option"
        print "to a value greater than 0."
        print

    if args.cv_input:
        print "Loading CV Data..."
        cv_batchsize = int(float(args.cv_batch_size) / 3) + 1
        cv_data = TrainingData(args.cv_input, cv_batchsize)
        X_cv, _, Y_cv, _ = iter(cv_data).next()
        cv = nn.clone(X_cv, Y_cv)

    print "Loading Initial Training Data..."
    print
    for cycle in range(args.num_cycles):
        for batch, (X, xname, Y, yname) in enumerate(training_data):
            if args.conjugate_gradient:
                logger.update_custom(cycle=cycle, batch=batch)
            else:
                logger.update_custom(cycle=cycle, batch=batch, alpha=alpha)
            nn.update_XY(X, Y)

            if args.check_gradient:
                msg = ('Calculated Gradient: {}; '
                       'Estimated Gradient: {}; '
                       'Error: {}')
                print "Checking Gradient Calculation..."
                grad_est_err = nn.check_gradient_sample(nn.theta, args.check_gradient)
                for grad, est, err in grad_est_err:
                    print msg.format(grad, est, err)

            if args.num_iterations > 0:
                if args.conjugate_gradient:
                    nn.train_cg(args.num_iterations, args.regularization)
                else:
                    nn.train_sgd(args.num_iterations, 
                                 args.regularization, 
                                 args.stochastic_gradient_descent)

            if (batch + 1) % args.vis_interval == 0:
                a1, a3, a5 = nn.accuracy_topn(1, 3, 5)
                print
                print
                print "Training accuracy:        ", a1
                print "Training accuracy (top 3):", a3
                print "Training accuracy (top 5):", a5
                print "Training entropy:         ", nn.entropy()

                if args.cv_input:
                    a1, a3, a5 = cv.accuracy_topn(1, 3, 5)
                    print
                    print "CV accuracy:              ", a1
                    print "CV accuracy (top 3):      ", a3
                    print "CV accuracy (top 5):      ", a5
                    print "CV entropy:               ", cv.entropy()

                    res = cv.predict()
                    print
                    print "Some predicted results:", res[0:20,:].argmax(axis=1)
                    print "The actual results:    ", Y_cv[0:20,:].argmax(axis=1)

                # TODO: Replace with vis plug-in interface here.   
                #key = ' ' + string.ascii_lowercase 
                
                if args.visualizer == 'markov':
                    print
                    markov_visualizer(nn)
                    print

        if args.save_theta is not None:
            tmpname = '.temp_' + args.save_theta
            nn.save_theta(tmpname)

    if args.save_theta is not None:
        if not args.save_theta.endswith('.npy'):
            savename = args.save_theta + '.npy'
        else:
            savename = args.save_theta

        try:
            os.rename('.temp_' + savename, savename)
        except OSError:
            nn.save_theta(savename)
    else:
        nn.save_theta('tmp_theta_' + '_'.join(nn.shape))
