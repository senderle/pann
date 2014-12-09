#!/usr/bin/python

# pann.py -- An implementation of a feedforward neural network in python.
#            Currently most useful for text modeling. 
#
#    Copyright 2014   by Jonathan Scott Enderle
#

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

import numpy
import tables
import scipy.optimize
import sys
import math
import string
import time
import argparse
import os.path
import itertools
import random
from textwrap import TextWrapper

class NetworkTrainer(object):
    '''A class for training feedforward networks of arbitrary sizes.
    Networks deeper than four layers are unlikely to train well; future
    versions of this program will include support for auto-encoders and
    other pre-trainers appropriate for deep learning problems.'''

    def __init__(self, network, X, Y):
        
        if not X.shape[1] + 1 == network.shape[0] or not Y.shape[1] == network.shape[-1]:
            raise ValueError("Network shape and training data are not aligned.")
        if not X.shape[0] == Y.shape[0]:
            raise ValueError("Number of inputs does not match number of outputs in training data.")
        
        self.network = network
        self.reg_lambda = 1
        self.update_XY(X, Y)
        self._last_theta = None
        self.progress_msg = ProgressMessage()

    def update_XY(self, X, Y):
        self._n_samples = n_samples = float(X.shape[0])
        
        ashapes, zshapes = self.network.azshapes(n_samples)
        self._avals = [numpy.ones(s) for s in ashapes]
        self._zvals = [numpy.ones(s) for s in zshapes]
        self._dvals = [numpy.ones(s) for s in zshapes]

        # first layer of activation layers -- X with additional bias unit
        self._avals[0][:,1:] = X
        self._Y = Y

    @property
    def X(self):
        return self._avals[0][:,1:]

    @property
    def Y(self):
        return self._Y

    @property
    def n_samples(self):
        return self._n_samples

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
        thetas = self.network.theta_list
        avals = self._avals
        zvals = self._zvals

        a_in = avals[0]
        mid_layers = zip(avals[1:], zvals, thetas)
        out_layer = mid_layers.pop()

        for a, z, t in mid_layers:
            z[:] = numpy.dot(a_in, t.T)
            a[:,1:] = self._sigmoid(z)
            a_in = a
        
        a, z, t = out_layer
        z[:] = numpy.dot(a_in, t.T)
        a[:] = self._sigmoid(z)

    def cost(self, theta, _log=numpy.log):
        self.network.theta = theta
        self._last_theta = theta
        
        thetas = self.network.theta_list
        n_samples = self.n_samples
        
        self._forward_propagation()

        h = self._avals[-1]
        Y = -self.Y

        cost = Y * _log(h) - (1 + Y) * _log(1 - h)
        cost = cost.sum() / n_samples

        reg_factor = self.reg_lambda / (2 * n_samples)
        for t in thetas:
            t2 = t * t
            t2[:,0] = 0
            cost += t2.sum() * reg_factor

        self.progress_msg.cost = cost
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


#    def gradient(self, theta):
#        if not self._is_last_theta(theta):
#            self.network.theta = theta
#            self._forward_propagation()
#        avals = self._avals
#        zvals = self._zvals
#
#        thetas = self.network.theta_list
#        n_samples = self.n_samples
#
#        d = avals[-1] - self.Y
#        dvals = [d]
#
#        for t, z in reversed(zip(thetas[1:], zvals)):
#            d = numpy.dot(d, t)[:,1:]
#            d *= self._sigmoid_grad(z)
#            dvals.append(d)
#
#        dvals.reverse()
#
#        theta_grads = [numpy.dot(d.T, a) / n_samples
#                       for d, a in zip(dvals, avals)]
#
#        for g, t in zip(theta_grads, thetas):
#            t_tail = self.reg_lambda / n_samples * t
#            t_tail[:,0] = 0
#            g += t_tail
#
#        return self.network.unroll(theta_grads)



    def gradient(self, theta):
        if not self._is_last_theta(theta):
            self.network.theta = theta
            self._forward_propagation()
        avals = self._avals
        zvals = self._zvals
        dvals = self._dvals
        
        thetas = self.network.theta_list
        n_samples = self.n_samples

        dvals[-1][:] = avals[-1]
        dvals[-1] -= self.Y

        # propagate error terms in reverse
        t_z_dprev_dnext = zip(thetas[1:], zvals, dvals, dvals[1:])
        for t, z, dprev, dnext in reversed(t_z_dprev_dnext):
            numpy.dot(dnext, t[:,1:], out=dprev)
            dprev *= self._sigmoid_grad(z)

        theta_grads = [numpy.dot(d.T, a) / n_samples
                       for d, a in zip(dvals, avals)]

        for g, t in zip(theta_grads, thetas):
            t_tail = self.reg_lambda / n_samples * t
            t_tail[:,0] = 0
            g += t_tail

        return self.network.unroll(theta_grads)

    def train(self, lamb, iters):
        self.reg_lambda = float(lamb)
        callback = self.progress_msg.update
        callback

        theta = scipy.optimize.fmin_cg(self.cost, 
                                       self.network.theta, 
                                       self.gradient, 
                                       maxiter=iters,
                                       disp=False,
                                       callback=callback)
        self.network.theta = theta

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

    def get_result(self, sample=False):
        self._forward_propagation()
        h = self._avals[-1]
        if sample:
            h = h[0] ** 1.3   # to weed out the less common letters
            return numpy.random.multinomial(1, h / h.sum())
        else:
            return h.argmax(axis=1)[:,None] == numpy.arange(h.shape[1])[None,:]

class Network(object):
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
        
        self.shape = (shape[0] + 1,) + shape[1:]
        self._theta_shapes = shapes
        self._theta_ranges = ranges
        self._theta_size = size
        
        self._theta = numpy.random.normal(0, 0.05, self._theta_size)

    @property
    def theta_list(self):
        return self.rollup(self._theta)

    @theta_list.setter
    def theta_list(self, theta):
        self._theta = self.unroll(theta)

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        self._theta = theta

    def azshapes(self, n_samples):
        '''Given a number of samples for training, determine the shape
        of the activation and z matrices. This is useful for preallocating
        and reusing memory for these matrices, which can be quite large.'''
        shape = self.shape

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

class ProgressMessage(object):
    def __init__(self, count=0):
        self.message = ('Training... Iteration {}: ' 
                        'Cost: {:6.4f} | Time Elapsed: {:.2f}s '
                        '(total), {:.2f}s (since last iteration) ')
        self.count = count
        self.cost = 0
        self.last_time = time.clock()
        self.total_time = 0

    def update(self, *args, **kwargs):
        self.count += 1
        if self.count > 1:
            now = time.clock()
            elapsed = now - self.last_time
            self.total_time += elapsed
            self.last_time = now
            message = '\r' + self.message.format(self.count, self.cost, self.total_time, elapsed)
            sys.stdout.write(message)
            sys.stdout.flush()
        else:
            self.last_time = time.clock()
            sys.stdout.write('Training... Iteration 1...')
            sys.stdout.flush()

def markov_visualizer(network, start, key, sample=False, n_cycles=100):
    n_features = network.theta_list[0].shape[1] - 1
    n_labels = network.theta_list[-1].shape[0]

    inv_key = {c:n for n, c in enumerate(key)}
    fakeY = numpy.asarray([[1]]) == numpy.arange(n_labels)

    n = NetworkTrainer(network, start, fakeY)
    for i in range(n_cycles):
        r = n.get_result(sample).astype(float)
        s = start[0].reshape(-1, n_labels)

        print ''.join(key[char.nonzero()[0][0]] for char in s)
        start[:,0:n_features - n_labels] = start[:,n_labels:]
        start[:,n_features - n_labels:] = r
        n.update_XY(start, fakeY)

class TrainingData(object):
    def __init__(self):  # eventually make some educated guesses about which classmethod constructor to call
                         # by checking if the input is a directory, h5 file, or pair of npy files
        self.generator = lambda: iter(())
        self.in_size = None
        self.out_size = None

    def __iter__(self):
        return self.generator()

    @classmethod
    def npy_dir(cls, training_dir):
        instance = cls()
        instance.load_npy_dir(training_dir)
        return instance

    @classmethod
    def npy_single(cls, input_output):
        instance = cls()
        instance.load_npy_single(input_output)
        return instance

    @classmethod
    def h5_table(cls, h5_file):
        instance = cls()
        instance.load_h5_table(h5_file)
        return instance

    def load_npy_dir(self, training_dir):
        # Have this shuffle the data across all files. It will 
        # require some refactoring.
        
        in_dir = os.path.join(training_dir, 'input')
        out_dir = os.path.join(training_dir, 'output')
        in_files  = [os.path.join(in_dir, f)  for f in os.listdir(in_dir)  
                                              if f.endswith('.npy')]
        out_files = [os.path.join(out_dir, f) for f in os.listdir(out_dir) 
                                              if f.endswith('.npy')]
        in_files.sort()
        out_files.sort()
        shuff = range(len(in_files))
        random.shuffle(shuff)

        in_files = [in_files[x] for x in shuff]
        out_files = [out_files[x] for x in shuff]

        self.in_size = numpy.load(in_files[0], mmap_mode='r').shape[1]
        self.out_size = numpy.load(out_files[0], mmap_mode='r').shape[1]
        self.generator = self._npy_gen(in_files, out_files)

    def load_npy_single(self, input_output):
        in_file, out_file = input_output

        self.in_size = numpy.load(in_file, mmap_mode='r').shape[1]
        self.out_size = numpy.load(out_file, mmap_mode='r').shape[1]
        self.generator = self._npy_gen([in_file], [out_file])

    def _npy_gen(self, in_seq, out_seq):
        data_zip = itertools.izip(in_seq, out_seq)
        def data_iter():
            for in_f, out_f in data_zip:
                in_arr = numpy.load(in_f).astype(float)
                out_arr = numpy.load(out_f).astype(float)
                yield in_arr, in_f, out_arr, out_f
        return data_iter

    def _h5_fast_bool_ix(self, h5_array, ix, read_chunksize=100000):
        '''Iterate over an h5 array chunkwise to select a random subset
        of the array. `h5_array` should be the array itself; `ix` should
        be a boolean index array with as many values as `h5_array` has
        rows, and you can optionally set the number of rows to read per
        chunk with `read_chunksize` (default is 100000). For some reason
        this is much faster than using `ix` to index the array directly.'''

        n_chunks = h5_array.shape[0] / read_chunksize
        slices = [slice(i * read_chunksize, (i + 1) * read_chunksize)
                  for i in range(n_chunks)]

        a = numpy.empty((ix.sum(), h5_array.shape[1]), dtype=float)
        a_start = 0
        for sl in slices:
            chunk = h5_array[sl][ix[sl]]
            a_end = a_start + chunk.shape[0]
            a[a_start:a_end] = chunk
            a_start = a_end

        return a

    # just to supply some intutition when I come back to this and feel 
    # completely baffled by what I have written. 
    # a = x[ix]
    # inv_ix[ix] = numpy.arange(10)
    # b[inv_ix] = x
    # (a == b).all() is True

    def _h5_fast_ix(self, h5_array, ix, read_chunksize=100000):

        n_chunks = h5_array.shape[0] / read_chunksize
        slices = [slice(i * read_chunksize, (i + 1) * read_chunksize)
                  for i in range(n_chunks)]

        bool_ix = numpy.zeros(h5_array.shape[0], dtype=bool)  
        bool_ix[ix] = 1                                      
        order_ix = numpy.zeros(h5_array.shape[0], dtype=int)
        order_ix[ix] = numpy.arange(ix.shape[0])

        a = numpy.empty((ix.shape[0], h5_array.shape[1]), dtype=float)
        for sl in slices:
            h5_chunk = h5_array[sl]
            bool_chunk = bool_ix[sl]
            order_chunk = order_ix[sl]
            a[order_chunk[bool_chunk]] = h5_chunk[bool_chunk]

        return a

    def load_h5_table(self, h5_file, chunksize=40000):
        f = tables.open_file(h5_file, 'r')

        in_rows = f.root.input.shape[0]
        n_chunks = in_rows / chunksize
        if n_chunks * chunksize < in_rows:
            n_chunks += 1

        mask = numpy.arange(in_rows) % n_chunks
        numpy.random.shuffle(mask)

        self.in_size = f.root.input.shape[1]
        self.out_size = f.root.output.shape[1]

        f.close()

        def data_iter():
            f = tables.open_file(h5_file, 'r')
            h5_in = f.root.input
            h5_out = f.root.output

            for i in range(n_chunks):
                bool_mask = mask == i

                order = numpy.arange(bool_mask.sum())
                numpy.random.shuffle(order)
                in_arr  = self._h5_fast_bool_ix(h5_in,  bool_mask)[order]
                out_arr = self._h5_fast_bool_ix(h5_out, bool_mask)[order]
                
                in_name = '{}-input-chunk-{}'
                in_name = in_name.format(h5_file, i)
                
                out_name = '{}-output-chunk-{}'
                out_name = out_name.format(h5_file, i)
                
                yield in_arr, in_name, out_arr, out_name
            f.close() 

        self.generator = data_iter

    def shape_network(self, layers, gap_size=0):
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

def train_cv(X, Y):
    n_points = X.shape[0]
 
    cv = n_points * 0.75
    X_data = (X[:cv,:], 
              X[cv:,:],)
    Y_data = (Y[:cv,:],
              Y[cv:,:],)
    
    return X_data, Y_data

textwrapper = TextWrapper(width=55).fill
def helptext(s):
    txt = ['\n', textwrapper(s), '\n\n']
    return ''.join(txt)

if __name__ == '__main__':

    nn_parser = argparse.ArgumentParser(description='Feedforward Neural Network.', formatter_class=argparse.RawTextHelpFormatter, add_help=False)
    nn_parser.add_argument('-h', '--help', action='help', help=helptext('Show this help message and exit.'))
    
    input_group = nn_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-I', '--training-input', type=str, nargs=2, metavar='file', help=helptext('Paths to two numpy-readable files containing input and output for training. Mutually exclusive with the --training-directory option.'))
    input_group.add_argument('-D', '--training-directory', type=str, metavar='directory', help=helptext('Path to a directory containing training data. Input and output data should be stored in separate directories `input` and `output`. The two folders should contain only training data, and should have an equal number of files. Inputs will be matched to outputs by sorting the filenames. Mutually exclusive with the --training-directory option.'))
    input_group.add_argument('-5', '--training-h5', type=str, metavar='h5_file', help=helptext('Path to an h5 file containing training data. Input and output data should be stored in arrays `root.input` and `root.output`.')) 

    shape_group = nn_parser.add_mutually_exclusive_group(required=True)
    shape_group.add_argument('-L', '--num-layers', metavar='integer', type=int, help=helptext('Number of layers. If this option is chosen, the sizes of the layers will be automatically determined using input and output data and a mid-layer size heuristic. Mutually exclusive with the --shape option.'))
    shape_group.add_argument('-s', '--shape', metavar='layer_size', nargs='+', type=int, help=helptext('Shape of network specified as a list of layer sizes, starting with the input layer. Mutually exclusive with the --num-layers option.'))


    nn_parser.add_argument('-c', '--cv-split', action='store_true', default=True, help=helptext('Set aside a quarter of the training data fur cross-validation (CV) purposes. Test data is assumed to be held separately. Currently, when running multiple training cycles, the entire input dataset will be randomly shuffled, mixing training and CV data from the previous cycle. In that case, it is best to set aside a separate CV dataset. (The CV output can still be useful for monitoring fit within a single cycle.)'))
    # here add separate CV and Test data inputs, mutually exclusive with '--cv-split'
    
    # There needs to be a way to store shape size alongside theta; 
    # it is clunky to require all these values to match. Once that's
    # implemented, this will be mutually exclusive with --num-layers. 
    # In the long run there should be separate training, testing, and
    # prediction commands.
    nn_parser.add_argument('-T', '--theta', metavar='file', help=helptext('Path to a numpy-readable file containing the weights of a trained network, represented as a flattened array. The shape should match that passed to --shape if used as well as the shape of -X and -Y.'))
    nn_parser.add_argument('-S', '--save-theta', metavar='file', help=helptext('Path to save the current theta values on exit. CAUTION: This currently does nothing to prevent you from overwriting a file, nor does it check that the save location exists.'))
   
    nn_parser.add_argument('-r', '--regularization', metavar='float', default=1, type=float, help=helptext('Regularization factor. Defaults to 1.'))
    nn_parser.add_argument('-i', '--num-iterations', metavar='integer', default=-1, type=int, help=helptext('Number of training iterations. Defaults to 0, in which case a prediction task is assumed.'))
    nn_parser.add_argument('-v', '--visualizer', metavar='mode', type=str, choices=['markov', 'markov-rand'], help=helptext('Chose visualization mode. Only `markov` and `markov-rand` are impemented (for character prediction).'))

    args = nn_parser.parse_args()

    if args.training_input is not None:
        training_data = TrainingData.npy_single(args.training_input)
    elif args.training_directory is not None:
        training_data = TrainingData.npy_dir(args.training_directory)
    else:
        training_data = TrainingData.h5_table(args.training_h5)

    if args.num_layers is not None:
        shape = training_data.shape_network(args.num_layers, gap_size=0)
    else:
        shape = args.shape
    print "Network Shape:", tuple(shape)
    
    nn = Network(shape)
    if args.theta is not None:
        nn.theta = numpy.load(args.theta)

    for i, (X, xname, Y, yname) in enumerate(training_data):
        print
        print "Training Input {}: \n\tX -- {}\n\tY -- {}".format(i, xname, yname)
        
        if args.cv_split:
            (X_tr, X_cv), (Y_tr, Y_cv) = train_cv(X, Y)
        else:
            X_tr, Y_tr = X, Y
        
        tr = NetworkTrainer(nn, X_tr, Y_tr)
        
        tr.train(args.regularization, args.num_iterations)
        a1, a3, a5 = tr.accuracy_topn(1, 3, 5)
        print
        print "Training accuracy:        ", a1
        print "Training accuracy (top 3):", a3
        print "Training accuracy (top 5):", a5

        if args.cv_split:
            tr.update_XY(X_cv, Y_cv)
            a1, a3, a5 = tr.accuracy_topn(1, 3, 5)
            print "CV accuracy:              ", a1
            print "CV accuracy (top 3):      ", a3
            print "CV accuracy (top 5):      ", a5

            res = tr.get_result()
            print
            print "Some predicted results:", res[0:20,:].argmax(axis=1)
            print "The actual results:    ", Y_cv[0:20,:].argmax(axis=1)
            print

        if args.visualizer == 'markov':
            markov_visualizer(nn, X_tr[0:1,:], list(' ' + string.lowercase))
        elif args.visualizer == 'markov-rand':
            markov_visualizer(nn, X_tr[0:1,:], list(' ' + string.lowercase), True)
    
    if args.save_theta is not None:
        numpy.save(args.save_theta, nn.theta)
