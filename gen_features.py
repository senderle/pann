#!/usr/bin/python

import numpy
import sys
import re
import string
import argparse
import os
import tables
from contextlib import closing

def strip_text(txt, alphabet, squeeze_whitespace=True):
    if squeeze_whitespace:
        txt = re.sub(r'\s+', ' ', txt)
    del_chars = set(txt) - set(alphabet)
    return txt.translate(None, ''.join(del_chars))

def gen_windows(arr, window_size):
    '''Given a window_size of 2, this creates [[1, 2], [2, 3], [3, 4]]
    from [1, 2, 3, 4] using no extra memory.'''
    shape = arr.shape
    shape = (shape[0] - window_size + 1, window_size)

    strides = arr.strides
    strides = (strides[0],) + strides

    return numpy.lib.stride_tricks.as_strided(arr, shape, strides)

def expand_arr(txt_arr, n_chars, feature_table):
    n_features = feature_table.shape[1]
    windows = gen_windows(txt_arr, n_chars)

    X_data = feature_table[windows[:,:-1]]
    Y_data = feature_table[windows[:,-1:]]

    return (X_data.reshape(-1, (n_chars - 1) * n_features),
            Y_data.reshape(-1, n_features))

def text_to_int(txt_arr, alphabet):
    out = numpy.empty(len(txt_arr), dtype='b')
    out[:] = (alphabet[:,None] == txt_arr[None,:]).T.nonzero()[1]
    return out

def construct_alphabet(alpha_string):
    symbols = set(alpha_string)
    alphabet = ''.join(sorted(c for c in string.printable if c in symbols))
    return numpy.array(list(alphabet))

def load_alphabet(alphabet_file):
    with open(alphabet_file) as alphabet:
        alphabet = alphabet.read(100000).replace('\n', ' ')
        return construct_alphabet(alphabet)

def file_iter(filenames):
    for fn in filenames:
        with open(fn) as txt:
            data = txt.read(50000)
            while data:
                yield data
                data = txt.read(50000)

def write_h5(alphabet, symbol_table, window_size, text_files, write_path):
    n_symbols = symbol_table.shape[1]
    n_features = n_symbols * (window_size - 1)

    with closing(tables.open_file(write_path, mode='w')) as training_data:

        a = tables.BoolAtom()
        bl_filter = tables.Filters(5, 'blosc')

        training_input = training_data.create_earray(
            training_data.root, 'input', a, (0, n_features),
            'Training Input', bl_filter, 4000000)
        training_output = training_data.create_earray(
            training_data.root, 'output', a, (0, n_symbols),
            'Training Output', bl_filter, 4000000)

        for txt in file_iter(text_files):
            X, Y = generate_segment(txt, alphabet, window_size, symbol_table)
            print X.shape
            print Y.shape
            training_input.append(X)
            training_output.append(Y)

def generate_segment(txt, alphabet, window_size, symbol_table):
    txt = strip_text(txt, alphabet)
    txt = numpy.array(list(txt))
    txt = text_to_int(txt, alphabet)
    return expand_arr(txt, window_size, symbol_table)

def gen_td_parser():
    td_parser = argparse.ArgumentParser(
        description='Generate Training Data for a Character-Level Language Model')
    td_parser.add_argument('text_files', metavar='file', type=str, nargs='+',
        help=('Text file(s) for processing.'))
    td_parser.add_argument('-t', '--symbol-table', metavar='filename',
        type=str, required=True, help=('Numpy data file containing a '
        'symbol table, with each row representing a feature vector for '
        'the corresponding character in the `--alphabet-file`, '
        'in _sorted order_.'))
    td_parser.add_argument('-a', '--alphabet-file', metavar='path',
        type=str, required=True, help=('A file containing characters '
        'for use as an alphabet. This will be reduced to a set with each '
        'character occurring only once, and will be sorted. All values must '
        'be printable.'))
    td_parser.add_argument('-c', '--num-chars', metavar='integer', type=int,
        default=50, help=('Number of characters per training sample. '
        'Uses the last character as the expected output value.'))
    td_parser.add_argument('-s', '--save_h5', metavar='h5_file', type=str,
        required=True, help=('The name of the h5 output file.'))
    return td_parser

def main(args):
    alphabet = load_alphabet(args.alphabet_file)
    symbol_table = numpy.load(args.symbol_table)
    write_h5(alphabet, symbol_table, args.num_chars,
             args.text_files, args.save_h5)

if __name__ == '__main__':
    td_parser = gen_td_parser()
    main(td_parser.parse_args())


