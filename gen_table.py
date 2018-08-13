#!/usr/bin/python

import argparse
import string
import numpy


def gen_ft_parser():
    ft_parser = argparse.ArgumentParser(
        description='Generate a Character-Feature Translation Table')
    ft_parser.add_argument('alphabet_file', metavar='alphabet_file',
        type=str, help='A file contianing all the characters that will '
        'appear in the translation table.')
    ft_parser.add_argument('save_file', metavar='save_path',
        type=str, help='The feature table filename.')
    return ft_parser

def construct_alphabet(alpha_string):
    symbols = set(alpha_string)
    alphabet = ''.join(sorted(c for c in string.printable if c in symbols))
    return numpy.array(list(alphabet))

def load_alphabet(alphabet_file):
    with open(alphabet_file) as alphabet:
        alphabet = alphabet.read(100000).replace('\n', ' ')
        return construct_alphabet(alphabet)

def gen_row(c, key):
    row = [False] * (len(key) + 1)
    row[key[c.lower()]] = True
    row[-1] = c.isupper()
    return row

def build_table(alphabet):
    code = ''.join(sorted(set(''.join(alphabet).lower())))
    key = {c: i for i, c in enumerate(code)}
    table = numpy.zeros((len(alphabet), len(key) + 1))
    for i, c in enumerate(alphabet):
        table[i] = gen_row(c, key)
    return table

def main(args):
    table = build_table(load_alphabet(args.alphabet_file))
    numpy.save(args.save_file, table)

if __name__ == "__main__":
    main(gen_ft_parser().parse_args())
