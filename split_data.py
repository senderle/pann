#!/usr/bin/python

import random

def main(args):
    train = open(args.base_name + '.train', 'w')
    validate = open(args.base_name + '.validate', 'w')
    test = open(args.base_name + '.test', 'w')

    if args.validation_ratio + args.test_ratio >= 0.9:
        test_th = 0.8
        val_th = 0.6
    else:
        test_th = 1.0 - args.test_ratio
        val_th = test_th - args.validation_ratio

    for t in args.text_files:
        with open(t) as f:
            chunk = f.read(50000)
            while chunk:
                select = random.random()
                if select > test_th:
                    test.write(chunk)
                elif select > val_th:
                    validate.write(chunk)
                else:
                    train.write(chunk)
                chunk = f.read(50000)

    train.close()
    validate.close()
    test.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Split data into training, validation, '
                                     'and test sets.')
    parser.add_argument('text_files', metavar='file', type=str, nargs='+',
                        help='Text file(s) for processing.')
    parser.add_argument('-v', '--validation-ratio', metavar='(0-1.0)',
                        default=0.2, help='The proportion of the input to use '
                        'for validation.')
    parser.add_argument('-t', '--test-ratio', metavar='(0-1.0)', default=0.2,
                        help='The proportion of the input to use for testing.')
    parser.add_argument('-n', '--base-name', metavar='filename',
                        default='data', help='The base filename. The output '
                        'sets will be named `{base-name}.train`, '
                        '`{base-name}.validate`, and `{base-name}.test`.')
    main(parser.parse_args())
