#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""Converts an (LR - PPL) log to graph."""

import argparse
import matplotlib.pyplot as plt
import re


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Converts an (LR - PPL) log to graph.')
    parser.add_argument('input', type=str, 
                        help='the input file.')
    parser.add_argument('output', type=str, nargs='?',
                        help='the output file. By default, the same as input '
                             'file, with the extension replaced by .png.')
    args = parser.parse_args()
    if not args.output:
        args.output = args.input.rsplit('.', 1)[0]
    if not args.output.endswith('.png'):
        args.output += '.png'
    return args


def get_data(input_file):
    """Reads the LR - PPL pairs from the log stream."""
    with open(input_file) as inf:
        p = re.compile(r'.*\| lr ([.0-9]+) \|.*\| ppl\s+([.0-9]+)\s*')
        data = []
        for line in inf:
            m = p.match(line)
            if m:
                data.append((float(m.group(1)), float(m.group(2))))
        return zip(*data)


def main():
    args = parse_arguments()
    lr, ppl = get_data(args.input)
    ppl = [p if p < 3000 else 3000 for p in ppl]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lr, ppl)
    ax.set_xlabel('learning rate', labelpad=20)
    ax.set_ylabel('perplexity', labelpad=20)
    ax.set_xlim([0, 2])
    ax.set_ylim([900, 3100])
    plt.tight_layout()
    plt.savefig(args.output)


if __name__ == '__main__':
    main()
