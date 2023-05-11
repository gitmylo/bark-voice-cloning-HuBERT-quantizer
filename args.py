from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--path', required=True)

args = parser.parse_args()
