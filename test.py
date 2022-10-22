import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str)
parser.add_argument('--exp', type=str)
parser.add_argument('--dataset', type=str)

args = parser.parse_known_args()
print(args)
