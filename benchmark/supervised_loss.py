"Supervised loss benchmark"
import json
import argparse
import tensorflow_datasets as tfds

def run(args):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training config')
    parser.add_argument('--config', '-c', help='config path')
    args = parser.parse_args()

    if not args.config or not args.dataset_path:
        parser.print_usage()
        quit()
    run(args)
