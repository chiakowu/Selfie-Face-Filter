import argparse
import src.models as models


def model_class(class_name):
    return getattr(models, class_name)


def argParser():
    parser = argparse.ArgumentParser(description='PyTorch Homework')
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--batchSize', default=4, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    return parser.parse_args()
