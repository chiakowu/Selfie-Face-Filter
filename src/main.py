
from src import dataloader, models


def main():
    data = dataloader.FacialExpressionDataLoader(data_file='../../fer2013/fer2013.csv')
    train_set = data.train
    val_set = data.val
    test_set = data.test



if __name__ == '__main__':
    main()