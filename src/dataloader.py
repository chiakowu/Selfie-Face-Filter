import csv
from torch.utils.data import Dataset


class FacialExpressionDataLoader(Dataset):
    def __init__(self, data_file, transform=None):
        with open(data_file) as f:
            csv_reader = csv.reader(f, delimiter=',')
            index = 0

            self.train = []
            self.val = []
            self.test = []

            for row in csv_reader:
                if row[-1] == "Training":
                    self.train.append([row[0], row[1]])
                elif row[-1] == "PublicTest":
                    self.val.append([row[0], row[1]])
                elif row[-1] == "PrivateTest":
                    self.test.append([row[0], row[1]])

    def __len__(self):
        return len(self.train)

    def __getitem__(self, idx):
        return {'label': self.train[idx][0], 'image': self.train[idx][1]}
