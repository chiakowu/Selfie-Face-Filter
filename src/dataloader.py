import csv
import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms


class csvDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, data_type, transform=None):
        with open(data_file) as f:
            csv_reader = csv.reader(f, delimiter=',')
            self.data = []

            for row in csv_reader:
                if row[-1] == data_type:
                    label = int(row[0])
                    if label == 0:
                        pixel = np.asarray(list(map(int, row[1].split()))).reshape(48, 48, -1)
                        self.data.append([2, pixel])
                    elif label == 3:
                        pixel = np.asarray(list(map(int, row[1].split()))).reshape(48, 48, -1)
                        self.data.append([1, pixel])
                    elif label == 6:
                        pixel = np.asarray(list(map(int, row[1].split()))).reshape(48, 48, -1)
                        self.data.append([0, pixel])

            self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = [self.data[idx][0]]
        pixels = self.data[idx][1]
        if self.transform is not None:
            pixels = self.transform(np.uint8(pixels))
        sample = (pixels, torch.LongTensor(label))
        return sample


class FacialExpressionDataLoader(object):
    def __init__(self, data_file):

        train_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(45),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.train_loader = csvDataset(data_file, "Training", transform=train_transform)
        self.val_loader = csvDataset(data_file, "PublicTest", transform=test_transform)
        self.test_loader = csvDataset(data_file, "PrivateTest", transform=test_transform)

        print('training data: ', len(self.train_loader))
        print('validation data: ', len(self.val_loader))
        print('testing data: ', len(self.test_loader))

        self.classes = ('Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral')

        print('All data loaded')


