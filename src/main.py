import torch
import time
import datetime
import dataloader, models


def train(model, data_loader, optimizer, criterion, epoch):
    running_loss = 0.0
    total_loss = 0.0

    for i, data in enumerate(data_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        labels = labels.view(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_loss += loss.item()
        if (i + 1) % 200 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0
    avg_total_loss = total_loss / i
    print('Final Summary:   loss: %.3f' % (avg_total_loss))
    return avg_total_loss


def test(model, data_loader, tag=''):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            outputs = model(images)
            labels = labels.view(-1)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print('%s Accuracy of the network: %d %%' % (tag, 100 * correct / total))
    return test_accuracy

    # class_correct = list(0. for i in range(7))
    # class_total = list(0. for i in range(7))
    # with torch.no_grad():
    #     for data in data_loader:
    #         images, labels = data
    #         outputs = model(images)
    #         labels = labels.view(-1)
    #         _, predicted = torch.max(outputs, 1)
    #         c = (predicted == labels).squeeze()
    #         for i in range(len(labels)):
    #             label = labels[i]
    #             class_correct[label] += c[i].item()
    #             class_total[label] += 1

    # for i in range(7):
    #     print('%s Accuracy of %5s : %2d %%' % (
    #         tag, data_loader.classes[i], 100 * class_correct[i] / class_total[i]))


def save_best_model(epochs, model, optimizer, loss, training_accuracy, val_accuracy, test_accuracy, file_path):
    torch.save({
        'epochs': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'training_accuracy': training_accuracy,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy,
    }, 'saved_model/' + file_path)


def main():
    batch_size = 16
    test_batch_size = 4
    epochs = 200
    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 0.0005

    data = dataloader.FacialExpressionDataLoader(data_file='../../fer2013/fer2013.csv')
    train_loader = torch.utils.data.DataLoader(data.train_loader, batch_size, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(data.val_loader, test_batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(data.test_loader, test_batch_size, shuffle=True, num_workers=0)

    conv_net = models.ConvNet()
    criterion = conv_net.criterion()
    optimizer = conv_net.optimizer(learning_rate, momentum, weight_decay)

    print('epochs: ', epochs, ' batch_size: ', batch_size, ' learning_rate: ', learning_rate, ' momentum: ', momentum, ' weight_decay: ', weight_decay)
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S.pt')

    best_accuracy = 0
    for epoch in range(epochs):
        conv_net.adjust_learning_rate(optimizer, epoch, learning_rate)
        loss = train(conv_net, train_loader, optimizer, criterion, epoch)
        train_accuracy = test(conv_net, train_loader, 'Train')
        val_accuracy = test(conv_net, val_loader, 'Val')
        test_accuracy = test(conv_net, test_loader, 'Test')
        if test_accuracy > best_accuracy:
            save_best_model(epoch, conv_net, optimizer, loss, train_accuracy, val_accuracy, test_accuracy, st)


if __name__ == '__main__':
    main()
