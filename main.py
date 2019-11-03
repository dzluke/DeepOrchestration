import torch
import torch.nn as nn
import torchvision.transforms as transforms
import argparse
from model import OrchMatchNet
from process import get_class_num, make_dataset
from dataset import OrchDataSet
import time


epoch_num = 50
batch_size = 50
learning_rate = 0.001


def arg_parse():
    parser = argparse.ArgumentParser(description='combination of orch')
    parser.add_argument('--model', default='cnn')

    return parser.parse_args()


def main():
    print("Starting parsing -----")
    arg = arg_parse()
    print("End parsing")

    # dataset
    print("Start loading data -----")

    trainset = OrchDataSet('./data/trainset1.pkl', transforms.ToTensor())
    testset = OrchDataSet('./data/testset1.pkl', transforms.ToTensor())

    # load data
    train_load = torch.utils.data.DataLoader(dataset=trainset,
                                             batch_size=batch_size,
                                             shuffle=True)

    test_load = torch.utils.data.DataLoader(dataset=testset,
                                            batch_size=batch_size,
                                            shuffle=True)
    print("End loading data")

    # model construction
    out_num = get_class_num()
    model = OrchMatchNet(out_num)

    if arg.model == 'cnn':
        train(model, train_load, test_load)


def train(model, train_load, test_load):
    print("Starting training")
    device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(train_load)
    for epoch in range(epoch_num):
        for i, (trains, labels) in enumerate(train_load):
            trains = trains.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(trains)
            # The first 107 belongs to the first intrusment
            labels = labels.long()
            loss_1 = criterion(outputs[:, :107], labels[:, 0])
            loss_2 = criterion(outputs[:, 107:], labels[:, 1])
            loss = loss_1 + loss_2
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            if (i+1)/10 == 0:
                print('Epoch:[{}/{}], Step:[{}/{}], Loss:{:.4f}'.format(epoch +
                                                                        1, epoch_num, i+1, total_step, loss))

        model.eval()

        with torch.no_grad:
            total_num = 0
            total_time = 0.0
            correct_num = 0
            correct_num_sound1 = 0
            correct_num_sound2 = 0

            for tests, labels in test_load:
                tests = tests.to(device)
                labels = labels.to(device)

                labels = labels.long()

                start = time.time()
                outputs = model(tests)
                predicts = get_pred(outputs)
                end = time.time()

                total_num += labels.size(0)
                total_time += float(end-start)

                correct_num, correct_num_sound1, correct_num_sound2 = count_correct(
                    predicts, labels)

                total_acc = 100*float(correct_num)/total_num
                first_acc = 100*float(correct_num_sound1)/total_num
                second_acc = 100*float(correct_num_sound2)/total_num

                print("Total Accuracy: {:.3f}%, first Acc: {:.3f}%, second Acc: {:.3f}%".format(
                    total_acc, first_acc, second_acc))

    torch.save(model.state_dict, 'model.ckpt')


def count_correct(predicts, labels):
    assert predicts.size[0] == labels.size[0]
    correct = 0
    correct_1 = 0
    correct_2 = 0
    for one, two in zip(predicts, labels):
        if torch.equal(one, two):
            correct += 1
        if one[0] == two[0]:
            correct_1 += 1
        if one[1] == two[1]:
            correct_2 += 1

    return correct, correct_1, correct_2


def get_pred(output):
    pred_1 = output[:, :107]
    pred_2 = output[:, 107:]

    idx_1 = pred_1.max(1, keepdim=True)[1]
    idx_2 = pred_2.max(1, keepdim=True)[1]

    pred = torch.cat((idx_1, idx_2), dim=1)

    return pred


if __name__ == "__main__":
    main()
