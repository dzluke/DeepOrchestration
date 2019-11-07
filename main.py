import torch
import torch.nn as nn
import torchvision.transforms as transforms
import argparse
from model import OrchMatchNet
from process import get_class_num, make_dataset
from dataset import OrchDataSet
import time
import os


epoch_num = 50
batch_size = 16
learning_rate = 0.001
model_path = './model'


def arg_parse():
    parser = argparse.ArgumentParser(description='combination of orch')
    parser.add_argument('--model', default='cnn')
    parser.add_argument('--is_resume', default='False',
                        choices=['True', 'False'])

    return parser.parse_args()


def main():
    print("Starting parsing -----")
    arg = arg_parse()
    print("End parsing")

    # dataset
    print("Start loading data -----")

    trainset = OrchDataSet('./data/trainset.pkl', transforms.ToTensor())
    testset = OrchDataSet('./data/testset.pkl', transforms.ToTensor())

    # load data
    train_load = torch.utils.data.DataLoader(dataset=trainset,
                                             batch_size=batch_size,
                                             shuffle=True)

    test_load = torch.utils.data.DataLoader(dataset=testset,
                                            batch_size=1,
                                            shuffle=False)
    print("End loading data")

    # model construction
    out_num = get_class_num()
    model = OrchMatchNet(out_num)
    start_epoch = 0

    # model load
    if arg.is_resume == 'True':
        ckpt = os.listdir(model_path)
        ckpts = []
        for x in ckpt:
            if x.endswith('.pth'):
                ckpts.append(int(x.split('.')[0].split('_')[-1]))

        if len(ckpt) != 0:
            model_resume_path = 'model_epoch_'+str(max(ckpts))+'.pth'
            state = torch.load(model_path+'/'+model_resume_path)
            model.load_state_dict(state['state_dict'])
            start_epoch = state['epoch']
            print("Load %s successfully! " % model_resume_path)
        else:
            print("[Error] no checkpoint ")

    # train model
    if arg.model == 'cnn':
        train(model, train_load, test_load, start_epoch)


def train(model, train_load, test_load, start_epoch):
    print("Starting training")

    device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_acc = 0
    best_epoch = None
    acc_sets = []

    for epoch in range(start_epoch, epoch_num):
        model.train()
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

            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print('Epoch:[{}/{}], Step:[{}/{}], Loss:{:.4f}, Loss_1:{:.4f}, Loss_2:{:.4f} '.format(epoch +
                                                                                                       1, epoch_num, i+1, len(train_load), loss, loss_1, loss_2))

        # save cuurent model
        model_name = model_path+'/model_epoch_'+str(epoch+1)+'.pth'
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
        }

        torch.save(state, model_name)
        print('model_epoch_'+str(epoch+1)+' saved')

        model.eval()

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

            correct_list = count_correct(predicts, labels)

            correct_num += correct_list[0]
            correct_num_sound1 += correct_list[1]
            correct_num_sound2 += correct_list[2]

        total_acc = 100*float(correct_num)/total_num
        first_acc = 100*float(correct_num_sound1)/total_num
        second_acc = 100*float(correct_num_sound2)/total_num

        avg_time = total_time/float(len(test_load))
        print("Average Time: {:2.3f} ms".format(1000*avg_time))
        print("Total Accuracy: {:.3f}%, first Acc: {:.3f}%, second Acc: {:.3f}%".format(
            total_acc, first_acc, second_acc))

        acc_sets.append([epoch, total_acc, first_acc, second_acc])

        if total_acc > best_acc:
            best_acc = total_acc
            best_epoch = epoch+1
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
            }
            torch.save(state, "epoch_best.pth")

        f = open('acc.csv', 'w')
        for acc in acc_sets:
            for a in acc:
                if a != acc[len(acc)-1]:
                    f.write(str(a)+',')
                else:
                    f.write(str(a))
            f.write('\n')
        f.close()

    print("Best test accuracy: {} at epoch: {}".format(best_acc, best_epoch))


def count_correct(predicts, labels):
    assert predicts.size(0) == labels.size(0)
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


def test():
    device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')

    testset = OrchDataSet('./data/testset.pkl', transforms.ToTensor())
    test_load = torch.utils.data.DataLoader(dataset=testset,
                                            batch_size=1,
                                            shuffle=False)

    out_num = get_class_num()
    model = OrchMatchNet(out_num)

    acc_sets = []
    model_paths = []

    print("Start testing")
    for x in os.listdir('./model'):
        if x.endswith('.pth'):
            x = x.split('.')[0].split('_')[-1]
            model_paths.append(int(x))
            model_paths.sort()

    for model_path in model_paths:
        state = torch.load('./model/model_epoch_'+str(model_path)+'.pth')
        model.load_state_dict(state['state_dict'])

        model.eval()

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

            correct_list = count_correct(predicts, labels)

            correct_num += correct_list[0]
            correct_num_sound1 += correct_list[1]
            correct_num_sound2 += correct_list[2]

        total_acc = 100*float(correct_num)/total_num
        first_acc = 100*float(correct_num_sound1)/total_num
        second_acc = 100*float(correct_num_sound2)/total_num

        avg_time = total_time/float(len(test_load))
        print("Average Time: {:2.3f} ms".format(1000*avg_time))
        print("Total Accuracy: {:.3f}%, first Acc: {:.3f}%, second Acc: {:.3f}%, at epoch: {} ".format(
            total_acc, first_acc, second_acc, model_path))

        acc_sets.append([model_path, total_acc, first_acc, second_acc])

        f = open('acc.csv', 'w')
        for acc in acc_sets:
            for a in acc:
                if a != acc[len(acc)-1]:
                    f.write(str(a)+',')
                else:
                    f.write(str(a))
            f.write('\n')
        f.close()


if __name__ == "__main__":
    main()
