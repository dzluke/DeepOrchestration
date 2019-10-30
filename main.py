import torch
import torch.nn as nn
import argparse
from model import OrchMatchNet
from process import get_class_num, make_dataset


epoch_num = 50
batch_size = 50
learning_rate = 0.001


def arg_parse():
    parser = argparse.ArgumentParser(description='combination of orch')
    parser.add_argument('--model', default='cnn')

    return parser.parse_args()


def main():
    arg = arg_parse()

    # dataset
    trainset = make_dataset('train')
    testset = make_dataset('test')

    # load data
    train_load = torch.utils.data.DataLoader(dataset=trainset,
                                             batch_size=batch_size,
                                             shuffle=True)

    test_load = torch.utils.data.DataLoader(dataset=testset,
                                            batch_size=batch_size,
                                            shuffle=True)

    # model construction
    out_num = get_class_num()
    model = OrchMatchNet(out_num)

    if arg.model == 'cnn':
        train(model, train_load, test_load)


def train(model, train_load, test_load):
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters, lr=learning_rate)

    total_step = len(train_load)
    for epoch in range(epoch_num):
        for i, (trains, labels) in enumerate(train_load):
            trains = trains.to(device)
            labels = labels.to(device)

            output = model(trains)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1)/10 == 0:
                print('Epoch:[{}/{}], Step:[{}/{}], Loss:{:.4f}'.format(epoch +
                                                                        1, epoch_num, i+1, total_step, loss))

    model.eval()
    with torch.no_grad:
        total = 0
        correct = 0
        for tests, labels in test_load:
            tests = tests.to(device)
            labels = labels.to(device)
            output = model(tests)
            _, predicts = torch.max(output.datas, 1)

            total += labels.size(0)
            corret += (predicts == label).sum().item()

            print("Accuracy: %.2f" % (correct/total))

    torch.save(model.state_dict, 'model.ckpt')


if __name__ == "__main__":
    main()
