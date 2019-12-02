from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn import metrics
from process import show_all_class_num
import pickle
import numpy as np
import time
import copy

ins = ['BTb', 'TpC', 'Hn', 'Tbn', 'Va', 'Vn',
       'Vc', 'Cb', 'Ob', 'Fl', 'Bn', 'ClBb']
# total = {'BTb': 19, 'TpC': 18, 'Hn': 25, 'Tbn': 30, 'Va': 24, 'Vn': 22,
#          'Vc': 30, 'Cb': 27, 'Ob': 24, 'Fl': 25, 'Bn': 30, 'ClBb': 33}
total = {'BTb': 136, 'TpC': 128, 'Hn': 147, 'Tbn': 131, 'Va': 135, 'Vn': 114,
         'Vc': 143, 'Cb': 133, 'Ob': 148, 'Fl': 119, 'Bn': 121, 'ClBb': 145}

train = pickle.load(open("./data/trainset_mini.pkl", 'rb'))
test = pickle.load(open("./data/testset_mini.pkl", 'rb'))

trainset = []
trainlabel = []
testset = []
testlabel = []

for x in train:
    trainset.append(x[0].numpy())
    trainlabel.append(x[1].numpy())

for y in test:
    testset.append(y[0].numpy())
    testlabel.append(y[1].numpy())

scale = preprocessing.MinMaxScaler()

trainset = np.array(trainset)
num, a, b, c = trainset.shape
trainset = trainset.reshape(num, a*b*c)
trainset = scale.fit_transform(trainset)

testset = np.array(testset)
num, a, b, c = testset.shape
testset = testset.reshape(num, a*b*c)
testset = scale.transform(testset)

trainlabel = np.array(trainlabel)
testlabel = np.array(testlabel)

# model
clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=60))
clf.fit(trainset, trainlabel)

# total = {}
# for i in ins:
#     total[i] = 0
# for t in testlabel:
#     total[ins[t]] += 1
# print(total)

result = {}
for i in ins:
    result[i] = 0

# pre = clf.predict(testset)
# for i in pre:
#     print(i)

pre_p = clf.predict_proba(testset)
pre_pp = np.zeros((800, 12))
for index, i in enumerate(pre_p):
    m1 = max(i)
    mid = list(i).index(m1)
    pre_pp[index][mid] = 1.0
    i[mid] = 0
    m2 = max(i)
    mid = list(i).index(m2)
    pre_pp[index][mid] = 1.0

s1 = metrics.accuracy_score(testlabel, pre_pp)
print(s1)


def decode(labels):
    decode_label = []
    labels_copy = copy.deepcopy(labels)
    for i in range(len(labels_copy)):
        one = list(labels_copy[i]).index(1)
        labels_copy[i][one] = 0
        two = list(labels_copy[i]).index(1)
        decode_label.append([one, two])

    return decode_label


def count_correct(predicts, labels, stat_result, error_stat=None):
    correct = 0
    correct_single = 0

    # !!! keep in mind to consider the sequence of the output
    for one, two in zip(predicts, labels):
        if one[0] == two[0] and one[1] == two[1] or one[0] == two[1] and one[1] == two[0]:
            correct += 1
            stat_result[ins[one[0]]] += 1
            stat_result[ins[one[1]]] += 1

        elif one[0] == two[0] or one[0] == two[1]:
            correct_single += 1
            stat_result[ins[one[0]]] += 1

        elif one[1] == two[1] or one[1] == two[0]:
            correct_single += 1
            stat_result[ins[one[1]]] += 1

    return [correct, correct_single], stat_result


pre_pp = decode(pre_pp)
testlabel = decode(testlabel)
_, result = count_correct(pre_pp, testlabel, result)

for i in result.keys():
    print("{}: {}/{} = {} %".format(i,
                                    result[i], total[i], 100*float(result[i])/float(total[i])))


# best = 0
# for _ in range(100):
#     clf.fit(trainset, trainlabel)
#     pre = clf.predict(testset)

#     s1 = metrics.accuracy_score(testlabel, pre)
#     if best < s1:
#         best = s1
#         print(s1)

# print("best: ", best)
