from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn import metrics
import pickle
import numpy as np
import time

ins = ['BTb', 'TpC', 'Hn', 'Tbn', 'Va', 'Vn',
       'Vc', 'Cb', 'Ob', 'Fl', 'Bn', 'ClBb']
total = {'BTb': 19, 'TpC': 18, 'Hn': 25, 'Tbn': 30, 'Va': 24, 'Vn': 22,
         'Vc': 30, 'Cb': 27, 'Ob': 24, 'Fl': 25, 'Bn': 30, 'ClBb': 33}
train = pickle.load(open("./data/trainset_svm.pkl", 'rb'))
test = pickle.load(open("./data/testset_svm.pkl", 'rb'))

trainset = []
trainlabel = []
testset = []
testlabel = []

for x in train:
    trainset.append(x[0].numpy())
    trainlabel.append(x[1])

for y in test:
    testset.append(y[0].numpy())
    testlabel.append(y[1])

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

print("starting grid searching")

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.001, 0.0002, 0.0003, 0.004],
                     'C': [10000]}]


print("# Tuning hyper-parameters")

clf = GridSearchCV(SVC(), tuned_parameters, cv=5)
clf.fit(trainset, trainlabel)

print("Best parameters set found on development set:")
print(clf.best_params_)
print("Grid scores on development set:")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    print()

# print("Detailed classification report:")

# y_true, y_pred = testlabel, clf.predict(testset)
# print(classification_report(y_true, y_pred))


# clf = OneVsRestClassifier(SVC(kernel='rbf',gamma=0.01, C=1000), n_jobs=-1)
# print("Starting training")
# start = time.clock()
# clf.fit(trainset, trainlabel)
# end = time.clock()
# print("times:{} ms".format(1000*(end-start)))

# pickle.dump(clf, open("svm_linear_stand.pkl", 'wb'))
# clf = pickle.load(open("svm_linear.pkl", 'rb'))
# predict = clf.predict(testset)
# print(predict)

# acc = cross_val_score(estimator=clf, X=trainset, y=trainlabel, cv=10)
# print(acc.mean())
# print(acc.std())

# pre = clf.predict(testset)
# print(clf.predict_proba())

# result = {}
# for i in ins:
#     result[i] = 0

# pre = clf.predict(testset)
# for one, two in zip(pre, testlabel):
#     if one == two:
#         result[ins[one]] += 1

# for i in result.keys():
#     print("{}: {}/{} = {} %".format(i,
#                                     result[i], total[i], float(result[i])/float(total[i])))

# s1 = metrics.accuracy_score(testlabel, pre)
# print(s1)
