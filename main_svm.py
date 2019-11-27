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

train = pickle.load(open("./data/trainset_svm_2.pkl", 'rb'))
test = pickle.load(open("./data/testset_svm_2.pkl", 'rb'))

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
# scale = preprocessing.StandardScaler()

trainset = np.array(trainset).reshape(1225, 2*128*128)
trainset = scale.fit_transform(trainset)

testset = np.array(testset).reshape(307, 2*128*128)
testset = scale.transform(testset)

trainlabel = np.array(trainlabel)
testlabel = np.array(testlabel)

# print("starting grid searching")

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]


# print("# Tuning hyper-parameters")

# clf = GridSearchCV(SVC(gamma='scale'), tuned_parameters, cv=5, n_jobs=-1)
# clf.fit(trainset, trainlabel)

# print("Best parameters set found on development set:")
# print(clf.best_params_)
# print("Grid scores on development set:")
# means = clf.cv_results_['mean_test_score']
# stds = clf.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
#     print()

# print("Detailed classification report:")

# y_true, y_pred = testlabel, clf.predict(testset)
# print(classification_report(y_true, y_pred))


clf = OneVsRestClassifier(SVC(kernel='linear'), n_jobs=-1)
print("Starting training")
start = time.clock()
clf.fit(trainset, trainlabel)
end = time.clock()
print("times:{} ms".format(1000*(end-start)))
# pickle.dump(clf, open("svm_linear_stand.pkl", 'wb'))
# clf = pickle.load(open("svm_linear.pkl", 'rb'))
# predict = clf.predict(testset)
# print(predict)

# acc = cross_val_score(estimator=clf, X=trainset, y=trainlabel, cv=10)
# print(acc.mean())
# print(acc.std())
score1 = clf.score(trainset, trainlabel)
print("score_train: ", score1)
score2 = clf.score(testset, testlabel)
print("score_test: ", score2)
