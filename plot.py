import matplotlib.pyplot as plt
import pickle

L = ['Ob-Hn', 'Ob-Hn-Vn', 'Ob-Hn-Vn-Fl']
SVM_acc = [0.399, 0.119, 0.0302]
epochs = list(range(4,80,5))

res = []

for i in range(6):
    res.append([])
    for j in epochs:
        f = open('./model/run{}/result{}.pkl'.format(i, j), 'rb')
        r = pickle.load(f)
        f.close()
        res[i].append(r)

accs = {'Overall accuracy' : []}
i = 5
for k in res[i][0]['pitch_acc'].keys():
    accs[k] = []

for j in res[i]:
    accs['Overall accuracy'].append(j['acc'])
    for k in j['pitch_acc'].keys():
        accs[k].append(j['pitch_acc'][k])

for k in accs:
    plt.plot(epochs, accs[k], '-o', label=k)
plt.grid()
plt.title('Accuracies for N=4 with Vn, Va, Vc and ClBb only')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()