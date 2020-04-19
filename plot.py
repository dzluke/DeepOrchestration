import matplotlib.pyplot as plt
import pickle

run_nb = 3
res = {'acc' : []}

epochs = list(range(4,50,5))

for j in epochs:
    f = open('./model/run{}/result{}.pkl'.format(run_nb, j), 'rb')
    r = pickle.load(f)
    if len(res.keys()) == 1:
        for k in r['pitch_acc']:
            res[k] = []
    
    res['acc'].append(r['acc'])
    for k in r['pitch_acc']:
        res[k].append(r['pitch_acc'][k])
    f.close()

for k in res:
    st = '-o'
    l = k + ' accuracy'
    if k == 'acc':
        st = 'o--'
        l = 'Overall accuracy'
    plt.plot(epochs, res[k], st, label=l)
        


plt.grid()
plt.title('Accuracies for an orchestra with {} instruments'.format(len(res.keys())-1))
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()