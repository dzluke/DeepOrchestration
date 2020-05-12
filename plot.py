import matplotlib.pyplot as plt
import pickle

def plot_run(path, run_nb):
    res = {'acc' : []}
    
    MM = len([x for x in os.listdir('{}/run{}'.format(path, run_nb)) if 'result' in x])
        
        
    epochs = list(range(4,MM*5,5))
    
    for j in epochs:
        f = open('{}/run{}/result{}.pkl'.format(path, run_nb, j), 'rb')
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
            
    max_epoch = res['acc'].index(max(res['acc']))
    print("Max accuracy achieved at epoch {}".format(epochs[max_epoch]))
    for k in res:
        print("Accuracy of {} : {}".format(k, res[k][max_epoch]))
    
    
    plt.grid()
    plt.title('Accuracies for an orchestra with {} instruments for the ResNet'.format(len(res.keys())-1))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
def get_instr(path, list_run):
    res = {'acc' : []}
    
    epochs = list(range(4,50,5))
    
    for i in list_run:
        m = 0
        p = {}
        for j in epochs:
            f = open('{}/run{}/result{}.pkl'.format(path, i, j), 'rb')
            r = pickle.load(f)
            if len(p.keys()) == 0:
                for k in r['pitch_acc']:
                    p[k] = 0
            
            m = max(m,r['acc'])
            for k in r['pitch_acc']:
                p[k] = max(p[k], r['pitch_acc'][k])
            f.close()
        res['acc'].append(m)
        for k in p:
            if k not in res.keys():
                res[k] = []
            res[k].append(p[k])
    return res

def plot_res(Ns, res, b):
    if b:
        plt.plot(Ns, res['acc'], '--o', label = 'Overall accuracy')
        plt.title('Best accuracy over number of instruments in orchestra')
    else:
        for i in res:
            if i != 'acc':
                plt.plot(Ns[-len(res[i]):], res[i], '--o', label=i)
        plt.title('Best per instrument accuracy over number of instruments in orchestra')
    plt.xlabel('Number of instruments')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.legend()