import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from matplotlib.colors import ListedColormap

path = './outputs.npy'

viridis = cm.get_cmap('viridis')
prec = 0.2
cmap = ListedColormap([viridis(x**prec) for x in np.linspace(0,1,256)])

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax3 = fig.add_subplot(2,2,2)
ax2 = fig.add_subplot(2,1,2)

loss_array = []
pred_loss = []
avg_pred_loss = []
avg_length = 100

def getPosNMax(l, N):
    ind = [0]
    sort_key=lambda x : l[x]
    for j in range(1,len(l)):
        if len(ind) == N:
            if l[j] > l[ind[0]]:
                ind[0] = j
                ind.sort(key=sort_key)
        else:
            ind.append(j)
            ind.sort(key=sort_key)
    return ind
            

def prediction(outputs, labels, N):
    pred = np.zeros(outputs.shape)
    for i in range(pred.shape[0]):
        for j in getPosNMax(outputs[i],N):
            pred[i,j] = 1.0
    return pred

def load(i):
    r = np.load(path, allow_pickle=True).item()
    epoch = r.get('epoch')
    step = r.get('step')
    outputs = r.get('outputs')
    labels = r.get('labels')
    loss_array.append(r.get('loss'))
    
    ax2.cla()
    ax1.cla()
    ax3.cla()
    plt.title('Epoch {}; Step {}'.format(epoch, step))
    line = np.ones((1,outputs.shape[1]))
    line[0,0] = 0.0
    line[0,-1] = 0.0
    N = len([x for x in labels[0] if x > 0.0])
    pred = prediction(outputs, labels, N)
    pred_loss.append(np.sum(pred*labels)/(N*labels.shape[0]))
    wdw = min(len(pred_loss), avg_length)
    avg_pred_loss.append(sum(pred_loss[-wdw:])/wdw)
    ax2.imshow(np.vstack([outputs, line, labels, line, pred]), cmap=cmap)
    ax1.plot(loss_array)
    ax1.plot([len(loss_array)-1, len(loss_array)-1], [r.get('loss_min'), loss_array[-1]], marker='o')
    ax1.grid()
    ax1.set_title('BCE with logits loss')
    ax3.plot(pred_loss)
    ax3.plot(avg_pred_loss)
    ax3.set_title('Accuracy over batch')
    ax3.grid()
    #print(outputs)
    plt.tight_layout()

anim = FuncAnimation(plt.gcf(), load, interval=500)

plt.tight_layout()
plt.show()