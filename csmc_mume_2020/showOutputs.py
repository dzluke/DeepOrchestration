import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from matplotlib.colors import ListedColormap

from train import getPosNMax, prediction

path = './model/run4/outputs.npy'

viridis = cm.get_cmap('viridis')
prec = 0.2
cmap = ListedColormap([viridis(x**prec) for x in np.linspace(0,1,256)])

fig = plt.figure()
ax1 = fig.add_subplot(2,3,1)
ax3 = fig.add_subplot(2,3,2)
ax4 = fig.add_subplot(2,3,3)
ax2 = fig.add_subplot(2,1,2)

loss_array = []
pred_loss = []
avg_pred_loss = []
avg_length = 100

global prev
prev = (-1,-1)

def load(i):
    global prev
    r = np.load(path, allow_pickle=True).item()
    epoch = r.get('epoch')
    step = r.get('step')
    outputs = r.get('outputs')
    labels = r.get('labels')
    
    
    if (epoch, step) != prev:
        prev = (epoch, step)
        loss_array.append(r.get('loss'))
        ax2.cla()
        ax1.cla()
        ax3.cla()
        ax4.cla()
        plt.title('Epoch {}; Step {}'.format(epoch, step))
        line = np.ones((1,outputs.shape[1]))
        line[0,0] = 0.0
        line[0,-1] = 0.0
        N = len([x for x in labels[0] if x > 0.0])
        pred = prediction(outputs, N)
        pred_loss.append(np.sum(pred*labels)/max(1.0,np.sum(labels)))
        wdw = min(len(pred_loss), avg_length)
        ax1.plot(loss_array)
        ax1.plot([len(loss_array)-1, len(loss_array)-1], [r.get('loss_min'), loss_array[-1]], marker='o')
        ax1.grid()
        ax1.set_title('BCE with logits loss')
        avg_pred_loss.append(sum(pred_loss[-wdw:])/wdw)
        ax2.imshow(np.vstack([outputs, line, labels, line, pred]), cmap=cmap)
        ax3.plot(pred_loss)
        ax3.plot(avg_pred_loss)
        ax3.set_title('Accuracy over batch')
        ax3.grid()
        ax4.hist(pred_loss[-wdw:], bins=30)
        ax4.set_title('Accuracy histogram over last {} points'.format(wdw))
        ax4.grid()
        #print(outputs)
        plt.tight_layout()

anim = FuncAnimation(plt.gcf(), load, interval=500)

plt.tight_layout()
plt.show()

import pickle
def fr(run, n):
    f = open("./model/run{}/result{}.pkl".format(run, n), 'rb')
    r = pickle.load(f)
    f.close()
    return r