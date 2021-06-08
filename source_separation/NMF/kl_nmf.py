# kl_nmf.py
# From https://github.com/CarmineCella/nmf_sources

import numpy as np

EPS = np.finfo(np.float32).eps  # as used in sklearn (nmf.py)

def kl_nmf(X, k=10, it=100):
    # VERIFIED with sklearn.decomposition.nmf

    nr = X.shape[0]
    nc = X.shape[1]

    # init matrices (sklearn method):
    avg = np.sqrt(X.mean() / k)
    rng = np.random.RandomState(0)  # fixed random seed
    H = avg * rng.randn(k, nc)
    W = avg * rng.randn(nr, k)
    np.abs(W, W)
    np.abs(H, H)

    I = np.ones([nr,nc])
    for i in range(it):
        # update W
        WH = np.dot(W,H)
        WH[WH == 0] = EPS
        N = np.dot(np.divide(X,WH),H.T)
        P = np.dot(I,H.T)
        P[P == 0] = EPS
        W *= np.divide(N,P)
        #update H
        WH = np.dot(W,H)
        WH[WH == 0] = EPS
        N = np.dot(W.T,np.divide(X,WH))
        P = np.dot(W.T,I)
        P[P == 0] = EPS
        H *= np.divide(N,P)

    return W,H
