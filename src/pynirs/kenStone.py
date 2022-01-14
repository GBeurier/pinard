import numpy as np


def skdist(X, precomputed=False):
    from sklearn import metrics
    if precomputed:
        return X
    return metrics.pairwise_distances(X, metric='euclidean', n_jobs=-1)

def scipydist(X, precomputed=False):
    from scipy.spatial import distance
    if precomputed:
        return X
    return distance.squareform(distance.pdist(X, metric='euclidean'))

def loadKS(input):
    try:
        X = np.loadtxt(input, delimiter="\t")
    except:
        import pandas as pd
        X = pd.read_table(input, delim_whitespace=True, header=None)
        return X.values
    return X

def kenStone(X, k, precomputed=False, verbose=False):
    n = len(X) # number of samples
    if verbose:
        print("Input Size:", n, "Desired Size:", k)
    assert n >= 2 and n >= k and k >= 2, "Error: number of rows must >= 2, k must >= 2 and k must > number of rows"
    # pair-wise distance matrix
    dist = skdist(X, precomputed)

    # get the first two samples
    i0, i1 = np.unravel_index(np.argmax(dist, axis=None), dist.shape)
    selected = set([i0, i1])
    k -= 2
    # iterate find the rest
    minj = i0
    while k > 0 and len(selected) < n:
        mindist = 0.0
        for j in range(n):
            if j not in selected:
                mindistj = min([dist[j][i] for i in selected])
                if mindistj > mindist:
                    minj = j
                    mindist = mindistj
        if verbose:
            print(selected, minj, [dist[minj][i] for i in selected])
        selected.add(minj)
        k -= 1
    if verbose:
        print("selected samples indices: ", selected)
    # return selected samples
    if precomputed:
        return list(selected)
    else:
        return X[list(selected), :]

def writeKS(output, X, precomputed=False):
    if precomputed:
        np.savetxt(output, X, fmt='%d')
    else:
        np.savetxt(output, X, fmt='%.5f')

def test():
    # take features
    input = 'test/distArray.txt'
    X = loadKS(input)
    Y = kenStone(X, 10)
    writeKS('test/KSfeatures.txt', Y)

    # precomputed
    input = 'test/matrix.txt'
    X = loadKS(input)
    Y = kenStone(X, 10, precomputed=True)
    writeKS('test/KSelected.txt', Y, precomputed=True)

