import numpy as np
from sklearn import neighbors
import sklearn
from sklearn.neighbors import KDTree


def distance_squarred(xj, xi):
    return np.sum((xi-xj)**2)


def compute_kernel(tree: sklearn.neighbors.KDTree, xj: np.ndarray, xi: np.ndarray,  k: int, bandwidth: float, type :str) -> np.float64:

    xj, xi = np.array(xj), np.array(xi)
    if (xj.ndim != 1 or xi.ndim != 1):
        raise ValueError('Enter one dimensional array ')

    dk_squarred = tree.query(np.array(xi) . reshape(1, -1), k=k)[0][0][-1]**2
    rdk_squarred = max(dk_squarred, distance_squarred(xj, xi))
    return np.exp(-0.5*rdk_squarred/(bandwidth**2*dk_squarred))/(dk_squarred**(len(xi)/2))


def m_nearest_neighbors(tree: sklearn.neighbors.KDTree, xi: np.ndarray, m: int) -> tuple:

    dist, ind = tree.query(np.array(xi) . reshape(1, -1), k=m)
    return (dist, ind)


def LDE(tree: sklearn.neighbors.KDTree, xj: np.ndarray, m: int, data: np.ndarray,  k: int,
        bandwidth: float, type : str ) -> tuple:
    _, indices = m_nearest_neighbors(tree, xj, m)
    kernel_computation = []
    for idx in indices[0]:
        kernel_computation.append(compute_kernel(
            tree,xj, data[idx], k, bandwidth, type))
    return indices,np.array(kernel_computation).sum()/(bandwidth**len(xj) * m*((2*np.pi)**(len(xj)/2)))

def build_tree (data : np.ndarray) -> sklearn.neighbors.KDTree:
    return (KDTree(data))

def LDF(tree : sklearn.neighbors.KDTree, xj: np.ndarray, m: int, data: np.ndarray,  k: int,
        bandwidth: float, type= 'gaussian', c = 0.1 ) -> np.float64 :
    neighbors , lde = LDE (tree,xj,m,data ,k, bandwidth, type) 
    neighbour_lde =[]
    for idx in neighbors[0] : 
        neighbour_lde.append (LDE(tree, data [idx], m,data , k ,bandwidth, type)[1])
    sum = np.array(neighbour_lde).sum()
    return (sum/(lde*m + c*sum))

def compute_scores (X: np.ndarray , data: np.ndarray,  m = 30,  k = 10 ,
        bandwidth = 1 , c=0.1 ,type='gaussian') -> np.ndarray:
    result = []
    tree = build_tree(data)
    for xj in X :
        result.append (LDF(tree,xj,m,data,k,bandwidth,type , c ))
    return (np.array(result))
