import scipy
from scipy import signal
import numpy as np

def GaussTemporalFilter(x, order=11):
    """ x: energy-normalized spectrum """

    window = signal.gaussian(order, std=order/5)
    window = window / np.sum(window)
    
    _x = np.zeros_like(x)
    for i in range(513):
        _x[:, i] = np.convolve(np.log10(x[:, i]), window, 'same')

    _x = np.power(10., _x)
    _x = _x / np.sum(_x, 1).reshape([-1, 1])

    return _x
    
    
def fast_MLGV(Input_seq,GV):
    LV = np.var(Input_seq,axis=0)
    X_mean = np.mean(Input_seq,axis=0)

    return np.sqrt(GV/LV)*(Input_seq-X_mean) + X_mean

def generalized_MLPG(Input_seq,Cov,dynamic_flag=2):
    # parameter for sequencial data
    T, sddim = Input_seq.shape
    # prepare W
    W = construct_dynamic_matrix(T, sddim//(dynamic_flag+1), dynamic_flag)
    # prepare U
    U = scipy.sparse.block_diag([Cov for i in range(T)], format='csr')
    U.eliminate_zeros()
    # calculate W'U
    WU = W.T.dot(U)
    # W'UW
    WUW = WU.dot(W)
    # W'Um
    WUm = WU.dot(Input_seq.flatten())
    # estimate y = (W'DW)^-1 * W'Dm
    odata = scipy.sparse.linalg.spsolve(
        WUW, WUm, use_umfpack=False).reshape(T, sddim//(dynamic_flag+1))
    # return odata
    return odata

def construct_dynamic_matrix(T, D, dynamic_flag=2):
    """
    Calculate static and delta transformation matrix

    Parameters
    ----------
    T : scala, `T`
        Scala of time length
    D : scala, `D`
        Scala of the number of dimentsion

    Returns
    -------
    W : array, shape (`2(or3) * D * T`, `D * T`)
        Array of static and delta transformation matrix.
    """
    static = [0, 1, 0]
    delta = [-0.5,0,0.5]
    delta2 = [1,-2,1]
    assert len(static) == len(delta)

    # generate full W
    DT = D * T
    ones = np.ones(DT)
    col = np.arange(DT)

    if dynamic_flag < 2:
        row = np.arange(2 * DT).reshape(2 * T, D)
        static_row = row[::2]
        delta_row = row[1::2]

        data = np.array([   ones * static[0], ones * static[1],ones * static[2], 
                            ones * delta[0], ones * delta[1], ones * delta[2]]).flatten()
        row = np.array([[static_row] * 3,  [delta_row] * 3]).flatten()
        col = np.array([[col - D, col, col + D] * 2]).flatten()
    else:
        row = np.arange(3 * DT).reshape(3 * T, D)
        static_row = row[::3]
        delta_row = row[1::3]
        delta2_row = row[2::3]

        data = np.array([   ones * static[0], ones * static[1],ones * static[2], 
                            ones * delta[0], ones * delta[1], ones * delta[2],
                            ones * delta2[0], ones * delta2[1], ones * delta2[2],]).flatten()
        row = np.array([[static_row] * 3,  [delta_row] * 3, [delta2_row] * 3]).flatten()
        col = np.array([[col - D, col, col + D] * 3]).flatten()

    # remove component at first and end frame
    valid_idx = np.logical_not(np.logical_or(col < 0, col >= DT))

    W = scipy.sparse.csr_matrix(
        (data[valid_idx], (row[valid_idx], col[valid_idx])), shape=((dynamic_flag+1) * DT, DT))
    W.eliminate_zeros()

    return W
