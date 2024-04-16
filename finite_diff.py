import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import scipy.optimize as opt
from tqdm import trange

import matplotlib.colors as mplc

 

 

def plot_res(x_mesh, T_mesh, res_mat, exp_name):
    T_steps = np.linspace(0, len(T_mesh)-1, 50, dtype=int)
    x_steps = np.linspace(0, len(x_mesh)-1, 500, dtype=int)
    vmin = np.min(res_mat[:,:,1][T_steps][:,x_steps])
    vmax = np.max(res_mat[:,:,1][T_steps][:,x_steps])
    umin = np.min(res_mat[:,:,0][T_steps][:,x_steps])
    umax = np.max(res_mat[:,:,0][T_steps][:,x_steps])
    
    ax = plt.figure().add_subplot(projection='3d')
    plt.title(exp_name+', U')
    for i in T_steps:
        ax.scatter(x_mesh[x_steps], res_mat[i,:,0][x_steps], zs=T_mesh[i], zdir='z', c=res_mat[i,:,0][x_steps], cmap="viridis", edgecolor='none', norm=mplc.Normalize(vmin=umin, vmax=umax))
    ax.set_xlabel('x')
    ax.set_ylabel('U(x)')
    ax.set_zlabel('T')
    
    ax = plt.figure().add_subplot(projection='3d')
    plt.title(exp_name+',V')
    for i in T_steps:
        ax.scatter(x_mesh[x_steps], res_mat[i,:,1][x_steps], zs=T_mesh[i], zdir='z', c=res_mat[i,:,1][x_steps], cmap="viridis", edgecolor='none', norm=mplc.Normalize(vmin=vmin, vmax=vmax))
    ax.set_xlabel('x')
    ax.set_ylabel('V(x)')
    ax.set_zlabel('T')
    

 
 
#%% Forward_Euler

def Forward_Euler(x_mesh, T_mesh, D_mat, gamma_val, af, bf, init_cond):
    """Compute forward euler scheme in 1D

    Args:
        x_mesh (array): space mesh (constant step)
        T_mesh (1D array): time mesh (constant step)
        D_mat (2x2 array): diffusion matrix
        F (function): non linear function of the RD system
        gamma_val (float): gamma value of RD system
        af, bf (float): parameter of the model
        init_cond (array): iniial conditions, same size as x_mesh

    Raises:
        ValueError: unappropriate time steps

    Returns:
        U (array): solutions of the PDE
    """
    
    x_number = len(x_mesh)
    x_step = (x_mesh[-1] - x_mesh[0]) / x_number
    T_number = len(T_mesh)
    T_step = (T_mesh[-1] - T_mesh[0]) / T_number
    
    mu = T_step / (x_step**2)
    print('mu :',mu)
    if mu * np.max(D_mat) > 0.5:
        raise ValueError('unstable scheme, chose a smaller time step')
    
    U = np.zeros((T_number, x_number, 2))
    U[0] = init_cond

    U = iter_scheme_fwd(U, T_number, D_mat, gamma_val, T_step, af, bf, mu)
    return U


@njit
def iter_scheme_fwd(U, T_number, D_mat, gamma_val, T_step, af, bf, mu):
    for i in range(1, T_number):
        U[i] = (1-2*mu*D_mat) * U[i-1] + gamma_val *  T_step * F_Schnack_fwd(U[i-1], af, bf)
        U[i,0] += 2*mu*D_mat * U[i-1,1]
        U[i,-1] += 2*mu*D_mat * U[i-1,-2]
        U[i,1:-1] += mu*D_mat * (U[i-1,:-2] + U[i-1,2:])
    return U


@njit
def F_Schnack_fwd(U, af, bf):
    res = np.column_stack((af - U[:,0] + U[:,0]**2 * U[:,1], bf - U[:,0]**2 * U[:,1]))
    return res
        

def init_cond_func(x_mesh):
    x_number = len(x_mesh)
    x_step = (x_mesh[0] - x_mesh[-1]) / x_number
    dim = list(np.shape(x_mesh))
    dim.append(2)
    res = np.ones(tuple(dim))
    res[:,0] =1
    res[:,1] =0.9
    res += x_step**2 * np.random.rand(*dim) 
    return res
 
 
#%% Fully Backward Euler

def Backward_Euler(x_mesh, T_mesh, D_mat, gamma_val, af, bf, init_cond):
    """Compute backward euler scheme in 1D

    Args:
        x_mesh (array): space mesh (constant step)
        T_mesh (1D array): time mesh (constant step)
        D_mat (2x2 array): diffusion matrix
        F (function): non linear function of the RD system
        gamma_val (float): gamma value of RD system
        af, bf (float): parameter of the model
        init_cond (array): iniial conditions, same size as x_mesh
        

    Returns:
        U (array): solutions of the PDE
    """
    
    x_number = len(x_mesh)
    x_step = (x_mesh[-1] - x_mesh[0]) / x_number
    T_number = len(T_mesh)
    T_step = (T_mesh[-1] - T_mesh[0]) / T_number
    
    mu = T_step / (x_step**2)
    
    U = np.zeros((T_number, x_number, 2))
    U[0] = init_cond
    
    
    for i in trange(1, T_number):
        l =len(U[i-1])
        ini_cond = np.zeros(2*l)
        ini_cond[:l] = U[i-1,:,0]
        ini_cond[l:] = U[i-1,:,1]
        op_res = opt.minimize(score_func_bwd, ini_cond, args=(U[i-1], D_mat, gamma_val, T_step, af, bf, mu))
        if not op_res.success:
            print('problem in optimization')
        U[i,:,0] = op_res.x[:l]
        U[i,:,1] = op_res.x[l:]
        
    return U

@njit
def score_func_bwd(V, U, D_mat, gamma_val, T_step, af, bf, mu):
    l=len(V)//2
    V = np.column_stack((V[:l],V[l:]))
    W = (1+2*mu*D_mat) * V + gamma_val *  T_step * F_Schnack_fwd(V, af, bf)
    W[0] -= 2*mu*D_mat * V[1]
    W[-1] -= 2*mu*D_mat * V[-2]
    W[1:-1] -= mu*D_mat * (V[:-2] + V[2:])

    return np.sum((W-U)**2)


#%% Crank Nicholson

def Crank_Nicholson(x_mesh, T_mesh, D_mat, gamma_val, af, bf, init_cond):
    """Compute Crank Nicholson scheme in 1D

    Args:
        x_mesh (array): space mesh (constant step)
        T_mesh (1D array): time mesh (constant step)
        D_mat (2x2 array): diffusion matrix
        F (function): non linear function of the RD system
        gamma_val (float): gamma value of RD system
        af, bf (float): parameter of the model
        init_cond (array): iniial conditions, same size as x_mesh

    Returns:
        U (array): solutions of the PDE
    """
    
    x_number = len(x_mesh)
    x_step = (x_mesh[-1] - x_mesh[0]) / x_number
    T_number = len(T_mesh)
    T_step = (T_mesh[-1] - T_mesh[0]) / T_number
    
    mu = T_step / (x_step**2)
    
    U = np.zeros((T_number, x_number, 2))
    U[0] = init_cond
    
        
        
    for i in trange(1, T_number):
        l =len(U[i-1])
        ini_cond = np.zeros(2*l)
        ini_cond[:l] = U[i-1,:,0]
        ini_cond[l:] = U[i-1,:,1]
        
        op_res = opt.minimize(score_func_CN, ini_cond, args=(U[i-1], D_mat, gamma_val, T_step, af, bf, mu))
        if not op_res.success:
            print('problem in optimization')
        else : 
            print('converged optimization')
        U[i,:,0] = op_res.x[:l]
        U[i,:,1] = op_res.x[l:]
        
    return U

@njit
def score_func_CN(V, U, D_mat, gamma_val, T_step, af, bf, mu):
    
    l=len(V)//2
    V = np.column_stack((V[:l],V[l:]))
    
    W =  -mu*D_mat*(U+V) - V + U +  0.5 *gamma_val *  T_step * (F_Schnack_fwd(V, af, bf)+F_Schnack_fwd(U, af, bf))
    W[0] += mu*D_mat * (V[1]+U[1])
    W[-1] += mu*D_mat * (V[-2]+U[-2])
    W[1:-1] += 0.5*mu*D_mat * (V[:-2] + V[2:] + U[:-2] + U[2:])

    return np.sum(W**2)


#%% IMEX1

def IMEX1(x_mesh, T_mesh, D_mat, gamma_val, af, bf, init_cond):
    """Compute IMEX scheme in 1D (only the diffusion is implicit)

    Args:
        x_mesh (array): space mesh (constant step)
        T_mesh (1D array): time mesh (constant step)
        D_mat (2x2 array): diffusion matrix
        F (function): non linear function of the RD system
        gamma_val (float): gamma value of RD system
        af, bf (float): parameter of the model
        init_cond (array): iniial conditions, same size as x_mesh

    Returns:
        U (array): solutions of the PDE
    """
    
    x_number = len(x_mesh)
    x_step = (x_mesh[-1] - x_mesh[0]) / x_number
    T_number = len(T_mesh)
    T_step = (T_mesh[-1] - T_mesh[0]) / T_number
    
    mu = T_step / (x_step**2)
    
    U = np.zeros((T_number, x_number, 2))
    U[0] = init_cond
    
    
    U = iter_scheme_IMEX(U, T_number, x_number, mu, D_mat, gamma_val, T_step, af, bf)
        
    return U

 
@njit
def Thomas_algo(amat, bmat, cmat, vec):
    """
    Implementing Thomas algorithm for solving linear system with tri-diagonal matrices : tri_diag_M @ x = vec
    
    Args:
        amat, bmat, cmat (numpy matrix): diagonals of the tri-diagonal square matrix (of dimension more then 2). bmat is the real diagonal. Stability is ensured if it is diagonally dominant (either by rows or columns) or symmetric positive definite
        vec (numpy vector): target vector
    
    Returns:
        x_vec (array): solutions of the system
    """ 
    dim = len(bmat)
    c_vec = np.zeros(dim-1)
    d_vec = np.zeros(dim-1)
    x_vec = np.zeros(dim)
    
    c_vec[0] = cmat[0] / bmat[0]
    d_vec[0] = vec[0] / bmat[0]
    for i in range(1, dim-1):
        c_vec[i] = cmat[i] / (bmat[i] - amat[i-1] * c_vec[i-1])
        d_vec[i] = (vec[i] - amat[i-1] * d_vec[i-1]) / (bmat[i] - amat[i-1] * c_vec[i-1])
    
    x_vec[dim-1] = (vec[dim-1] - amat[dim-2] * d_vec[dim-2]) / (bmat[dim-1] - amat[dim-2] * c_vec[dim-2])
    for i in range(dim-2, -1, -1):
        x_vec[i] = d_vec[i] - c_vec[i] * x_vec[i+1]
    
    return x_vec


@njit
def iter_scheme_IMEX(U, T_number, x_number, mu, D_mat, gamma_val, T_step, af, bf):
    up_diagu = -mu*D_mat[0] *  np.ones(x_number-1)
    up_diagu[0] = - 2 * mu*D_mat[0]
    diagu = (2*mu*D_mat[0]+1) * np.ones(x_number)
    
    up_diagv = -mu*D_mat[1] *  np.ones(x_number-1)
    up_diagv[0] = - 2 * mu*D_mat[1]
    diagv = (2*mu*D_mat[1]+1) * np.ones(x_number)
    
    for i in range(1, T_number):
        V =  gamma_val *  T_step * F_Schnack_fwd(U[i-1], af, bf) + U[i-1]
        U[i,:,0] = Thomas_algo(up_diagu[::-1], diagu, up_diagu, V[:,0])
        U[i,:,1] = Thomas_algo(up_diagv[::-1], diagv, up_diagv, V[:,1])
    return U


#%% IMEX2

def IMEX2(x_mesh, T_mesh, D_mat, gamma_val, af, bf, init_cond):
    """Compute IMEX scheme in 1D (the diffusion and some part of the non linear term are implicit)

    Args:
        x_mesh (array): space mesh (constant step)
        T_mesh (1D array): time mesh (constant step)
        D_mat (2x2 array): diffusion matrix
        F (function): non linear function of the RD system
        gamma_val (float): gamma value of RD system
        af, bf (float): parameter of the model
        init_cond (array): iniial conditions, same size as x_mesh

    Returns:
        U (array): solutions of the PDE
    """
    
    x_number = len(x_mesh)
    x_step = (x_mesh[-1] - x_mesh[0]) / x_number
    T_number = len(T_mesh)
    T_step = (T_mesh[-1] - T_mesh[0]) / T_number
    
    mu = T_step / (x_step**2)
    
    U = np.zeros((T_number, x_number, 2))
    U[0] = init_cond
    
    U = iter_scheme_IMEX2(U, T_number, x_number, gamma_val, T_step, af, bf, mu, D_mat)
        
    return U


@njit
def iter_scheme_IMEX2(U, T_number, x_number, gamma_val, T_step, af, bf, mu, D_mat):

    for i in range(1, T_number):
        up_diagu = -mu*D_mat[0] *  np.ones(x_number-1)
        up_diagu[0] = - 2 * mu*D_mat[0]
        diagu = (2*mu*D_mat[0]+1+gamma_val*T_step) * np.ones(x_number) - gamma_val*T_step * U[i-1,:,0] * U[i-1,:,1]
        
        up_diagv = -mu*D_mat[1] *  np.ones(x_number-1)
        up_diagv[0] = - 2 * mu*D_mat[1]
        diagv = (2*mu*D_mat[1]+1) * np.ones(x_number) + gamma_val*T_step * U[i-1,:,0]**2
    
        V =  U[i-1] + gamma_val *  T_step * np.array([af,bf])
        
        U[i,:,0] = Thomas_algo(up_diagu[::-1], diagu, up_diagu, V[:,0])
        U[i,:,1] = Thomas_algo(up_diagv[::-1], diagv, up_diagv, V[:,1])
    return U


def main():
    d=10
    D = np.array([1,d])
    gamma = 100
    a = 0.1
    b = 0.9
    mesh = np.linspace(0,10,1000)
    time = np.linspace(0,5,10000)
    time_fwd = np.linspace(0,5,2000000)
    
    init_conditions = init_cond_func(mesh)
    
    solution = Forward_Euler(mesh, time_fwd, D, gamma, a, b, init_conditions)
    plot_res(mesh, time_fwd, solution, 'fwd euler')
    
    # solution = Backward_Euler(mesh, time, D, gamma, a, b, init_conditions)
    # plot_res(mesh, time, solution, 'bwd euler')
    
    # solution = Crank_Nicholson(mesh, time, D, gamma, a, b, init_conditions)
    # plot_res(mesh, time, solution, 'Crank Nicholson')
    
    solution = IMEX1(mesh, time, D, gamma, a, b, init_conditions)
    plot_res(mesh, time, solution, 'IMEX1')
    
    solution = IMEX2(mesh, time, D, gamma, a, b, init_conditions)
    plot_res(mesh, time, solution, 'IMEX2')
    
    plt.show()

if __name__ == '__main__':
    main()
    
    
    