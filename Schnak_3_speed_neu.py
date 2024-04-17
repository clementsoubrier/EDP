
# # Schnakenberg model
# +
import os

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.colors as mplc


import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, log, plot, mesh
from dolfinx.fem import Function, functionspace, Expression, FunctionSpaceBase
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_unit_square, GhostMode
from dolfinx.nls.petsc import NewtonSolver
from ufl import dx, grad, inner, SpatialCoordinate


def update_msh(m, v0, v1, midpoint0, midpoint1):
    x_cord = m.geometry.x[:,0]
    mov = np.zeros(np.shape(x_cord))

    x_2 = x_cord[-1]
    x_1 = x_cord[0]
    

    mov[:midpoint0] = v0*(x_cord[:midpoint0] - x_cord[midpoint0])/(x_cord[midpoint0]-x_1)
    mov[midpoint1+1:] = v1*(x_cord[midpoint1+1:] - x_cord[midpoint1])/(x_2-x_cord[midpoint1])


    m.geometry.x[:,0] = m.geometry.x[:,0]+mov
    
    
    
def run_simulation():
    """Simulation of RD system on evolving domains

    Returns:
        time_range (1d array): time of the solutions
        x_array (2d array): mesh geometry over time
        uv_array (3d array): values of solutions over time
    """
    
    ''' Parameters '''
    # Save all logging to file
    log.set_output_file("log.txt")
    # -
    
    # Next, various model parameters are defined:

    dt = 5.0e-04            # time step
    step_number = 1500      # time step number
    time_range = np.linspace(0, step_number* dt,  step_number+1)
    norm_stop = 0.1e-6      
    
    cell_number = 1000       # cell number for the mesh
    v_0 = 0.0007     # speed of left pole
    v_1 = 0.0003           # speed of right pole
    v_1_bis = 0.0007
    mid_point_l = cell_number//5
    mid_point_r = cell_number//5*4

    uv_array = np.zeros((step_number+1, cell_number+1, 2))  # initalize solutions
    x_array = np.zeros((step_number+1, cell_number+1))      # initalize mesh geometry

    #initial conditions
    au = 1
    bu = 1.1
    av = 0.9
    bv = 0.95

    # Parameters for weak statement of the equations

    k = dt
    d = 10
    gamma = 1500
    a = 0.1
    b = 0.9

    d1 = 1.0
    d2 = d


    # mesh and linear lagrange elements are created

    msh = mesh.create_unit_interval(comm=MPI.COMM_WORLD,
                                nx=cell_number,
                                ghost_mode=GhostMode.shared_facet)
    P1 = element("Lagrange", msh.basix_cell(), 1)
    ME = functionspace(msh, mixed_element([P1, P1]))

    # Trial and test functions of the space `ME` are now defined:

    q, v = ufl.TestFunctions(ME)

    u = Function(ME)  # current solution
    u0 = Function(ME)  # solution from previous converged step
    m_mass_var = Function(ME)  # function for variation of mass matrix
    
    # Split mixed functions

    u1, u2 = ufl.split(u)
    u10, u20 = ufl.split(u0)
    m_mass_var_1, m_mass_var_2 = ufl.split(m_mass_var)


    # Interpolate initial condition
    u.sub(0).interpolate(lambda x: (bu-au)*np.random.rand(x.shape[1]) +au)
    u.sub(1).interpolate(lambda x: (bv-av)*np.random.rand(x.shape[1]) +av)

    u.x.scatter_forward()
    



    
    # Output file
    file1 = XDMFFile(MPI.COMM_WORLD, "Schnakenberg1D/outputu1.xdmf", "w")
    file1.write_mesh(msh)
    file2 = XDMFFile(MPI.COMM_WORLD, "Schnakenberg1D/outputu2.xdmf", "w")
    file2.write_mesh(msh)

    # reporting values
    l2_norm = []
    u1 = u.sub(0).collapse()
    u2 = u.sub(1).collapse()
    file1.write_function(u1, 0, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']")
    file2.write_function(u2, 0, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']")
    u0.x.array[:] = u.x.array
    
    x_array[0] = msh.geometry.x[:,0]
    uv_array[0,:,0] = u1.x.array
    uv_array[0,:,1] = u2.x.array




    for i, t in enumerate(time_range[1:]):
        
        x_m = msh.geometry.x[:,0]
        m_mass_var.x.array[:mid_point_l] = (1+v_0/(x_m[mid_point_l]-x_m[0]))**(-2)
        m_mass_var.x.array[mid_point_l:mid_point_r] = 1
        m_mass_var.x.array[mid_point_r:] = (1+v_1/(x_m[-1]-x_m[mid_point_r]))**(-2)
        
            # Split mixed functions

        u1, u2 = ufl.split(u)
        u10, u20 = ufl.split(u0)
        m_mass_var_1, m_mass_var_2 = ufl.split(m_mass_var)
        
        # # Weak formulation (original)
        # F = ((u1 - u10) / k)*q*dx + d1*inner(grad(u1), grad(q))*dx\
        # -(gamma*(u1*u1*u2-u1+a))*q*dx \
        # + ((u2 - u20) / k)*v*dx + d2*inner(grad(u2), grad(v))*dx\
        # -(gamma*(-u1*u1*u2+b))*v*dx 
        
        # Weak formulation (specific term for mass matrix variation)
        F = u1/k*q*dx + d1*inner(grad(u1), grad(q))*dx\
            -(gamma*(u1*u1*u2-u1+a))*q*dx \
            - (u10/k*m_mass_var_1)*q*dx \
            + u2/k*v*dx + d2*inner(grad(u2), grad(v))*dx\
            -(gamma*(-u1*u1*u2+b))*v*dx \
            - (u20/k*m_mass_var_2)*v*dx


        # Create nonlinear problem and Newton solver
        problem = NonlinearProblem(F, u)
        solver = NewtonSolver(MPI.COMM_WORLD, problem)
        solver.convergence_criterion = "incremental"
        solver.rtol = np.sqrt(np.finfo(default_real_type).eps) * 1e-2
        ksp = solver.krylov_solver
        opts = PETSc.Options()  
        option_prefix = ksp.getOptionsPrefix()
        opts[f"{option_prefix}ksp_type"] = "preonly"
        opts[f"{option_prefix}pc_type"] = "lu"
        ksp.setFromOptions()

        i+=1
        # solving the variational problem
        r = solver.solve(u)
        print(f"Step {i}: num iterations: {r[0]}")
        
        # reporting values
        l2_norm.append(np.linalg.norm(u.x.array-u0.x.array)/dt)
        u1 = u.sub(0).collapse()
        u2 = u.sub(1).collapse()
        x_array[i] = msh.geometry.x[:,0]
        uv_array[i,:,0] = u1.x.array
        uv_array[i,:,1] = u2.x.array
        
        u0.x.array[:] = u.x.array
        # updating mesh geometry
        if t<= step_number* dt/2:
            update_msh(msh, v_0, v_1, mid_point_l, mid_point_r)
        else:
            update_msh(msh, v_0, v_1_bis, mid_point_l, mid_point_r)

        
        # Save solution to file (VTK)

        name = "mesh_at_t"+str(t)
        msh.name = name

        file1.write_mesh(msh)
        file1.write_function(u1, t, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']")
        file2.write_mesh(msh)
        file2.write_function(u2, t, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']")


        if l2_norm[-1] < norm_stop:
            print("l2_norm convergence")
            break
    


    file1.close()
    file2.close()
    
    return time_range, x_array, uv_array




def plot(time_range, x_array, uv_array, spacenum, timenum):
    """Plotting the solution of the PDE

    Args:
        time_range (1d array): time of the solutions
        x_array (2d array): mesh geometry over time
        uv_array (3d array): values of solutions over time
        spacenum (int): number of space steps for plot
        timenum (int): number of time steps for plot
    """
    # extracting data at the right place
    T_steps = np.linspace(0, len(time_range)-1, timenum, dtype=int)
    x_steps = np.linspace(0, len(x_array[0])-1, spacenum, dtype=int)
    
    vmin = np.min(uv_array[:,:,1][T_steps][:,x_steps])
    vmax = np.max(uv_array[:,:,1][T_steps][:,x_steps])
    umin = np.min(uv_array[:,:,0][T_steps][:,x_steps])
    umax = np.max(uv_array[:,:,0][T_steps][:,x_steps])
    
    # plotting U
    ax = plt.figure().add_subplot(projection='3d')
    plt.title('U')
    for i in T_steps:
        ax.scatter(x_array[i][x_steps], uv_array[i,:,0][x_steps], zs=time_range[i], zdir='z', c=uv_array[i,:,0][x_steps], cmap="viridis", edgecolor='none', norm=mplc.Normalize(vmin=umin, vmax=umax))
    ax.set_xlabel('x')
    ax.set_ylabel('U(x)')
    ax.set_zlabel('T')
    
    # plotting V
    ax = plt.figure().add_subplot(projection='3d')
    plt.title('V')
    for i in T_steps:
        ax.scatter(x_array[i][x_steps], uv_array[i,:,1][x_steps], zs=time_range[i], zdir='z', c=uv_array[i,:,1][x_steps], cmap="viridis", edgecolor='none', norm=mplc.Normalize(vmin=vmin, vmax=vmax)) 
    ax.set_xlabel('x')
    ax.set_ylabel('V(x)')
    ax.set_zlabel('T')
    plt.show()
    
    
    
    
def main():
    time_range, x_array, uv_array = run_simulation()
    plot(time_range, x_array, uv_array, 200, 50)
    
    
    
if __name__ == '__main__':
    main()


