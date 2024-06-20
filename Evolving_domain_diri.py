
# # Schnakenberg model
# +
from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import scipy


import matplotlib.pyplot as plt
import matplotlib.colors as mplc
from matplotlib import cm

import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, log, plot, mesh
from dolfinx.fem import Function, functionspace, assemble_matrix, form,dirichletbc, Function, locate_dofs_geometrical
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import GhostMode
from dolfinx.nls.petsc import NewtonSolver
from petsc4py.PETSc import ScalarType
from ufl import dx, grad, inner

def boundary_D(x):
    return np.isclose(x[0], 0)

def update_msh(m, v0, v1, midpoint):
    x_cord = m.geometry.x[:,0]
    mov = np.zeros(np.shape(x_cord))

    x_2 = x_cord[-1]
    x_1 = x_cord[0]
    

    mov[:midpoint] = v0*(x_cord[:midpoint] - x_cord[midpoint])/(x_cord[midpoint]-x_1)
    mov[midpoint+1:] = v1*(x_cord[midpoint+1:] - x_cord[midpoint])/(x_2-x_cord[midpoint])


    m.geometry.x[:,0] = m.geometry.x[:,0]+mov
    
    
    
def run_simulation(d=None, gamma=None):
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
    step_number = 3000      # time step number
    step_ini = 1000
    
    time_range = np.linspace(0, step_number* dt,  step_number+1)
    time_ini = np.linspace(0, step_ini* dt,  step_ini+1)
    norm_stop = 0.1e-6      
    
    cell_number = 100       # cell number for the mesh
    v_0 = 0.       # speed of left pole
    v_1 = 0.0006           # speed of right pole before neto
    v_1_bis = 0.0006        # speed of right pole after neto
    mid_point = cell_number//2

    uv_array = np.zeros((step_number+1, cell_number+1, 2))  # initalize solutions
    x_array = np.zeros((step_number+1, cell_number+1))      # initalize mesh geometry

    #initial conditions
    au = 1
    bound1 = au +0.1
    bu = 1.1
    av = 0.9
    bound2 = av -0.1
    bv = 0.95

    # Parameters for weak statement of the equations

    k = dt
    if d is None:
        d = 10
    if gamma is None:
        gamma = 100
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
    
    
    #extracting stiffness matrix
    ME_test = functionspace(msh, P1)
    q_test = ufl.TestFunction(ME_test)
    u_test = ufl.TrialFunction(ME_test)
    a_mass_mat =form( u_test * q_test * dx)
    mass_mat = assemble_matrix(a_mass_mat)
    old_mass = mass_mat.to_dense()

    # Trial and test functions of the space `ME` are now defined:

    q, v = ufl.TestFunctions(ME)

    u = Function(ME)  # current solution
    u0 = Function(ME)  # solution from previous converged step
    
    # Split mixed functions

    u1, u2 = ufl.split(u)
    u10, u20 = ufl.split(u0)


    # Interpolate initial condition
    u.sub(0).interpolate(lambda x: (bu-au)*np.random.rand(x.shape[1]) +au)
    u.sub(1).interpolate(lambda x: (bv-av)*np.random.rand(x.shape[1]) +av)

    u.x.scatter_forward()
    for i, t in enumerate(time_ini[1:]):
        
        u0.x.array[:] = u.x.array
        
        
        u1, u2 = ufl.split(u)
        u10, u20 = ufl.split(u0)
        
        ME0, _ = ME.sub(0).collapse()
        ME1, _ = ME.sub(1).collapse()
        dofs_D1 = locate_dofs_geometrical((ME.sub(0), ME0), boundary_D)
        dofs_D2 = locate_dofs_geometrical((ME.sub(1), ME1), boundary_D)
        bc_1 = dirichletbc(ScalarType(bound1), dofs_D1[0], ME.sub(0))
        bc_2 = dirichletbc(ScalarType(bound2), dofs_D2[0], ME.sub(1))

        # Weak formulation (specific term for mass matrix variation)
        F = u1/k*q*dx + d1*inner(grad(u1), grad(q))*dx\
            -(gamma*(u1*u1*u2-u1+a))*q*dx \
            - (u10/k)*q*dx \
            + u2/k*v*dx + d2*inner(grad(u2), grad(v))*dx\
            -(gamma*(-u1*u1*u2+b))*v*dx \
            - (u20/k)*v*dx


        # Create nonlinear problem and Newton solver
        problem = NonlinearProblem(F, u, [bc_1, bc_2])
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
        print(f"Initial Step {i}: num iterations: {r[0]}")
        
        # reporting values
        u1 = u.sub(0).collapse()
        u2 = u.sub(1).collapse()
        
        



        if np.linalg.norm(u.x.array-u0.x.array)/dt < norm_stop:
            print("l2_norm convergence")
            break
    



    

    
    # Output file
    file1 = XDMFFile(MPI.COMM_WORLD, "Schnakenberg1D/outputu1.xdmf", "w")
    file1.write_mesh(msh)
    file2 = XDMFFile(MPI.COMM_WORLD, "Schnakenberg1D/outputu2.xdmf", "w")
    file2.write_mesh(msh)

    # reporting values

    u1 = u.sub(0).collapse()
    
    u2 = u.sub(1).collapse()
    file1.write_function(u1, 0, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']")
    file2.write_function(u2, 0, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']")
    u0.x.array[:] = u.x.array
    
    x_array[0] = msh.geometry.x[:,0]
    uv_array[0,:,0] = u1.x.array
    uv_array[0,:,1] = u2.x.array




    for i, t in enumerate(time_range[1:]):
        
        
        
        # Split mixed functions

        ME_test = functionspace(msh, P1)
        q_test = ufl.TestFunction(ME_test)
        u_test = ufl.TrialFunction(ME_test)
        a_mass_mat =form( u_test * q_test * dx)
        mass_mat = assemble_matrix(a_mass_mat)
        new_mass = mass_mat.to_dense()
        u10 = u0.sub(0).collapse()
        u20 = u0.sub(1).collapse()
        inv_new = scipy.linalg.inv(new_mass)
        u10.x.array[:] = inv_new @ old_mass @ u10.x.array
        u20.x.array[:] = inv_new @ old_mass @ u20.x.array
        
        u1, u2 = ufl.split(u)
        u10, u20 = ufl.split(u0)
        
        ME0, _ = ME.sub(0).collapse()
        ME1, _ = ME.sub(1).collapse()
        dofs_D1 = locate_dofs_geometrical((ME.sub(0), ME0), boundary_D)
        dofs_D2 = locate_dofs_geometrical((ME.sub(1), ME1), boundary_D)
        bc_1 = dirichletbc(ScalarType(bound1), dofs_D1[0], ME.sub(0))
        bc_2 = dirichletbc(ScalarType(bound2), dofs_D2[0], ME.sub(1))
        
        
        # Weak formulation (specific term for mass matrix variation)
        F = u1/k*q*dx + d1*inner(grad(u1), grad(q))*dx\
            -(gamma*(u1*u1*u2-u1+a))*q*dx \
            - (u10/k)*q*dx \
            + u2/k*v*dx + d2*inner(grad(u2), grad(v))*dx\
            -(gamma*(-u1*u1*u2+b))*v*dx \
            - (u20/k)*v*dx


        # Create nonlinear problem and Newton solver
        problem = NonlinearProblem(F, u, [bc_1, bc_2])
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
        u1 = u.sub(0).collapse()
        u2 = u.sub(1).collapse()
        x_array[i] = msh.geometry.x[:,0]
        uv_array[i,:,0] = u1.x.array
        uv_array[i,:,1] = u2.x.array
        
        u0.x.array[:] = u.x.array
        # updating mesh geometry
        if t<= step_number* dt/2:
            update_msh(msh, v_0, v_1, mid_point)
        else:
            update_msh(msh, v_0, v_1_bis, mid_point)
        old_mass = new_mass
        

        # Save solution to file (VTK)

        name = "mesh_at_t"+str(t)
        msh.name = name

        file1.write_mesh(msh)
        file1.write_function(u1, t, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']")
        file2.write_mesh(msh)
        file2.write_function(u2, t, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']")

    


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
    fig, ax = plt.subplots()
    plt.title('U')
    for i in T_steps:
        ax.scatter(x_array[i][x_steps], time_range[i]* np.ones(len(x_steps)), c=uv_array[i,:,0][x_steps], cmap="viridis", edgecolor='none', norm=mplc.Normalize(vmin=umin, vmax=umax))
    ax.set_xlabel('x')
    ax.set_ylabel('T')
    fig.colorbar(cm.ScalarMappable(norm=mplc.Normalize(vmin=umin, vmax=umax), cmap="viridis"), ax=ax)
    
    # plotting V
    fig, ax = plt.subplots()
    plt.title('V')
    for i in T_steps:
        ax.scatter(x_array[i][x_steps], time_range[i]* np.ones(len(x_steps)), c=uv_array[i,:,1][x_steps], cmap="viridis", edgecolor='none', norm=mplc.Normalize(vmin=vmin, vmax=vmax)) 
    ax.set_xlabel('x')
    ax.set_ylabel('T')
    fig.colorbar(cm.ScalarMappable(norm=mplc.Normalize(vmin=umin, vmax=umax), cmap="viridis"), ax=ax)
    # plt.show()
    
    
    
    
def main():
    # param = [(10,29),(10,100), (10, 250), (10, 470), (9, 700), (9, 1000), (8.6, 1400), (8.6, 1900)]
    # for elem in param :
    #     time_range, x_array, uv_array = run_simulation(d=elem[0], gamma=elem[1])
    #     plot(time_range, x_array, uv_array, 100, 50)
    # plt.show()
    
    time_range, x_array, uv_array = run_simulation(d=15,gamma=500)
    plot(time_range, x_array, uv_array, 200, 50)
    plt.show()
    
    
if __name__ == '__main__':
    main()


