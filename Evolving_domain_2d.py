
# # Schnakenberg model
# +
import os

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import scipy

import matplotlib.pyplot as plt
import matplotlib.colors as mplc

import gmsh


import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, log, plot, mesh
from dolfinx.fem import Function, functionspace, Expression, FunctionSpaceBase, assemble_matrix, form, ElementMetaData
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import XDMFFile, gmshio
from dolfinx.mesh import CellType, create_unit_square, GhostMode
from dolfinx.nls.petsc import NewtonSolver
from ufl import dx, grad, inner, SpatialCoordinate


def create_2d_cell_mesh(name):
    """Create a mesh of a rod shape bacterium 

    Args:
        name: Name (identifier) of the mesh to add.

    Returns:
        dolfinx mesh

    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    # Create model
    model = gmsh.model()
    model.add(name)
    model.setCurrent(name)
    cyl_dim_tags = model.occ.addCylinder(-1.5, 0, 0, 3, 0, 0, 0.5)
    sphere1_dim_tags = model.occ.addSphere(-1.5, 0, 0, 0.5)
    sphere2_dim_tags = model.occ.addSphere(1.5, 0, 0, 0.5)
    model_dim_tags = model.occ.fuse([(3, cyl_dim_tags)], [(3, sphere1_dim_tags),(3, sphere2_dim_tags)])
    
    model.occ.synchronize()
    
    boundary = model.getBoundary(model_dim_tags[0], oriented=False)
    boundary_ids = [b[1] for b in boundary]
    model.addPhysicalGroup(2, boundary_ids, tag=1)
    model.setPhysicalName(2, 1, "Surface")
    
    gmsh.option.setNumber("Mesh.MeshSizeFactor", 0.5)
    model.mesh.generate(2)
    
    filename = f"out_gmsh/mesh_rank_{MPI.COMM_WORLD.rank}.xdmf"
    msh, ct, ft = gmshio.model_to_mesh(model, MPI.COMM_SELF, rank=0,gdim=3)
    msh.name = name
    ct.name = f"{msh.name}_cells"
    ft.name = f"{msh.name}_facets"
    with XDMFFile(msh.comm, filename, "w") as file:
        msh.topology.create_connectivity(1, 2)
        file.write_mesh(msh)
        file.write_meshtags(
            ct, msh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry"
        )
        file.write_meshtags(
            ft, msh.geometry, geometry_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']/Geometry"
        )
    return msh
    




def update_msh_1(m, v0, v1):
    x_cord = m.geometry.x[:,0]
    min_x = np.min(x_cord)
    max_x = np.max(x_cord)
    d0 = (min_x-v0)/min_x
    d1 = (max_x+v1)/max_x
    mov = d0 * (x_cord<0).astype(float) + d1 * (x_cord>0).astype(float) 


    m.geometry.x[:,0] = m.geometry.x[:,0]*mov
    

def update_msh_2(m, v0, v1):
    x_cord = m.geometry.x[:,0]
    
    min_x = np.min(x_cord)
    max_x = np.max(x_cord)
    left_s = min_x + 0.5
    right_s = max_x - 0.5
    
    point_l = -1
    point_r = 1
    
    mov = np.zeros(np.shape(x_cord))
    mov[x_cord<=left_s] = -v0
    mov[x_cord>=right_s] = v1
    mask = np.logical_and(x_cord > left_s, x_cord < point_l)
    mov[mask] = v0*(x_cord[mask] - point_l)/(point_l-left_s)
    mask = np.logical_and(x_cord < right_s, x_cord > point_r)
    mov[mask] = v1*(x_cord[mask] - point_r)/(right_s - point_r)

    m.geometry.x[:,0] = m.geometry.x[:,0] + mov
    



    
    
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

    dt = 5e-4          # time step 5.0e-04 
    step_number = 3000      # time step number
    step_ini = 3000
    
    time_range = np.linspace(0, step_number* dt,  step_number+1)
    time_ini = np.linspace(0, step_ini* dt,  step_ini+1)
    
    v_0 = 0.0005        # speed of left pole
    v_1 = 0.0005           # speed of right pole
    norm_stop = 0.1e-6  
    
    #initial conditions
    au = 1
    bu = 1.1
    av = 0.9
    bv = 0.95

    # Parameters for weak statement of the equations

    k = dt
    d = 10
    gamma = 300
    a = 0.1
    b = 0.9

    d1 = 1.0
    d2 = d


    # mesh and linear lagrange elements are created

    msh = create_2d_cell_mesh('cell')
    P1 = element("Lagrange", msh.basix_cell(), 1, gdim=msh.ufl_cell().geometric_dimension())
    ME = functionspace(msh, mixed_element([P1, P1], gdim=msh.ufl_cell().geometric_dimension())) 
    
    
    #extracting stiffness matrix
    ME_test = functionspace(msh, P1)
    q_test = ufl.TestFunction(ME_test)
    u_test = ufl.TrialFunction(ME_test)
    a_mass_mat = form( u_test * q_test * dx)
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
        
        
        # Weak formulation (specific term for mass matrix variation)
        F = u1/k*q*dx + d1*inner(grad(u1), grad(q))*dx\
            -(gamma*(u1*u1*u2-u1+a))*q*dx \
            - (u10/k)*q*dx \
            + u2/k*v*dx + d2*inner(grad(u2), grad(v))*dx\
            -(gamma*(-u1*u1*u2+b))*v*dx \
            - (u20/k)*v*dx


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
        print(f"Initial Step {i}: num iterations: {r[0]}")
        
        if np.linalg.norm(u.x.array-u0.x.array)/dt < norm_stop:
            print("l2_norm convergence")
            break
        




    
    # Output file
    file1 = XDMFFile(MPI.COMM_WORLD, "Schnakenberg2D/outputu1.xdmf", "w")
    file1.write_mesh(msh)
    file2 = XDMFFile(MPI.COMM_WORLD, "Schnakenberg2D/outputu2.xdmf", "w")
    file2.write_mesh(msh)

    
    
    
    
    
    
    
    # reporting values
    u1 = u.sub(0).collapse()
    u2 = u.sub(1).collapse()
    file1.write_function(u1, 0, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']")
    file2.write_function(u2, 0, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']")
    u0.x.array[:] = u.x.array
    

    


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
        
        
        # Weak formulation (specific term for mass matrix variation)
        F = u1/k*q*dx + d1*inner(grad(u1), grad(q))*dx\
            -(gamma*(u1*u1*u2-u1+a))*q*dx \
            - (u10/k)*q*dx \
            + u2/k*v*dx + d2*inner(grad(u2), grad(v))*dx\
            -(gamma*(-u1*u1*u2+b))*v*dx \
            - (u20/k)*v*dx


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
        u1 = u.sub(0).collapse()
        u2 = u.sub(1).collapse()
        
        u0.x.array[:] = u.x.array
        # updating mesh geometry
        update_msh_2(msh, v_0, v_1)

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
    




    
    
    
    
def main():
    run_simulation()
    
    
    
if __name__ == '__main__':
    main()


