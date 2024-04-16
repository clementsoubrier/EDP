
# # Schnakenberg model
# +
import os

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import math

import ufl
from basix.ufl import element, mixed_element
from dolfinx import default_real_type, log, plot, mesh
from dolfinx.fem import Function, functionspace, Expression, FunctionSpaceBase
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import XDMFFile
from dolfinx.mesh import CellType, create_unit_square, GhostMode
from dolfinx.nls.petsc import NewtonSolver
from ufl import dx, grad, inner, SpatialCoordinate

# try:
#     import pyvista as pv
#     import pyvistaqt as pvqt
#     have_pyvista = True
#     if pv.OFF_SCREEN:
#         pv.start_xvfb(wait=0.5)
# except ModuleNotFoundError:
#     print("pyvista and pyvistaqt are required to visualise the solution")
#     have_pyvista = False

# Save all logging to file
log.set_output_file("log.txt")
# -

# Next, various model parameters are defined:

dt = 5.0e-04  # time step

# A unit square mesh with 96 cells edges in each direction is created,
# and on this mesh a
# {py:class}`FunctionSpaceBase <dolfinx.fem.FunctionSpaceBase>` `ME` is built
# using a pair of linear Lagrange elements.

msh = mesh.create_unit_interval(comm=MPI.COMM_WORLD,
                            nx=50,
                            ghost_mode=GhostMode.shared_facet)
P1 = element("Lagrange", msh.basix_cell(), 1)
ME = functionspace(msh, mixed_element([P1, P1]))

x = msh.geometry.x

print("flag 1")
#print(x[:,0])

# Trial and test functions of the space `ME` are now defined:

q, v = ufl.TestFunctions(ME)

# ```{index} split functions

# +
u = Function(ME)  # current solution
u0 = Function(ME)  # solution from previous converged step

# Split mixed functions

u1, u2 = ufl.split(u)
u10, u20 = ufl.split(u0)

au = 0.9
bu = 1.1
av = 0.8
bv = 0.95

# Interpolate initial condition

u.sub(0).interpolate(lambda x: (bu-au)*np.random.rand(x.shape[1]) +au)
u.sub(1).interpolate(lambda x: (bv-av)*np.random.rand(x.shape[1]) +av)

u.x.scatter_forward()

# Weak statement of the equations

k = dt
d = 10
gamma = 250
a = 0.1
b = 0.9

d1 = 1.0
d2 = d

# variable for coordinates

x_coord = SpatialCoordinate(msh)

F = ((u1 - u10) / k)*q*dx + d1*inner(grad(u1), grad(q))*dx\
   -(gamma*(u1*u1*u2-u1+a))*q*dx \
  + ((u2 - u20) / k)*v*dx + d2*inner(grad(u2), grad(v))*dx\
  -(gamma*(-u1*u1*u2+b))*v*dx 


# This is a statement of the time-discrete equations presented as part
# of the problem statement, using UFL syntax.
#
# ```{index} single: Newton solver; (in Cahn-Hilliard demo)
# ```
#
# The DOLFINx Newton solver requires a
# {py:class}`NonlinearProblem<dolfinx.fem.NonlinearProblem>` object to
# solve a system of nonlinear equations

# +
# Create nonlinear problem and Newton solver
problem = NonlinearProblem(F, u)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = np.sqrt(np.finfo(default_real_type).eps) * 1e-2

# We can customize the linear solver used inside the NewtonSolver by
# modifying the PETSc options
ksp = solver.krylov_solver
opts = PETSc.Options()  # type: ignore
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
ksp.setFromOptions()
# -



# The setting of `convergence_criterion` to `"incremental"` specifies
# that the Newton solver should compute a norm of the solution increment
# to check for convergence (the other possibility is to use
# `"residual"`, or to provide a user-defined check). The tolerance for
# convergence is specified by `rtol`.
#
# To run the solver and save the output to a VTK file for later
# visualization, the solver is advanced in time from $t_{n}$ to
# $t_{n+1}$ until a terminal time $T$ is reached:

# +
# Output file
file1 = XDMFFile(MPI.COMM_WORLD, "Schnakenberg1D/outputu1.xdmf", "w")
file2 = XDMFFile(MPI.COMM_WORLD, "Schnakenberg1D/outputu2.xdmf", "w")
file1.write_mesh(msh)
file2.write_mesh(msh)

# Step in time
t = 0.0

#  Reduce run time if on test (CI) server
if "CI" in os.environ.keys() or "GITHUB_ACTIONS" in os.environ.keys():
    T = 3 * dt
else:
    T = 1000* dt

# Get the sub-space for c and the corresponding dofs in the mixed space
# vector
V0, dofs = ME.sub(0).collapse()

# Prepare viewer for plotting the solution during the computation
# if have_pyvista:
#     # Create a VTK 'mesh' with 'nodes' at the function dofs
#     topology, cell_types, x = plot.vtk_mesh(V0)

u1 = u.sub(0)
u2 = u.sub(1)

u0.x.array[:] = u.x.array

t_array = []
l2_norm = []
norm_stop = 0.1e-6

z = 0
inc = 0 
impresion = 100

x_initial = msh.geometry.x[:,0]

u_array = []
xp_array = []
tp_array = []


sigma = 10e-5

V0, dofs = ME.sub(0).collapse()

while (t < T):
    t += dt
    r = solver.solve(u)
    inc += 1

    print(f"Step {int(t/dt)}: num iterations: {r[0]}")

    t_array.append(t)
    l2_norm.append(np.linalg.norm(u.x.array-u0.x.array)/dt)
    
    u0.x.array[:] = u.x.array

    #print (f"u1 array shape {u.x.array[dofs].shape}")

    num=inc/impresion-1 

    r = 1 + sigma*math.sqrt(t)

    msh.geometry.x[:,0] = x_initial*r


    if (int(num)==z) or (int(num-z)==0):
        us, vs = ufl.split(u)

        print(us)
        u_array.append(u.x.array[dofs])
        xp_array.append(msh.geometry.x[:,0])
        tp_array.append(t)

        # Save solution to file (VTK)

        name = "mesh_at_t"+str(t)
        msh.name = name

        file1.write_mesh(msh)
        file1.write_function(u1, t, mesh_xpath=f"/Xdmf/Domain/Grid[@Name='{msh.name}']")

        z += 1
    

    if l2_norm[-1] < norm_stop:
        print("l2_norm convergence")
        break
 
    # Update the plot window
    # if have_pyvista:
        # Update plot
        # p.add_text(f"time: {t:.2e}", font_size=12, name="timelabel")
        # grid.point_data["u"] = u.x.array[dofs].real
        # p.app.processEvents()

file1.close()
file2.close()

tn = np.array(tp_array)
xn = np.array(xp_array)
un = np.array(u_array)

rowsx,columnx = xn.shape
rowsu,columnu = un.shape

print(f"xn shape {rowsx} , {columnx} ")
print(f"un shape {rowsu} , {columnu} ")

# import matplotlib.pyplot as plt

# X, T = np.meshgrid(np.linspace(np.min(xn), np.max(xn), rowsx), tn)

# from scipy.interpolate import griddata

# points = np.array([(tn[i], xn[i, j]) for i in range(rowsx) for j in range(columnu)])
# values = un.flatten()
# U = griddata(points, values, (T, X), method='cubic')

# plt.figure()

# surf = plt.plot_surface(X, T, U, levels=50, cmap='viridis')

# plt.colorbar(surf)
# plt.title('u')
# plt.xlabel('Spatial Coordinate')
# plt.ylabel('Time')
# plt.savefig("plotsurf.png")
