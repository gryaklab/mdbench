"""
Code adapted from the FEniCSx tutorial: https://jsdokken.com/dolfinx-tutorial/index.html
"""

from mpi4py import MPI
from dolfinx import fem, mesh, io
from dolfinx.fem.petsc import assemble_vector, assemble_matrix, create_vector, apply_lifting, set_bc
from petsc4py import PETSc
import ufl
import numpy as np
import meshio

import os
import json
from copy import deepcopy
from typing import Dict

P_HEAT_FEM ={
    'name': 'diffusion_2d',
    'dim': 2,
    'equation': "u_t = nabla^2u + f",
    'start_time': 0,
    'end_time': 1,
    'num_steps': 50,  #Number of time steps
    #mesh parameters
    'Nx': 50,
    'Ny': 50,
    #grid parameters - lower left point and upper right corners
    'Gl': [-2,-2],
    'Gr': [2,2],
    #initial condition scale
    'ic_scale': 5
}

P_HEAT_EXACT ={
    'start_time': 0,
    'end_time': 2,
    'num_steps': 20,  #Number of time steps
    #parameters for exact solution
    'alpha':3,
    'beta': 1.2,
    #mesh parameters
    'Nx': 5,
    'Ny': 5
}

def process_data(params,solution_name="uh",xdmf_file=None):
    """
    This reads in an XDMF file, which is used to store mesh-related data, and transforms it
    to something similar to the ODE output format.

    It is easier to do this after solving the PDE as FEniCSx uses MPI, which partitions the mesh
    and attendant solutions across CPUs that have to be re-joined later.
    """
    #copy parameters used to solve PDE
    dataset = deepcopy(params)
    
    #set filename if not specified
    if xdmf_file is None:
        xdmf_file = params['name']+".xdmf"

    #read in file (note that both the H5 and XDMF files must have the same name and be in the same path)
    reader = meshio.xdmf.TimeSeriesReader(xdmf_file)
    #points: numpy array of coordinates
    #cell_list: list of cell blocks
    points, cell_list = reader.read_points_cells()
    #cell blocks - corresponding to cells used in finite element mesh
    #cells = cell_list[0]
    
    #extract spatiotemporal components and solution values    
    evaluations, spatial_dims = points.shape
    timesteps = reader.num_steps
    t = np.zeros(timesteps)
    u=np.zeros((evaluations,timesteps))

    for k in range(reader.num_steps):
        t[k], point_data, cell_data = reader.read_data(k)
        u[:,k]=np.squeeze(point_data[solution_name])

    #time coordinates
    dataset['t'] = t.tolist()
    #spatial coordinates
    dataset['X'] = points.tolist()
    #solution values
    dataset['u'] = u.tolist()

    #save to JSON
    prefix, ext = os.path.splitext(xdmf_file)
    json.dump(dataset, open(prefix+".json", "w"), indent=1)

def heat_equation_fem(params: Dict | None = None):
    """Solves the heat equation on a rectangular grid with a Gaussian initial
    condition and zero boundary condition.

    The parameters dictionary can be used to modify the spatiotemporal resolution of the problem.
    """
    if params is None:
        params = P_HEAT_FEM

    #define timestep
    curr_t=params['start_time']
    dt = (params['end_time'] - params['start_time']) / params['num_steps']  

    # Define mesh
    domain = mesh.create_rectangle(MPI.COMM_WORLD, [np.array(params['Gl']), np.array(params['Gr'])],
                                [params['Nx'], params['Ny']], mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 1))

    # Create initial condition
    def initial_condition(x, a=params['ic_scale']):
        return np.exp(-a * (x[0]**2 + x[1]**2))
    
    u_n = fem.Function(V)
    u_n.name = "u_n"
    u_n.interpolate(initial_condition)

    #create output file, which can collect data across multiple processes, and save the mesh to it
    xdmf = io.XDMFFile(domain.comm, params['name']+".xdmf", "w")
    xdmf.write_mesh(domain)

    # Define solution variable
    uh = fem.Function(V)
    uh.name = "uh"
    #interpolate initial solution (for plotting only)
    uh.interpolate(initial_condition)
    xdmf.write_function(uh, curr_t)

    # Create boundary condition
    fdim = domain.topology.dim - 1
    boundary_facets = mesh.locate_entities_boundary(
        domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    bc = fem.dirichletbc(PETSc.ScalarType(0), fem.locate_dofs_topological(V, fdim, boundary_facets), V)

    #Setup variational problem
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    f = fem.Constant(domain, PETSc.ScalarType(0))
    a = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = (u_n + dt * f) * v * ufl.dx

    #prepare time dependent structures (matrix and vector)
    bilinear_form = fem.form(a)
    linear_form = fem.form(L)

    #A remains constant across time steps, b does not
    A = assemble_matrix(bilinear_form, bcs=[bc])
    A.assemble()
    b = create_vector(linear_form)

    #Setup solver. Can't use LinearProblem as with Poisson because we already constructed the bilinear form A
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)

    #solve PDE
    for i in range(params['num_steps']):
        curr_t += dt

        # Update the right hand side reusing the initial vector
        with b.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b, linear_form)

        # Apply Dirichlet boundary condition to the vector
        apply_lifting(b, [bilinear_form], [[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, [bc])

        # Solve linear problem
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()

        # Update solution at previous time step (u_n)
        u_n.x.array[:] = uh.x.array

        # Write solution to file
        xdmf.write_function(uh, curr_t)

    xdmf.close()

class heat_exact_solution():
    """Implements the exact solution use in heat_equation_exact()."""
    def __init__(self, alpha, beta, t):
        self.alpha = alpha
        self.beta = beta
        self.t = t

    def __call__(self, x):
        return 1 + x[0]**2 + self.alpha * x[1]**2 + self.beta * self.t
    
def heat_equation_exact(params: Dict | None = None):
    """Validates the variational form of the heat equation.

    Unlike in the ODE case, we won't have an "exact solution" to compare to:
    the FEM solution will be the ground truth. This is used to verify that
    the variational form of the heat equation is correct by comparing the 
    FEM solution to an exact solution of a simple, known function, namely
    1 + x^2 + alpha * y^2 + beta * t.
    """
    if params is None:
        params = P_HEAT_EXACT

    #define timestep
    dt = (params['end_time'] - params['start_time']) / params['num_steps']  

    #define mesh    
    domain = mesh.create_unit_square(MPI.COMM_WORLD, params['Nx'], params['Ny'], mesh.CellType.triangle)
    V = fem.functionspace(domain, ("Lagrange", 1))

    #define exact solution
    u_exact = heat_exact_solution(params['alpha'], params['beta'], params['start_time'])
    u_D = fem.Function(V)

    #define boundary condition
    u_D.interpolate(u_exact)
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets))

    #define variational form
    u_n = fem.Function(V)
    u_n.interpolate(u_exact)
    #function is time-independent, i.e., constant over t
    f = fem.Constant(domain, params['beta'] - 2 - 2 * params['alpha'])

    #setup variational problem
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    F = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx - (u_n + dt * f) * v * ufl.dx
    a = fem.form(ufl.lhs(F))
    L = fem.form(ufl.rhs(F))

    #A remains constant across time steps, b does not
    A = assemble_matrix(a, bcs=[bc])
    A.assemble()
    b = create_vector(L)
    uh = fem.Function(V)

    #Configure solver
    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)

    for n in range(params['num_steps']):
        #Update Dirichlet boundary condition
        u_exact.t += dt
        u_D.interpolate(u_exact)

        #Update the right hand side reusing the initial vector
        with b.localForm() as loc_b:
            loc_b.set(0)
        assemble_vector(b, L)

        #Apply Dirichlet boundary condition to the vector
        apply_lifting(b, [a], [[bc]])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, [bc])

        #Solve linear problem
        solver.solve(b, uh.x.petsc_vec)
        uh.x.scatter_forward()

        #Update solution at previous time step (u_n)
        u_n.x.array[:] = uh.x.array

    #Compute L2 error and error at nodes
    V_ex = fem.functionspace(domain, ("Lagrange", 2))
    u_ex = fem.Function(V_ex)
    u_ex.interpolate(u_exact)
    error_L2 = np.sqrt(domain.comm.allreduce(fem.assemble_scalar(fem.form((uh - u_ex)**2 * ufl.dx)), op=MPI.SUM))
    if domain.comm.rank == 0:
        #Expected output: L2-error: 2.83e-02
        print(f"L2-error: {error_L2:.2e}")

    #Compute values at mesh vertices
    error_max = domain.comm.allreduce(np.max(np.abs(uh.x.array - u_D.x.array)), op=MPI.MAX)
    if domain.comm.rank == 0:
        #Expected output: Error_max: 2.66e-15
        print(f"Error_max: {error_max:.2e}")

if __name__ == '__main__':
     heat_equation_fem()
     #heat_equation_exact()
     process_data(P_HEAT_FEM)