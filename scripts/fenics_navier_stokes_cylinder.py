"""
Code adapted from original source:
[FEniCS '16](https://doi.org/10.1007/978-3-319-52462-7) *Solving PDEs in Python: The FEniCS Tutorial I* by Langtangen and Logg
"""

from fenics import *
from mshr import *
from mpi4py import MPI

P_NS_CYLINDER ={
    'name': 'navier_stokes_cylinder',
    'dim': 2,
    'equation': [r"rho(u_t + u dot nabla(u)) - div(sigma(u, p)) = f",
                 r"div(u) = 0"],
    'start_time': 0.0,
    'end_time': 5.0,
    'num_steps': 5000,
    #grid parameters - lower left point and upper right corners
    'Gl': [0.0, 0.0],
    'Gr': [2.2, 0.41],
    #cylinder parameters
    'cylinder_center':[0.2,0.2],
    'cylinder_radius': 0.05,
    #mesh parameters
    'num_cells': 80,
    #system parameters
    'mu': 0.001,        # dynamic viscosity
    'rho': 1.0          # density
}

def navier_stokes_cylinder(params=None):
    """
    FEniCS tutorial demo program: Incompressible Navier-Stokes equations
    for flow around a cylinder using the Incremental Pressure Correction
    Scheme (IPCS).

    u' + u . nabla(u)) - div(sigma(u, p)) = f
                                    div(u) = 0
    """
    #get process number if running in parallel
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if params is None:
        params = P_NS_CYLINDER

    # Create mesh
    channel = Rectangle(Point(*params['Gl']), Point(*params['Gr']))
    cylinder = Circle(Point(*params['cylinder_center']), params['cylinder_radius'])
    domain = channel - cylinder
    mesh = generate_mesh(domain, params['num_cells'])

    # Define function spaces
    V = VectorFunctionSpace(mesh, 'P', 2)
    Q = FunctionSpace(mesh, 'P', 1)

    # Define boundaries
    inflow   = 'near(x[0], 0)'
    outflow  = 'near(x[0], 2.2)'
    walls    = 'near(x[1], 0) || near(x[1], 0.41)'
    cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'

    # Define inflow profile
    inflow_profile = ('4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0')

    # Define boundary conditions
    bcu_inflow = DirichletBC(V, Expression(inflow_profile, degree=2), inflow)
    bcu_walls = DirichletBC(V, Constant((0, 0)), walls)
    bcu_cylinder = DirichletBC(V, Constant((0, 0)), cylinder)
    bcp_outflow = DirichletBC(Q, Constant(0), outflow)
    bcu = [bcu_inflow, bcu_walls, bcu_cylinder]
    bcp = [bcp_outflow]

    # Define trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)
    p = TrialFunction(Q)
    q = TestFunction(Q)

    # Define functions for solutions at previous and current time steps
    u_n = Function(V)
    uh  = Function(V, name='uh')
    p_n = Function(Q)
    ph  = Function(Q, name='ph')

    # Define expressions used in variational forms
    dt=(params['end_time']-params['start_time'])/params['num_steps']
    U  = 0.5*(u_n + u)
    n  = FacetNormal(mesh)
    f  = Constant((0, 0))
    k  = Constant(dt)
    mu = Constant(params['mu'])
    rho = Constant(params['rho'])

    # Define symmetric gradient
    def epsilon(u):
        return sym(nabla_grad(u))

    # Define stress tensor
    def sigma(u, p):
        return 2*mu*epsilon(u) - p*Identity(len(u))

    # Define variational problem for step 1
    F1 = rho*dot((u - u_n) / k, v)*dx \
    + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
    + inner(sigma(U, p_n), epsilon(v))*dx \
    + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
    - dot(f, v)*dx
    a1 = lhs(F1)
    L1 = rhs(F1)

    # Define variational problem for step 2
    a2 = dot(nabla_grad(p), nabla_grad(q))*dx
    L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(uh)*q*dx

    # Define variational problem for step 3
    a3 = dot(u, v)*dx
    L3 = dot(uh, v)*dx - k*dot(nabla_grad(ph - p_n), v)*dx

    # Assemble matrices
    A1 = assemble(a1)
    A2 = assemble(a2)
    A3 = assemble(a3)

    # Apply boundary conditions to matrices
    [bc.apply(A1) for bc in bcu]
    [bc.apply(A2) for bc in bcp]

    #create output files, which can collect data across multiple processes
    #velocity file
    xdmf_file_u = XDMFFile(params['name']+"_u.xdmf")
    xdmf_file_u.parameters["flush_output"] = True
    xdmf_file_u.parameters["functions_share_mesh"] = True
    xdmf_file_u.parameters["rewrite_function_mesh"] = False
    #pressure file
    xdmf_file_p = XDMFFile(params['name']+"_p.xdmf")
    xdmf_file_p.parameters["flush_output"] = True
    xdmf_file_p.parameters["functions_share_mesh"] = True
    xdmf_file_p.parameters["rewrite_function_mesh"] = False

    # Create time series (for use in reaction_system.py)
    timeseries_u = TimeSeries('navier_stokes_cylinder_velocity_series')
    timeseries_p = TimeSeries('navier_stokes_cylinder_pressure_series')

    # Save mesh to file (for use in reaction_system.py), using only the first process
    if rank == 0:
        File('navier_stokes_cylinder.xml.gz') << mesh

    # Time-stepping
    t = 0
    for n in range(params['num_steps']):

        # Update current time
        t += dt

        # Step 1: Tentative velocity step
        b1 = assemble(L1)
        [bc.apply(b1) for bc in bcu]
        solve(A1, uh.vector(), b1, 'bicgstab', 'hypre_amg')

        # Step 2: Pressure correction step
        b2 = assemble(L2)
        [bc.apply(b2) for bc in bcp]
        solve(A2, ph.vector(), b2, 'bicgstab', 'hypre_amg')

        # Step 3: Velocity correction step
        b3 = assemble(L3)
        solve(A3, uh.vector(), b3, 'cg', 'sor')

        # Save solution to file (XDMF/HDF5)
        xdmf_file_u.write(uh, t)
        xdmf_file_p.write(ph, t)

        # Save nodal values to file
        timeseries_u.store(uh.vector(), t)
        timeseries_p.store(ph.vector(), t)

        # Update previous solution
        u_n.assign(uh)
        p_n.assign(ph)
    
    #close file
    xdmf_file_u.close()
    xdmf_file_p.close()

if __name__ == '__main__':
    navier_stokes_cylinder()
