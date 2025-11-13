"""
Code adapted from original source:
[FEniCS '16](https://doi.org/10.1007/978-3-319-52462-7) *Solving PDEs in Python: The FEniCS Tutorial I* by Langtangen and Logg
"""

from fenics import *
from copy import deepcopy

P_NS_CHANNEL_1 ={
    'name': 'navier_stokes_channel_1',
    'dim': 2,
    'equation': [r"rho(u_t + u dot nabla(u)) - div(sigma(u, p)) = f",
                 r"div(u) = 0"],
    'start_time': 0.0,
    'end_time': 10.0,
    'num_steps': 500,
    #mesh parameters
    'square_length':16,
    #system parameters - note that if the ratio of rho/mu exceeds 40, this will yield turbulent flow
    'mu': 1.0,          # kinematic viscosity
    'rho': 1.0          # density
}

P_NS_CHANNEL_2= deepcopy(P_NS_CHANNEL_1)
P_NS_CHANNEL_2['name']='navier_stokes_channel_2'
P_NS_CHANNEL_2['mu']=.5
P_NS_CHANNEL_2['rho']=10

def navier_stokes_channel(params=None):
    """
    FEniCS tutorial demo program: Incompressible Navier-Stokes equations
    for channel flow (Poisseuille) on the unit square using the
    Incremental Pressure Correction Scheme (IPCS).

    u' + u . nabla(u)) - div(sigma(u, p)) = f
                                    div(u) = 0
    """

    if params is None:
        params = P_NS_CHANNEL_1

    # Create mesh and define function spaces
    mesh = UnitSquareMesh(params['square_length'], params['square_length'])
    V = VectorFunctionSpace(mesh, 'P', 2)
    Q = FunctionSpace(mesh, 'P', 1)
    
    # Define boundaries
    inflow  = 'near(x[0], 0)'
    outflow = 'near(x[0], 1)'
    walls   = 'near(x[1], 0) || near(x[1], 1)'

    # Define boundary conditions
    bcu_noslip  = DirichletBC(V, Constant((0, 0)), walls)
    bcp_inflow  = DirichletBC(Q, Constant(8), inflow)
    bcp_outflow = DirichletBC(Q, Constant(0), outflow)
    bcu = [bcu_noslip]
    bcp = [bcp_inflow, bcp_outflow]

    # Define trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)
    p = TrialFunction(Q)
    q = TestFunction(Q)

    # Define functions for solutions at previous and current time steps
    u_n = Function(V)
    uh = Function(V, name='uh')
    p_n = Function(Q)
    ph = Function(Q, name='ph')
    
    # Define expressions used in variational forms
    # time step size
    dt=(params['end_time']-params['start_time'])/params['num_steps']
    U   = 0.5*(u_n + u)
    n   = FacetNormal(mesh)
    f   = Constant((0, 0))
    k   = Constant(dt)
    mu  = Constant(params['mu'])
    rho = Constant(params['rho'])

    # Define strain-rate tensor
    def epsilon(u):
        return sym(nabla_grad(u))

    # Define stress tensor
    def sigma(u, p):
        return 2*mu*epsilon(u) - p*Identity(len(u))

    # Define variational problem for step 1
    F1 = rho*dot((u - u_n) / k, v)*dx + \
        rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
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

    # Time-stepping
    t = 0

    for n in range(params['num_steps']):

        # Update current time
        t += dt

        # Step 1: Tentative velocity step
        b1 = assemble(L1)
        [bc.apply(b1) for bc in bcu]
        solve(A1, uh.vector(), b1)

        # Step 2: Pressure correction step
        b2 = assemble(L2)
        [bc.apply(b2) for bc in bcp]
        solve(A2, ph.vector(), b2)

        # Step 3: Velocity correction step
        b3 = assemble(L3)
        solve(A3, uh.vector(), b3)

        #write to file
        xdmf_file_u.write(uh,t)
        xdmf_file_p.write(ph,t)

        # Update previous solution
        u_n.assign(uh)
        p_n.assign(ph)
    
    #close file
    xdmf_file_u.close()
    xdmf_file_p.close()

if __name__ == '__main__':
    navier_stokes_channel(P_NS_CHANNEL_1)
    navier_stokes_channel(P_NS_CHANNEL_2)
