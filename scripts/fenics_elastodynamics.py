from dolfin import *
import numpy as np
from typing import Dict

P_ED ={
    'name': 'elastodynamics_3d',
    'dim': 3,
    'equation': ["nabla dot sigma + rho*b = rho *u_tt",
                 "sigma= lambda * trace(eps)I_2 +2*mu*eps",
                 "eps=(nabla u + (nabla u)^T)/2"],
    'start_time': 0.0,
    'end_time': 10.0,
    'num_steps': 125,  #Number of time steps
    #mesh parameters
    'Nx': 60,
    'Ny': 10,
    'Nz': 5,
    #grid parameters - lower left point and upper right corners
    'Gl': [0.0, 0.0, 0.0],
    'Gr': [1., 0.1, 0.04],
    #system parameters
    # Elastic parameters
    'E':  1000.0,
    'nu': 0.3,
    # Mass density
    'rho': Constant(1.0),
    # Rayleigh damping coefficients
    'eta_m': Constant(0.),
    'eta_k':Constant(0.),
    #imposed loading
    'p0': 1
}

P_ED_DAMP0_1 ={
    'name': 'elastodynamics_3d_damp_0_1',
    'dim': 3,
    'equation': ["nabla dot sigma + rho*b = rho *u_tt",
                 "sigma= lambda * trace(eps)I_2 +2*mu*eps",
                 "eps=(nabla u + (nabla u)^T)/2"],
    'start_time': 0.0,
    'end_time': 10.0,
    'num_steps': 125,  #Number of time steps
    #mesh parameters
    'Nx': 60,
    'Ny': 10,
    'Nz': 5,
    #grid parameters - lower left point and upper right corners
    'Gl': [0.0, 0.0, 0.0],
    'Gr': [1., 0.1, 0.04],
    #system parameters
    # Elastic parameters
    'E':  1000.0,
    'nu': 0.3,
    # Mass density
    'rho': Constant(1.0),
    # Rayleigh damping coefficients
    'eta_m': Constant(0.1),
    'eta_k':Constant(0.1),
    #imposed loading
    'p0': 1
}
def elastodynamics(params: Dict | None = None):
    """
    This computes the stress tensor of transient elastodynamics problem related to deformation in an elastic structure, 
    using the generalized-alpha method [ERL2002].

    Two sets of parameters are defined - one with damping and one without.

    Code adapted from: https://comet-fenics.readthedocs.io/en/latest/demo/elastodynamics/demo_elastodynamics.py.html
    Refs:
    [ERL2002] Silvano Erlicher, Luca Bonaventura, Oreste Bursi. The analysis of the Generalized-alpha method for non-linear dynamic problems. Computational Mechanics, Springer Verlag, 2002, 28, pp.83-104, doi:10.1007/s00466-001-0273-z
    """
    if params is None:
        params = P_ED
       
    #dolphin parameters
    parameters["form_compiler"]["cpp_optimize"] = True
    parameters["form_compiler"]["optimize"] = True

    # Define mesh
    mesh = BoxMesh(Point(*params['Gl']), Point(*params['Gr']), params['Nx'], params['Ny'], params['Nz'])

    # Sub domain for clamp at left end
    def left(x, on_boundary):
        return near(x[0], 0.) and on_boundary

    # Sub domain for rotation at right end
    def right(x, on_boundary):
        return near(x[0], 1.) and on_boundary

    #Elastic parameters
    mu    = Constant(params['E'] / (2.0*(1.0 + params['nu'])))
    lmbda = Constant(params['E']*params['nu'] / ((1.0 + params['nu'])*(1.0 - 2.0*params['nu'])))

    # Parameters used for the time discretization scheme are now defined. First, the four parameters used by the
    # generalized-:math:`\alpha` method are chosen. Here, we used the optimal dissipation and second-order accuracy
    # choice for :math:`\beta` and :math:`\gamma`, namely :math:`\beta=\dfrac{1}{4}\left(\gamma+\dfrac{1}{2}\right)^2` and
    # :math:`\gamma=\dfrac{1}{2}+\alpha_m-\alpha_f` with :math:`\alpha_m=0.2` and :math:`\alpha_f=0.4` ensuring unconditional stability.

    # Generalized-alpha method parameters
    alpha_m = Constant(0.2)
    alpha_f = Constant(0.4)
    gamma   = Constant(0.5+alpha_f-alpha_m)
    beta    = Constant((gamma+0.5)**2/4.)

    # We also define the final time of the interval, the number of time steps and compute the associated time interval
    # between two steps::

    #compute delta t
    dt = Constant((params['end_time']-params['start_time'])/params['num_steps'])

    # We now define the time-dependent loading. Body forces are zero and the imposed loading consists of a uniform vertical traction
    # applied at the ``right`` extremity. The loading amplitude will vary linearly from 0 to p_0=1 over the time interval
    # [0;T_c=T*0.2], after T_c the loading is removed. 
    cutoff_Tc = (params['end_time']-params['start_time'])*.2
    
    # Define the loading as an expression depending on t. In particular, it uses a conditional syntax using operators ? and :
    p = Expression(("0", "t <= tc ? p0*t/tc : 0", "0"), t=params['start_time'], tc=cutoff_Tc, p0=params['p0'], degree=0)

    # Define function space for displacement, velocity and acceleration
    V = VectorFunctionSpace(mesh, "CG", 1)
    # Define function space for stresses
    Vsig = TensorFunctionSpace(mesh, "DG", 0)

    # Test and trial functions
    du = TrialFunction(V)
    u_ = TestFunction(V)
    # Current (unknown) displacement
    u = Function(V, name="Displacement")
    # Fields from previous time step (displacement, velocity, acceleration)
    u_old = Function(V)
    v_old = Function(V)
    a_old = Function(V)

    # Create mesh function over the cell facets
    boundary_subdomains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_subdomains.set_all(0)
    force_boundary = AutoSubDomain(right)
    force_boundary.mark(boundary_subdomains, 3)

    # Define measure for boundary condition integral
    dss = ds(subdomain_data=boundary_subdomains)

    # Set up boundary condition at left end
    zero = Constant((0.0, 0.0, 0.0))
    bc = DirichletBC(V, zero, left)

    # Python functions are now defined to obtain the elastic stress tensor \sigma (linear isotropic elasticity), 
    # the bilinear mass and stiffness forms as well as the damping form obtained as a linear combination of the mass and 
    # stiffness forms (Rayleigh damping). The linear form corresponding to the work of external forces is also defined.

    # Stress tensor
    def sigma(r):
        return 2.0*mu*sym(grad(r)) + lmbda*tr(sym(grad(r)))*Identity(len(r))

    # Mass form
    def m(u, u_):
        return params['rho']*inner(u, u_)*dx

    # Elastic stiffness form
    def k(u, u_):
        return inner(sigma(u), sym(grad(u_)))*dx

    # Rayleigh damping form
    def c(u, u_):
        return params['eta_m']*m(u, u_) + params['eta_k']*k(u, u_)

    # Work of external forces
    def Wext(u_):
        return dot(u_, p)*dss(3)

    # Functions for implementing the time stepping scheme are define
    # The keyword ``ufl`` enables UFL representations by using ``Constant`` types (ufl=True) or floats otherwise.

    # Update formula for acceleration
    # a = 1/(2*beta)*((u - u0 - v0*dt)/(0.5*dt*dt) - (1-2*beta)*a0)
    def update_a(u, u_old, v_old, a_old, ufl=True):
        if ufl:
            dt_ = dt
            beta_ = beta
        else:
            dt_ = float(dt)
            beta_ = float(beta)
        return (u-u_old-dt_*v_old)/beta_/dt_**2 - (1-2*beta_)/2/beta_*a_old

    # Update formula for velocity
    # v = dt * ((1-gamma)*a0 + gamma*a) + v0
    def update_v(a, u_old, v_old, a_old, ufl=True):
        if ufl:
            dt_ = dt
            gamma_ = gamma
        else:
            dt_ = float(dt)
            gamma_ = float(gamma)
        return v_old + dt_*((1-gamma_)*a_old + gamma_*a)

    def update_fields(u, u_old, v_old, a_old):
        """Update fields at the end of each time step."""

        # Get vectors (references)
        u_vec, u0_vec  = u.vector(), u_old.vector()
        v0_vec, a0_vec = v_old.vector(), a_old.vector()

        # use update functions using vector arguments
        a_vec = update_a(u_vec, u0_vec, v0_vec, a0_vec, ufl=False)
        v_vec = update_v(a_vec, u0_vec, v0_vec, a0_vec, ufl=False)

        # Update (u_old <- u)
        v_old.vector()[:], a_old.vector()[:] = v_vec, a_vec
        u_old.vector()[:] = u.vector()

    # The system variational form is now built by expressing the new acceleration {u_tt}_{n+1} as a function of
    # the TrialFunction du using ``update_a``, which here works as a UFL expression. Using this new acceleration, the same is
    # done for the new velocity using ``update_v``. Intermediate averages using parameters \alpha_m,\alpha_f of the generalized-\alpha
    # method are obtained with a user-defined function ``avg``. The weak form evolution equation is then written using all these
    # quantities. Since the problem is linear, we then extract the bilinear and linear parts using ``rhs`` and ``lhs``.

    def avg(x_old, x_new, alpha):
        return alpha*x_old + (1-alpha)*x_new

    # Residual
    a_new = update_a(du, u_old, v_old, a_old, ufl=True)
    v_new = update_v(a_new, u_old, v_old, a_old, ufl=True)
    res = m(avg(a_old, a_new, alpha_m), u_) + c(avg(v_old, v_new, alpha_f), u_) \
        + k(avg(u_old, du, alpha_f), u_) - Wext(u_)
    a_form = lhs(res)
    L_form = rhs(res)


    # Since the system matrix to solve is the same for each time step (constant time step), it is not necessary to factorize the system at each increment.
    # Define solver for reusing factorization
    K, res = assemble_system(a_form, L_form, bc)
    solver = LUSolver(K, "mumps")
    solver.parameters["symmetric"] = True

    # Time-stepping
    time = np.linspace(params['start_time'], params['end_time'], params['num_steps']+1)
    # Can only record tip if not using MPI - might not be domain
    #u_tip = np.zeros((params['num_steps']+1,))
    energies = np.zeros((params['num_steps']+1, 4))
    E_damp = 0
    sig = Function(Vsig, name="sigma")

    xdmf_file = XDMFFile(params['name']+".xdmf")
    xdmf_file.parameters["flush_output"] = True
    xdmf_file.parameters["functions_share_mesh"] = True
    xdmf_file.parameters["rewrite_function_mesh"] = False


    def local_project(v, V, u=None):
        """Element-wise projection using LocalSolver"""
        dv = TrialFunction(V)
        v_ = TestFunction(V)
        a_proj = inner(dv, v_)*dx
        b_proj = inner(v, v_)*dx
        solver = LocalSolver(a_proj, b_proj)
        solver.factorize()
        if u is None:
            u = Function(V)
            solver.solve_local_rhs(u)
            return u
        else:
            solver.solve_local_rhs(u)
            return

    #solve PDE
    for (i, dt) in enumerate(np.diff(time)):
        t = time[i+1]

        # Forces are evaluated at t_{n+1-alpha_f}=t_{n+1}-alpha_f*dt
        p.t = t-float(alpha_f*dt)

        # Solve for new displacement
        res = assemble(L_form)
        bc.apply(res)
        solver.solve(K, u.vector(), res)

        # Update old fields with new quantities
        update_fields(u, u_old, v_old, a_old)

        # Save solution to XDMF format
        xdmf_file.write(u, t)

        # Compute stresses and save to file
        local_project(sigma(u), Vsig, sig)
        xdmf_file.write(sig, t)

        p.t = t
        # Record tip displacement and compute energies
        # Can only record tip if not using MPI - might not be domain
        #u_tip[i+1] = u(1., 0.05, 0.)[1]
        E_elas = assemble(0.5*k(u_old, u_old))
        E_kin = assemble(0.5*m(v_old, v_old))
        E_damp += dt*assemble(c(v_old, v_old))
        E_tot = E_elas+E_kin+E_damp #-E_ext
        energies[i+1, :] = np.array([E_elas, E_kin, E_damp, E_tot])

if __name__ == '__main__':
    #elastodynamics()
    elastodynamics(P_ED_DAMP0_1)