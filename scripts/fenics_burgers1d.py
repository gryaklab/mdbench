from dolfin import *

P_BURGERS_1D ={
    'name': 'burgers-1d',
    'dim': 1,
    'equation': [r"u_t = \nu*u_xx + u*u_x"],
    'start_time': 0.0,
    'end_time': 1.0,
    'num_steps': 100,
    #mesh parameters
    'Nx': 32,
    #grid parameters - left and right bounds
    'Gl': [-1.0,1.0],
    #system parameters
    'nu': 0.05  #viscosity
}

def burgers_time_viscous(params=None):
    """
    FENICS code that solves the time-dependent 1D viscous Burgers equation over the interval [-1,+1].
    Author: John Burkardt
    Source: https://people.sc.fsu.edu/~jburkardt/fenics_src/burgers_time_viscous/burgers_time_viscous.html
    """

    if params is None:
      params = P_BURGERS_1D

    # Create mesh and define function spaces  x_left = -1.0
    mesh = IntervalMesh(params['Nx'], *params['Gl'])
    V = FunctionSpace(mesh, "CG", 1)

    #define boundary conditions
    #left boundary
    def on_left(x, on_boundary):
      return (on_boundary and near (x[0], params['Gl'][0]) )
    bc_left = DirichletBC(V, params['Gl'][0], on_left)

    #right boundary
    def on_right (x, on_boundary):
      return (on_boundary and near (x[0], params['Gl'][1]) )
    bc_right = DirichletBC(V, params['Gl'][1], on_right)
    bc = [bc_left, bc_right]
    
    #Define the trial functions (u) and test functions (v).
    uh = Function(V, name="uh")
    u_n = Function(V)
    v = TestFunction(V)

    #Define the initial condition.
    u_init = Expression("x[0]", degree = 1)
    uh.interpolate(u_init)
    u_n.assign(uh)

    # Define expressions used in variational forms
    dt=(params['end_time']-params['start_time'])/params['num_steps']
    DT = Constant(dt)
    f = Expression("0.0", degree = 0)

    # Define variational problem
    F = (dot(uh - u_n, v) / DT \
         + params['nu'] * inner(grad(uh), grad(v)) \
          + inner(uh * uh.dx(0), v) \
            - dot(f, v)) * dx
    #  Specify the Jacobian
    J = derivative(F, uh)

    #create output files, which can collect data across multiple processes
    xdmf_file = XDMFFile(params['name']+".xdmf")
    xdmf_file.parameters["flush_output"] = True
    xdmf_file.parameters["functions_share_mesh"] = True
    xdmf_file.parameters["rewrite_function_mesh"] = False 

    # Time-stepping
    t = 0
    for _ in range(params['num_steps']):
      # Update current time
      t += dt
      
      # solve system
      solve(F == 0, uh, bc, J = J)
      
      #write to file
      xdmf_file.write(uh,t)
      
      u_n.assign(uh)

    #close file
    xdmf_file.close()

if __name__ == '__main__':
    burgers_time_viscous()
