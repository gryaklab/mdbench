"""
Code adapted from original source:
[FEniCS '16](https://doi.org/10.1007/978-3-319-52462-7) *Solving PDEs in Python: The FEniCS Tutorial I* by Langtangen and Logg
"""

from fenics import *

P_DR_CYLINDER ={
    'name': 'diffusion_reaction_cylinder',
    'dim': 2,
    'equation': [r"w_t + w dot nabla(w)) - div(sigma(w, p)) = f",
                 r"div(w) = 0",
                 r"u_1 +  w dot nabla(u_1) - div(eps*grad(u_1)) = f_1 - K*u_1*u_2",
                 r"u_2' + w dot nabla(u_2) - div(eps*grad(u_2)) = f_2 - K*u_1*u_2",
                 r"u_3' + w dot nabla(u_3) - div(eps*grad(u_3)) = f_3 + K*u_1*u_2 - K*u_3"],
    'start_time': 0.0,
    'end_time': 5.0,
    'num_steps': 500,
    #grid, cylinder, and mesh parameters determined by fenics_navier_stokes_cylinder.py
    #path to mesh file
    'mesh_file': r"navier_stokes_cylinder.xml.gz",
    #path to NS cylinder velocity timeseries - DO NOT INCLUDE the extension (h5) - it is added automatically
    'velocity_file': r"navier_stokes_cylinder_velocity_series",
    #path to store concentration data - DO NOT INCLUDE the extension (h5) - it is added automatically
    'concentration_file': r"diffusion_reaction_cylinder_concentration_series",
    #system parameters
    'eps': 0.01,    # diffusion coefficient
    'K': 10.0       # reaction rate
}

def diffusion_reaction_cylinder(params=None):
  """
  FEniCS tutorial demo program: Convection-diffusion-reaction for a system
  describing the concentration of three species A, B, C undergoing a simple
  first-order reaction A + B --> C with first-order decay of C. The velocity
  is given by the flow field w from the demo navier_stokes_cylinder.py.

    u_1' + w . nabla(u_1) - div(eps*grad(u_1)) = f_1 - K*u_1*u_2
    u_2' + w . nabla(u_2) - div(eps*grad(u_2)) = f_2 - K*u_1*u_2
    u_3' + w . nabla(u_3) - div(eps*grad(u_3)) = f_3 + K*u_1*u_2 - K*u_3

  """
  if params is None:
     params = P_DR_CYLINDER
  
  # Read mesh from file
  mesh = Mesh(params['mesh_file'])

  # Define function space for velocity - must match that specified in fenics_navier_stokes_cylinder.py.
  W = VectorFunctionSpace(mesh, 'P', 2)

  # Define function space for system of concentrations
  P1 = FiniteElement('P', triangle, 1)
  element = MixedElement([P1, P1, P1])
  V = FunctionSpace(mesh, element)

  # Define test functions
  v_1, v_2, v_3 = TestFunctions(V)

  # Define functions for velocity and concentrations
  w = Function(W)
  uh = Function(V, name="uh")
  u_n = Function(V)

  # Split system functions to access components
  u_1, u_2, u_3 = split(uh)
  u_n1, u_n2, u_n3 = split(u_n)

  # Define source terms
  f_1 = Expression('pow(x[0]-0.1,2)+pow(x[1]-0.1,2)<0.05*0.05 ? 0.1 : 0',
                  degree=1)
  f_2 = Expression('pow(x[0]-0.1,2)+pow(x[1]-0.3,2)<0.05*0.05 ? 0.1 : 0',
                  degree=1)
  f_3 = Constant(0)

  # Define expressions used in variational forms
  dt=(params['end_time']-params['start_time'])/params['num_steps']
  k = Constant(dt)
  K = Constant(params['K'])
  eps = Constant(params['eps'])

  # Define variational problem
  F = ((u_1 - u_n1) / k)*v_1*dx + dot(w, grad(u_1))*v_1*dx \
    + eps*dot(grad(u_1), grad(v_1))*dx + K*u_1*u_2*v_1*dx  \
    + ((u_2 - u_n2) / k)*v_2*dx + dot(w, grad(u_2))*v_2*dx \
    + eps*dot(grad(u_2), grad(v_2))*dx + K*u_1*u_2*v_2*dx  \
    + ((u_3 - u_n3) / k)*v_3*dx + dot(w, grad(u_3))*v_3*dx \
    + eps*dot(grad(u_3), grad(v_3))*dx - K*u_1*u_2*v_3*dx + K*u_3*v_3*dx \
    - f_1*v_1*dx - f_2*v_2*dx - f_3*v_3*dx

  #create time series for reading velocity data
  timeseries_w = TimeSeries(params['velocity_file'])
  #create time series for writing concentration data
  timeseries_u = TimeSeries(params['concentration_file'])

  #create output files, which can collect data across multiple processes
  xdmf_file = XDMFFile(params['name']+".xdmf")
  xdmf_file.parameters["flush_output"] = True
  xdmf_file.parameters["functions_share_mesh"] = True
  xdmf_file.parameters["rewrite_function_mesh"] = False 

  # Time-stepping
  t = 0
  for n in range(params['num_steps']):

      # Update current time
      t += dt

      # Read velocity from file
      timeseries_w.retrieve(w.vector(), t)

      # Solve variational problem for time step
      solve(F == 0, uh)

      #save concentrations to time series
      timeseries_u.store(uh.vector(), t)

      # Save solution to file
      _u_1, _u_2, _u_3 = uh.split()
      xdmf_file.write(_u_1, t)
      xdmf_file.write(_u_2, t)
      xdmf_file.write(_u_3, t)

      # Update previous solution
      u_n.assign(uh)
      
  #close file
  xdmf_file.close()

if __name__ == '__main__':
   diffusion_reaction_cylinder()
