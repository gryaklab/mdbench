from dolfin import *
from ufl_legacy import indices
import numpy
from typing import Dict

P_HEAT_LASER_FEM ={
    'name': 'heat3d_laser',
    'dim': 3,
    'equation': ["rho*c*u_t = kappa*nabla^2u + rho*Laser"],
    'start_time': 0.0,
    'end_time': 50.0,
    'num_steps': 500,  #Number of time steps
    #mesh parameters
    'Nx': 200,
    'Ny': 200,
    'Nz': 2,
    #grid parameters - lower left point and upper right corners
    'Gl': [0.0, 0.0, 0.0],
    #set to [l,l,thickness], update if these parameters are changed below
    'Gr': [0.1,0.1,0.001],
    #system parameters
    'rho': 7860.0,      # mass density of steel in kg/m^3
    'c': 624.0,	        # heat capacity in J/(kg K)
    'kappa': 30.1,	    # thermal conductivity in W/(m K)
    'h': 18.0,		    # heat convection out of the surface into ambient in W/(m^2 K)
    'Ta': 300.0,	    # ambient temperature in K
    'l': 0.1,		    # length in x and y directions in m 
    'thickness': 0.001,	# thickness of the plate in m
    'P': 3.0e6,		    # laser power in W/kg 
    'speed': 0.02	    # laser speed in m/s
}

def heat_equation_laser(params: Dict | None = None):
    """Solves the heat equation on a 3D metal plate grid with a laser heat source
    and no boundary conditions.

    The parameters dictionary can be used to modify the physical system and spatiotemporal
    resolution of the problem.

    Code adapted from:
    [Abali 2017] Computational Reality, Problem 10 http://dx.doi.org/10.1007/978-981-10-2444-3
    """
    if params is None:
        params = P_HEAT_LASER_FEM
    
    #dolphin parameters
    parameters["allow_extrapolation"]=True
    parameters["form_compiler"]["cpp_optimize"] = True

    #Create mesh and define function space
    mesh = BoxMesh(Point(*params['Gl']), Point(*params['Gr']), params['Nx'], params['Ny'], params['Nz'])
    #degree 1 Lagrange
    Space = FunctionSpace(mesh, 'P', 1)
    cells = MeshFunction('size_t',mesh,mesh.topology().dim())
    facets = MeshFunction('size_t',mesh,mesh.topology().dim()-1)
    da = Measure('ds', domain=mesh, subdomain_data=facets)
    dv = Measure('dx', domain=mesh, subdomain_data=cells)

    #create output file, which can collect data across multiple processes
    xdmf_file = XDMFFile(params['name']+".xdmf")
    xdmf_file.parameters["flush_output"] = True
    xdmf_file.parameters["functions_share_mesh"] = True
    xdmf_file.parameters["rewrite_function_mesh"] = False

    #compute delta t
    Dt=(params['end_time']-params['start_time'])/params['num_steps']
       
    #initial condition and u's prior time step value
    initial_u = Expression("u_ini", degree=1, u_ini=params['Ta'])
    u_n = interpolate(initial_u, Space)
    
    #setup time-dependent heat source; a higher degree (4) is used here due to the higher rate of change in the laser function
    Laser = Expression("P*exp(-50000.0*(pow(x[0]-0.5*l*(1+0.5*sin(2*pi*t/t_e)), 2)+pow(x[1]-velo*t, 2)))",
                       degree=4,P=params['P'],t=params['start_time'],t_e=params['end_time']/10.,l=params['l'],velo=params['speed'])

    #creates spatial indices, which can then be used in the variational form; they adhere to Einstein notation
    i, j =indices(2)

    #setup variational form
    u = TrialFunction(Space)
    v = TestFunction(Space)
    Form = (params['rho']*params['c']/Dt*(u-u_n)*v \
        + params['kappa']*u.dx(i)*v.dx(i) \
        - params['rho']*Laser*v ) * dv \
        + params['h']*(u-params['Ta'])*v*da

    #separate form in to left (bilinear) and right (linear) parts
    left=lhs(Form)
    right=rhs(Form)

    #assemble bilinear form, which is time-independent
    A = assemble(left)
    # dynamically assembled at each timestep
    b = None

    #the solution
    uh = Function(Space, name='uh')

    #solve PDE
    for t in numpy.arange(params['start_time'],params['end_time'],Dt):
        #update heat source
        Laser.t=t
        #assemble time-dependent right-hand side
        b=assemble(right, tensor=b) 
        #solve linear problem
        solve(A, uh.vector(), b, 'cg')
        #write solution value at time t to file
        xdmf_file.write(uh,t)
        #update prior solution
        u_n.assign(uh)
    
    #close file
    xdmf_file.close()

if __name__ == '__main__':
    heat_equation_laser()