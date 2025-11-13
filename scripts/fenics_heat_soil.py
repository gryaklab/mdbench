"""
Temperature variations in the ground.
Code adapted from original source:
[FEniCS '12](https://doi.org/10.1007/978-3-642-23099-8) *Automated Solution of Differential Equations by the Finite Element Method: The FEniCS Book* by Logg et al.
"""

from dolfin import *
from copy import deepcopy

P_HEAT_MESH_3D ={
    'name': 'heat_soil_3d',
    'dim': 3,
    'equation': r"\rho*c*u_t = \kappa*nabla^2u + f",
    'start_time': 0,
    'num_periods': 2,       #as the boundary condition is periodic, choose the number of periods to capture (2 days)
    'step_duration': 300,   #duration of step in seconds (5 minutes)
    #mesh parameters
    'Nx': 50,
    'Ny': 50,
    'Nz': 10,
    #spatial parameters
    'depth': 1.5,            #soil depth in meters
    'width': None            #soil width, by default will be half of 
}

P_HEAT_SYSTEM_P1 = {
    'T_R':      10,      #reference temperature, Celsius
    'T_A':      10,      #amplitude of temperature variations, Celsius
    'omega':    7.27e-5, #frequency of temperature variations, Hz
    'rho':      1500,    #soil density, kg/m^3
    'c':        1600,    #heat capacity, N*m*kg^-1
    'kappa_0':  2.3,     #thermal conductivity in large region, in K−1*N*s−1 
    'kappa_1':  100      #thermal conductivity in small region, in K−1*N*s−1 
}

P_HEAT_SYSTEM_P2 = {
    'T_R':      10,      #reference temperature, Celsius
    'T_A':      10,      #amplitude of temperature variations, Celsius
    'omega':    7.27e-5, #frequency of temperature variations, Hz
    'rho':      1500,    #soil density, kg/m^3
    'c':        1600,    #heat capacity, N*m*kg^-1
    'kappa_0':  12.3,    #thermal conductivity in large region, in K−1*N*s−1 
    'kappa_1':  10e4     #thermal conductivity in small region, in K−1*N*s−1 
}

P_HEAT_MESH_1D = deepcopy(P_HEAT_MESH_3D)
P_HEAT_MESH_1D['name']='heat_soil_1d'
P_HEAT_MESH_1D['dim']=1

P_HEAT_MESH_2D = deepcopy(P_HEAT_MESH_3D)
P_HEAT_MESH_2D['name']='heat_soil_2d'
P_HEAT_MESH_2D['dim']=2

P_HEAT_1D_P1 = P_HEAT_MESH_1D | P_HEAT_SYSTEM_P1
P_HEAT_1D_P1['name']=P_HEAT_1D_P1['name']+'_p1'
P_HEAT_1D_P2 = P_HEAT_MESH_1D | P_HEAT_SYSTEM_P2
P_HEAT_1D_P2['name']=P_HEAT_1D_P2['name']+'_p2'

P_HEAT_2D_P1 = P_HEAT_MESH_2D | P_HEAT_SYSTEM_P1
P_HEAT_2D_P1['name']=P_HEAT_2D_P1['name']+'_p1'
P_HEAT_2D_P2 = P_HEAT_MESH_2D | P_HEAT_SYSTEM_P2
P_HEAT_2D_P2['name']=P_HEAT_2D_P2['name']+'_p2'

P_HEAT_3D_P1 = P_HEAT_MESH_3D | P_HEAT_SYSTEM_P1
P_HEAT_3D_P1['name']=P_HEAT_3D_P1['name']+'_p1'
P_HEAT_3D_P2 = P_HEAT_MESH_3D | P_HEAT_SYSTEM_P2
P_HEAT_3D_P2['name']=P_HEAT_3D_P2['name']+'_p2'

def heat_equation_soil(params):
    #Create mesh and define function space
    D=params['depth']
    if params['width'] is None:
        W = D/2.0
    if params['dim'] == 1:
        mesh = IntervalMesh(params['Nx'], -D, 0)
    elif params['dim'] == 2:
        mesh = RectangleMesh(Point(-W/2, -D), Point(W/2, 0), params['Nx'], params['Ny'])
    elif params['dim'] == 3:
        mesh = BoxMesh(Point(-W/2, -W/2, -D), Point(W/2, W/2, 0),
                params['Nx'], params['Ny'], params['Nz'])
    V = FunctionSpace(mesh, 'Lagrange', 1)

    #set boundary conditions
    T_0 = Expression('T_R + T_A*sin(omega*t)',
                    T_R=params['T_R'], T_A=params['T_A'], omega=params['omega'], t=params['start_time'], degree=1)

    def surface(x, on_boundary):
        return on_boundary and abs(x[params['dim']-1]) < 1E-14

    bc = DirichletBC(V, T_0, surface)

    #Setup time steps
    period = 2*pi/params['omega']
    t_stop = period*params['num_periods']
    dt = params['step_duration']

    #used to specify time discretization method (theta=1 yields backward difference)
    theta = 1

    #Set thermal conductivity of regions via CPP conditional statements
    kappa_str = {
        1:'(x[0] > -D/2 && x[0] < (-D/2 + D/4)) ? kappa_1 : kappa_0',
        2:'(x[0] > -W/4 && x[0] < W/4 '\
                '&& x[1] > -D/2 && x[1] < (-D/2 + D/4)) ? '\
                'kappa_1 : kappa_0',
        3:'(x[0] > -W/4 && x[0] < W/4 '\
                '&& x[1] > -W/4 && x[1] < W/4 '\
                '&& x[2] > -D/2 && x[2] < (-D/2 + D/4)) ? '\
                'kappa_1 : kappa_0'
    }
    kappa = Expression(kappa_str[params['dim']],
                    D=D, W=W, kappa_0=params['kappa_0'], kappa_1=params['kappa_1'], degree=1)

    # Define initial condition
    u_n = interpolate(Constant(params['T_R']), V)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(0)
    a = params['rho']*params['c']*u*v*dx + theta*dt*kappa*\
        inner(nabla_grad(v), nabla_grad(u))*dx
    L = (params['rho']*params['c']*u_n*v + dt*f*v -
        (1-theta)*dt*kappa*inner(nabla_grad(v), nabla_grad(u)))*dx

    A = assemble(a)
    b = None  # variable used for memory savings in assemble calls

    #create output file, which can collect data across multiple processes
    xdmf_file = XDMFFile(params['name']+".xdmf")
    xdmf_file.parameters["flush_output"] = True
    xdmf_file.parameters["functions_share_mesh"] = True
    xdmf_file.parameters["rewrite_function_mesh"] = False


    #the solution
    uh = Function(V, name='uh')

    #solve PDE
    t = dt
    while t <= t_stop:
        #assemble RHS
        b = assemble(L, tensor=b)
        #update boundary condition
        T_0.t = t
        #apply boundary condition
        bc.apply(A, b)
        #linear solve
        solve(A, uh.vector(), b)
        #write to file
        xdmf_file.write(uh,t)
        t += dt
        u_n.assign(uh)


    #close file
    xdmf_file.close()

if __name__ == '__main__':
    heat_equation_soil(P_HEAT_1D_P1)
    heat_equation_soil(P_HEAT_2D_P1)
    heat_equation_soil(P_HEAT_3D_P1)
    heat_equation_soil(P_HEAT_1D_P2)
    heat_equation_soil(P_HEAT_2D_P2)
    heat_equation_soil(P_HEAT_3D_P2)
