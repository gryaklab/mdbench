from fenics import *
import numpy as np
from scipy.interpolate import griddata
import sys

PREFIX = r"diffusion_reaction_cylinder_structured_"
SAMPLES_X = 250
SAMPLES_Y = 50

def diffusion_reaction_cylinder_structured_numpy():
    #read mesh
    msh = Mesh("navier_stokes_cylinder.xml.gz")

    #setup functions
    P1 = FiniteElement('P', triangle, 1)
    element = MixedElement([P1, P1, P1])
    V = FunctionSpace(msh, element)
    uh = Function(V, name="uh")
    
    #function spaces must match those specified in fenics_navier_stokes_cylinder.py.
    W = VectorFunctionSpace(msh, 'P', 2)
    w = Function(W)
    Q = FunctionSpace(msh, 'P', 1)
    ph = Function(Q, name='ph')

    #h5 file, must leave off file extension
    timeseries_u = TimeSeries(r"diffusion_reaction_cylinder_concentration_series")
    timeseries_w = TimeSeries(r"navier_stokes_cylinder_velocity_series")
    timeseries_p = TimeSeries(r"navier_stokes_cylinder_pressure_series")

    #get mesh x,y coordinates
    xy = np.array(msh.coordinates())

    #original LL and UR box coordinates
    #'Gl': [0.0, 0.0],
    #'Gr': [2.2, 0.41],
    
    #new uniform grid
    num_samples = SAMPLES_X*SAMPLES_Y
    um_X,um_Y = np.mgrid[0:2.2:complex(0,SAMPLES_X), 0:0.41:complex(0,SAMPLES_Y)]

    #save mesh and time steps separately to save space
    np.savetxt(PREFIX+"mesh.csv",
               np.column_stack(
                   (um_X.reshape(num_samples,),
                    um_Y.reshape(num_samples,))),
                    delimiter=',')
    
    time_steps = timeseries_u.vector_times()
    np.savetxt(PREFIX+"timesteps.csv",
               np.array(time_steps).reshape(len(time_steps),1),
               delimiter=',')

    with open(PREFIX+"concentrations.csv","a") as f:
        #process remaining time steps
        for t in time_steps:
            #Read velocity and pressure solutions from file
            timeseries_w.retrieve(w.vector(), t)
            timeseries_p.retrieve(ph.vector(), t)

            #Read concentrations from file
            timeseries_u.retrieve(uh.vector(), t)
            
            w_values = np.array([w(pt) for pt in xy])
            p_values = np.array([ph(pt) for pt in xy])
            w_0_i = griddata(xy, w_values[:,0], (um_X, um_Y), method='linear')
            w_1_i = griddata(xy, w_values[:,1], (um_X, um_Y), method='linear')
            p_i = griddata(xy, p_values, (um_X, um_Y), method='linear')
            
            u_values = np.array([uh(pt) for pt in xy])
            u_0_i = griddata(xy, u_values[:,0], (um_X, um_Y), method='linear')
            u_1_i = griddata(xy, u_values[:,1], (um_X, um_Y), method='linear')
            u_2_i = griddata(xy, u_values[:,2], (um_X, um_Y), method='linear')
            
            #create numpy array
            curr_data = np.column_stack(
                (w_0_i.reshape(num_samples,),
                 w_1_i.reshape(num_samples,),
                 p_i.reshape(num_samples,),
                 u_0_i.reshape(num_samples,),
                 u_1_i.reshape(num_samples,),
                 u_2_i.reshape(num_samples,))
            )
            np.savetxt(f,curr_data,delimiter=",")


def diffusion_reaction_cylinder_structured_fenics():
    #read mesh
    msh = Mesh("navier_stokes_cylinder.xml.gz")

    #setup functions
    P1 = FiniteElement('P', triangle, 1)
    element = MixedElement([P1, P1, P1])
    V = FunctionSpace(msh, element)
    uh = Function(V, name="uh")

    #function spaces must match those specified in fenics_navier_stokes_cylinder.py.
    W = VectorFunctionSpace(msh, 'P', 2)
    w = Function(W)
    Q = FunctionSpace(msh, 'P', 1)
    ph = Function(Q, name='ph')

    #h5 file, must leave off file extension
    timeseries_u = TimeSeries(r"diffusion_reaction_cylinder_concentration_series")
    timeseries_w = TimeSeries(r"navier_stokes_cylinder_velocity_series")
    timeseries_p = TimeSeries(r"navier_stokes_cylinder_pressure_series")

    #get mesh x,y coordinates
    xy = np.array(msh.coordinates())

    #original LL and UR box coordinates
    #'Gl': [0.0, 0.0],
    #'Gr': [2.2, 0.41],
    
    #new uniform grid
    num_samples = SAMPLES_X*SAMPLES_Y

    #create mesh for function evaluation
    x0 = np.linspace(0,2.2,SAMPLES_X)
    y0 = np.linspace(0,0.41,SAMPLES_Y)
    xv,yv = np.meshgrid(x0,y0,indexing='ij')

    xy = np.zeros((num_samples,2))
    ix_sample=0
    for i in range(SAMPLES_X):
        for j in range(SAMPLES_Y):
            xy[ix_sample,0] = xv[i,j]
            xy[ix_sample,1] = yv[i,j]
            ix_sample+=1

    #save mesh and time steps separately to save space
    np.savetxt(PREFIX+"mesh.csv", xy, delimiter=',')

    time_steps = timeseries_u.vector_times()
    np.savetxt(PREFIX+"timesteps.csv",
               np.array(time_steps).reshape(len(time_steps),1),
               delimiter=',')
    
    with open(PREFIX+"concentrations.csv","a") as f:
        #process remaining time steps
        radius_squared = .05**2
        for t in time_steps:
            curr_data = np.zeros((num_samples,6))

            #Read velocity and pressure solutions from file
            timeseries_w.retrieve(w.vector(), t)
            timeseries_p.retrieve(ph.vector(), t)

            #Read concentrations from file
            timeseries_u.retrieve(uh.vector(), t)
            
            #compute values on the uniform grid            
            for i in range(num_samples):
                #test if point is inside the domain, i.e., not the cylinder
                if ( (xy[i,0] - .2)**2 + (xy[i,1] - .2)**2) > radius_squared:
                    w_values = w(xy[i,:])
                    p_value = ph(xy[i,:])
                    u_values = uh(xy[i,:])
                    curr_data[i,0] = w_values[0]
                    curr_data[i,1] = w_values[1]
                    curr_data[i,2] = p_value
                    curr_data[i,3] = u_values[0]
                    curr_data[i,4] = u_values[1]
                    curr_data[i,5] = u_values[2]
            
            np.savetxt(f,curr_data,delimiter=",")

if __name__ == '__main__':
    if (len(sys.argv) -1 ) == 1 and sys.argv[1] == "1":
        PREFIX += "fenics_"
        diffusion_reaction_cylinder_structured_fenics()
    else:
        PREFIX += "numpy_"
        diffusion_reaction_cylinder_structured_numpy()