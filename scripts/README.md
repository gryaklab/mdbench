## Run FEniCS Script
First, install [FEniCS software](https://fenics.readthedocs.io/en/latest/installation.html). Then, you can run the script using a single processor:
`python3 fenics_heat_soil.py`
or using MPI with multiple processors (up to 24 on Panther):
`mpirun -n 6 python3 fenics_heat_soil.py`

## FEniCS vs.FEniCSx
FEniCS**x** is a revamped API of the original FEniCS package. Most scripts utilize the older version (FEniCS). The prefix for each script indicates which version of the library is used, e.g., *fenics*_burgers1d.py vs. *fenicsx*_heat2d.py. Both versions are available on Panther and will be copied into the virtual environment if you include system packages as above.

## FEniCS Documentation
- [FEniCS Book](https://doi.org/10.1007/978-3-642-23099-8) *Automated Solution of Differential Equations by the Finite Element Method: The FEniCS Book* by Logg et al. is the most comprehensive reference for FEniCS. THe book is located in the Documentation folder of the project's Dropbox folder.
- [FEniCS Tutorial](https://doi.org/10.1007/978-3-319-52462-7) *Solving PDEs in Python: The FEniCS Tutorial I* by Langtangen and Logg is a more complete introduction.

## FEniCSx Documentation
- [FEniCSx Tutorial](https://jsdokken.com/dolfinx-tutorial/index.html) - This covers the latest version and includes some additional systems to consider implementing.
- [DolfinX Demos](https://docs.fenicsproject.org/dolfinx/v0.9.0/python/demos.html) - Has a number of additional systems and their implementations to consider.
- [DolfinX API Documentation](https://docs.fenicsproject.org/dolfinx/v0.9.0/python/api.html) - Very terse.
- [MeshIO Documentation](https://github.com/nschloe/meshio) - Very poor, but you can determine things via the source code.

## Steps to Generating a PDE Dataset using FEniCS
1. Identify the computational domain ($\Omega$), the PDE, its boundary conditions, and source terms ($f$).
2. Reformulate the PDE as a finite element variational problem.
3. Write a Python program which defines the computational domain, the variational problem, the boundary conditions, and source terms, using the corresponding FEniCS abstractions.
4. Call FEniCS to solve the boundary-value problem.
5. Processes the mesh solution into an appropriate format for the pipeline. All results are currently stored in XMDF format, with the main solution named `uh`.

## Structured Data
FEniCS and other FEM software typically use *unstructured* meshes, meaning that the solution for a specified PDE system is computed on cells whose boundaries are not aligned with a rectilinear (i.e., *structured*) grid. However, most model discovery methods require a structured grid. As such, there are scripts of the form `<system_name>_structured.py` that will
load the data related to `<system_name>` and generate it in a structured format for downstream use. The format of this structured data is as follows:
- `<system_name>_structured_mesh.csv` - A uniformly-spaced mesh in rectilinear coordinates (current systems are 2D, but this can be readily extended to higher order rectilinear coordinate systems),
- `<system_name>_structured_timesteps.csv` - The time steps at which the solution was computed.
- `<system_name>_<solution_name>.csv` - The interpolated values of the solution on the structured mesh at each time point. The rows are ordered in increasing time steps, e.g., if there are 2,500 grid points and 5 time steps, the first 2,500 rows of the file will be for $t_1$, while the last 2,500 rows will be for $t_5$. The current method uses linear interpolation, though nearest neighbor or cubic interpolation can be readily computed (see [SciPy `griddata` documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.griddata.html)). For the current systems, the files contain the following columns of data in the order specified below:
    - **navier_stokes_cylinder**: Velocity_component_1, Velocity Component 2, Pressure
    - **diffusion_reaction_cylinder**: Velocity_component_1, Velocity Component 2, Pressure, Concentration_1, Concentration_2, Concentration_3

## Additional Datasets

| Dataset Name                      | Source   | Equation |
|-----------------------------------|----------|----------|
| 1D Burgers (Viscous)              |[Burkardt](https://people.sc.fsu.edu/~jburkardt/fenics_src/burgers_time_viscous/burgers_time_viscous.html)  | $u_t = -uu_x + 0.05u_{xx}$ |
| 2D Heat                           |[FEniCSx](https://jsdokken.com/dolfinx-tutorial/index.html) | $u_t = \nabla^2u + f$ |
| 3D Linear Elasticity (Undamped)   |ERL 2002  | $\nabla \cdot \sigma + \rho b = \rho u_{tt}$ <br> $\sigma= \lambda \mathrm{Tr}(\epsilon) \mathbf{I}_2 +2\mu\epsilon$ <br> $\epsilon=(\nabla u + (\nabla u)^\intercal)/2$|
| 3D Linear Elasticity (Damped)     |ERL 2002  | Same as above |
| 1D, 2D, 3D Heat Soil (P1)         |FEniCS '12| $\rho c u_t = \kappa \nabla^2u + f$, $\kappa$ varies across domain |
| 1D, 2D, 3D Heat Soil (P2)         |FEniCS '12| Same as above |

### Sources
- [Abali](https://doi.org/10.1007/978-981-10-2444-3) *Computational Reality* by Abali
- [ERL 2002](https://doi.org/10.1007/s00466-001-0273-z) *The analysis of the generalized-alpha method for non-linear dynamic problems* by Erlicher et al. [Code](https://comet-fenics.readthedocs.io/en/latest/demo/elastodynamics/demo_elastodynamics.py.html)
- [FEniCS '12](https://doi.org/10.1007/978-3-642-23099-8) *Automated Solution of Differential Equations by the Finite Element Method: The FEniCS Book* by Logg et al.
- [FEniCS '16](https://doi.org/10.1007/978-3-319-52462-7) *Solving PDEs in Python: The FEniCS Tutorial I* by Langtangen and Logg

