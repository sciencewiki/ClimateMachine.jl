# # Heat equation tutorial

# In this tutorial, we'll be solving the [heat equation](https://en.wikipedia.org/wiki/Heat_equation):

# ``
# \frac{∂ T}{∂ t} +  ∇ ⋅(wT) + ∇ ⋅ (-α ∇T) = 0
# ``

# where
#  - `α` is the eddy diffusivity
#  - `T` is the temperature (unknown we're solving for)
#  - `w` is the vertical velocity (unknown we're solving for)

# To put this in the form of CLIMA's [`BalanceLaw`](@ref CLIMA.DGMethods.BalanceLaw), we'll re-write the equation as:

# ``
# \frac{∂ T}{∂ t} + ∇ ⋅ (F(T,t)) = 0
# ``

# where
#  - `F(T,t) = -α ∇T + uT` is the diffusive flux

# with boundary conditions
#  - Fixed temperature `T_surface` at z_{min} (non-zero Dirichlet)
#  - No thermal flux at z_{min} (zero Neumann)

# Solving these equations is broken down into the following steps:
# - 1) Preliminary configuration
# - 2) PDEs
# - 3) Space discretization
# - 4) Time discretization
# - 5) Solver hooks / callbacks
# - 6) Solve
# - 7) Post-processing

# YAIR 
# think about the code structure 
# add the source functions calls
# think about how to define an arbitrary number of subdomains and their variables , BC etc 
# is the current solver indeed explicit ?
# how do I use the thermynamics for buoyancy and such 



# # Preliminary configuration

# ## Loading code

# First, we'll load our pre-requisites
#  - load external packages:
using MPI
using Test
using Logging
using Printf
using NCDatasets
using LinearAlgebra
using OrderedCollections
using Interpolations
using DelimitedFiles
using Plots
using StaticArrays

#  - load necessary CLIMA modules:
using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.Writers
using CLIMA.VTK
using CLIMA.Mesh.Elements: interpolationmatrix
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.DGmethods: BalanceLaw, LocalGeometry
using CLIMA.MPIStateArrays
using CLIMA.GenericCallbacks: EveryXWallTimeSeconds, EveryXSimulationSteps
using CLIMA.GenericCallbacks
using CLIMA.ODESolvers
using CLIMA.VariableTemplates

#  - import necessary CLIMA modules: (`import` indicates that we must provide implementations of these structs/methods)
import CLIMA.DGmethods: vars_state_auxiliary,
                        vars_state_conservative,
                        vars_state_gradient,
                        vars_state_gradient_flux,
                        source!,
                        flux_second_order!,
                        flux_first_order!,
                        compute_gradient_argument!,
                        compute_gradient_flux!,
                        update_aux!,
                        nodal_update_aux!,
                        init_state_auxiliary!,
                        init_state_conservative!,
                        boundary_state!


include(joinpath("Closures","entr_detr.jl"))
include(joinpath("Closures","buoyancy.jl"))
include(joinpath("Closures","mixing_length.jl"))
include(joinpath("Closures","microphysics.jl"))
include(joinpath("Closures","subdomain_statistics.jl"))
include(joinpath("Closures","surface.jl"))
include(joinpath("Closures","pressure.jl"))
include(joinpath("Closures","eddy_diffusivity.jl"))



# ## Initialization

# Define the float type (`Float64` or `Float32`)
FT = Float64;
# Initialize CLIMA for CPU.
CLIMA.init(; disable_gpu=true);

# Load some helper functions (soon to be incorporated into `CLIMA/src`)
include(joinpath("..","helper_funcs.jl"))
include(joinpath("..","plotting_funcs.jl"))

# # Define the set of Partial Differential Equations (PDEs)

# Model parameters can be stored in the particular [`BalanceLaw`](@ref CLIMA.DGMethods.BalanceLaw), in this case, a `TurbulenceConvectionModel`:

Base.@kwdef struct TurbulenceConvectionModel{FT} <: BalanceLaw
  "density"
  ρ::FT = 1
  "eddy diffusivity"
  α::FT = 0.01
  "Initial conditions for temperature"
  initialT::FT = 295.15
  "Surface boundary value for temperature (Dirichlet boundary conditions)"
  surfaceT::FT = 300.0
  "Initial conditions for velocity"
  initialw::FT = 0.0
  "Surface boundary value for velocity (zero)"
  surfacew::FT = 0.0
end

# Create an instance of the `TurbulenceConvectionModel`:
m = TurbulenceConvectionModel{FT}();

# All of the methods defined below, in this section, were `import`ed in [Loading code](@ref), and we must provide our own implementation as these methods are called inside `solve!`. Each of these methods have a fall-back of "do nothing" if we don't, or fail to, implement these methods after `import`ing.

# Specify auxiliary variables for `TurbulenceConvectionModel` (stored in `aux`)
vars_state_auxiliary(::TurbulenceConvectionModel, FT) = @vars(z::FT, T::FT, w::FT);

# Specify state variables, the variables solved for in the PDEs, for `TurbulenceConvectionModel` (stored in `Q`)
vars_state_conservative(::TurbulenceConvectionModel, FT) = @vars(ρT::FT, ρw::FT);

# Specify state variables whose gradients are needed for `TurbulenceConvectionModel`
vars_state_gradient(::TurbulenceConvectionModel, FT) = @vars(T::FT, w::FT);

# Specify gradient variables for `TurbulenceConvectionModel`
vars_state_gradient_flux(::TurbulenceConvectionModel, FT) = @vars(∇T::SVector{3,FT}, ∇w::SVector{3,FT});

# Specify the initial values in `aux::Vars`, which are available in `init_state_conservative!`. Note that
# - this method is only called at `t=0`
# - `aux.z` and `aux.T` are available here because we've specified `z` and `T` in `vars_state_auxiliary`
function init_state_auxiliary!(m::TurbulenceConvectionModel, aux::Vars, geom::LocalGeometry)
  aux.z = geom.coord[3]
  aux.T = m.initialT
  aux.w = m.initialw
end;

# Specify the initial values in `state::Vars`. Note that
# - this method is only called at `t=0`
# - `state.ρT` is available here because we've specified `ρT` in `vars_state_conservative`
function init_state_conservative!(m::TurbulenceConvectionModel, state::Vars, aux::Vars, coords, t::Real)
  state.ρT = m.ρ * aux.T
  state.ρw = m.ρ * aux.w
end;

# The remaining methods, defined in this section, are called at every time-step in the solver by the [`BalanceLaw`](@ref CLIMA.DGMethods.BalanceLaw) framework.

# Overload `update_aux!` to call `nodal_update_aux!`, or any other auxiliary methods
function update_aux!(dg::DGModel, m::TurbulenceConvectionModel, Q::MPIStateArray, t::Real, elems::UnitRange)
  nodal_update_aux!(nodal_update_aux!, dg, m, Q, t, elems)
  return true # TODO: remove return true
end;

# Compute/update all auxiliary variables at each node. Note that
# - `aux.T` is available here because we've specified `T` in `vars_state_auxiliary`
function nodal_update_aux!(m::TurbulenceConvectionModel, state::Vars, aux::Vars, t::Real)
  aux.T = state.ρT / m.ρ
  aux.w = state.ρw / m.ρ
end;

# Since we have diffusive fluxes, we must tell CLIMA to compute the gradient of `T`. Here, we specify how `T` is computed. Note that
#  - `transform.T` is available here because we've specified `T` in `vars_state_gradient`
function compute_gradient_argument!(m::TurbulenceConvectionModel, transform::Vars, state::Vars, aux::Vars, t::Real)
  transform.T = state.ρT / m.ρ
  transform.w = state.ρw / m.ρ
end;

# Specify where in `diffusive::Vars` to store the computed gradient in `compute_gradient_argument!`. Note that:
#  - `diffusive.∇T` is available here because we've specified `∇T` in `vars_state_gradient_flux`
#  - `∇transform.T` is available here because we've specified `T`  in `vars_state_gradient`
function compute_gradient_flux!(m::TurbulenceConvectionModel, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real)
  diffusive.∇T = ∇transform.T
  diffusive.∇w = ∇transform.w
end;

# We do no have sources, nor non-diffusive fluxes.
# function source!(m::TurbulenceConvectionModel, _...); end;
function source!(m::TurbulenceConvectionModel, source::Vars, state::Vars, diffusive::Vars, aux::Vars, t::Real, direction)
    source.ρT += 0.01 * state.ρT
    # add here the source functions calls
end;

# function flux_first_order!(m::TurbulenceConvectionModel, _...); end;
function flux_first_order!(m::TurbulenceConvectionModel,flux::Grad,state::Vars,aux::Vars,t::Real)
   ρ = m.ρ
   ρinv = 1 / ρ
   ρw = state.ρw
   w = ρinv * ρw
   # advective terms
   flux.ρw = ρ * w .* w'
   flux.ρT = w * state.ρT
end;

# Compute diffusive flux (``F(T,t) = -α ∇T`` in the original PDE). Note that:
# - `diffusive.∇T` is available here because we've specified `∇T` in `vars_state_gradient_flux`
function flux_second_order!(m::TurbulenceConvectionModel, flux::Grad, state::Vars, diffusive::Vars, hyperdiffusive::Vars, aux::Vars, t::Real)
   flux.ρT -= m.α * diffusive.∇T
   flux.ρw -= m.α * diffusive.∇w
end;

# ### Boundary conditions

# Boundary conditions are specified for diffusive and non-diffusive terms

# The boundary conditions for `ρT` are specified here for non-diffusive terms
function boundary_state!(nf, m::TurbulenceConvectionModel, state⁺::Vars, aux⁺::Vars,
                         nM, state⁻::Vars, aux⁻::Vars, bctype, t, _...)
  if bctype == 1 # surface
    state⁺.ρT = m.ρ * m.surfaceT
    state⁺.ρw = m.ρ * m.surfacew
  elseif bctype == 2 # bottom
    state⁺.ρw = m.ρ * m.surfacew
  end
end;

# The boundary conditions for `ρT` are specified here for diffusive terms
function boundary_state!(nf, m::TurbulenceConvectionModel, state⁺::Vars, diff⁺::Vars,
                         aux⁺::Vars, nM, state⁻::Vars, diff⁻::Vars, aux⁻::Vars,
                         bctype, t, _...)
  if bctype == 1 # surface
    state⁺.ρT = m.ρ * m.surfaceT
    state⁺.ρw = m.ρ * m.surfacew
  elseif bctype == 2 # bottom
    diff⁺.∇T = -diff⁻.∇T
    state⁺.ρw = m.ρ * m.surfacew
  end
end;

# # Spatial discretization

# Prescribe vector of vertical elements (in meters)
velems = collect(0:10) / 10;

# Prescribe polynomial order of basis functions in finite elements
N_poly = 5;

# Define topology (i.e., element connectivity)
topl = StackedBrickTopology(
  MPI.COMM_WORLD,
  (0.0:1,0.0:1,velems);
  periodicity = (true,true,false),
  boundary=((0,0),(0,0),(1,2)));

# Define grid, based on topology
grid = DiscontinuousSpectralElementGrid(
  topl,
  FloatType = FT,
  DeviceArray = Array,
  polynomialorder = N_poly);

# Configure the Discontinuous Galerkin (DG) model, based on the PDEs, grid and penalty terms
dg = DGModel(
  m,
  grid,
  # penalty terms for discretizations:
  CentralNumericalFluxFirstOrder(),
  CentralNumericalFluxSecondOrder(),
  CentralNumericalFluxGradient())

# # Time discretization

# We'll define the time-step based on the [Fourier number](https://en.wikipedia.org/wiki/Fourier_number)
Δ = min_node_distance(grid)

given_Fourier = 0.08;
Fourier_bound = given_Fourier*Δ^2 / m.α;
dt = Fourier_bound

# ## Initialize the state vector
# This initializes the state vector and allocates memory for the solution in space (`dg` has the model `m`, which describes the PDEs as well as the function used for initialization):
Q = init_ode_state(dg, Float64(0))

# ## Initialize the ODE solver
# Here, we use an explicit Low-Storage [Runge-Kutta](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods) method
lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0);

# ## Inspect the initial conditions

# Let's export a plot of the initial state
# output_dir = joinpath(dirname(dirname(pathof(CLIMA))), "output", "land");
output_dir = joinpath(@__DIR__);
output_dir = pwd();
mkpath(output_dir)

state_vars = get_vars_from_stack(grid, Q, m, vars_state_conservative)
aux_vars = get_vars_from_stack(grid, dg.state_auxiliary, m, vars_state_auxiliary)
all_vars = OrderedDict(state_vars..., aux_vars...)
export_plot_snapshot(all_vars, ("ρT",), joinpath(output_dir, "initial_condition.png"))
# ![](initial_condition.png)

# It matches what we have in `init_state_conservative!(m::TurbulenceConvectionModel, ...)`, so let's continue.

# # Solver hooks / callbacks

# Define simulation time (all units are SI) and number of outputs over this interval
const timeend = 40;
const n_outputs = 5;

# This equates to exports every ceil(Int, timeend/n_outputs) time-step:
const every_x_simulation_time = ceil(Int, timeend/n_outputs);

# Create a dictionary for `z` coordinate (and convert to cm) NCDatasets IO:
z_scale = 100 # convert from meters to cm
dims = OrderedDict("z" => collect(get_z(grid, z_scale)))

# Create a DataFile, which is callable to get the name of each file given a step
output_data = DataFile(joinpath(output_dir, "output_data"))

# CLIMA's time-steppers provide hooks, or callbacks, which allow users to inject code to be executed at specified intervals. In this callback, the state and aux variables are collected, combined into a single `OrderedDict` and written to a NetCDF file (for each output step `step`).
step = [0]
callback = GenericCallbacks.EveryXSimulationTime(every_x_simulation_time, lsrk) do (init = false)
  state_vars = get_vars_from_stack(grid, Q, m, vars_state_conservative; exclude=[])
  aux_vars = get_vars_from_stack(grid, dg.state_auxiliary, m, vars_state_auxiliary; exclude=["z"])
  all_vars = OrderedDict(state_vars..., aux_vars...)
  write_data(NetCDFWriter(), output_data(step[1]), dims, all_vars, gettime(lsrk))
  step[1]+=1
  nothing
end

# # Solve

# This is the main "solve" method call. While users do not have access to the time-stepping loop, code and function calls may be injected via `callbacks` keyword argument, which is a `Tuple` of [`GenericCallbacks`](@ref).
solve!(Q, lsrk; timeend=timeend, callbacks=(callback,))

# # Post-processing

# Our solution has now been calculated and exported to NetCDF files in `output_dir`. Let's collect them all into a dictionary whose keys are output interval and values of NCDataset's:

all_data = collect_data(output_data, step[1])

# To get `T` at time-step 0, we can use `T_at_t_0 = all_data[0]["T"][:]`

# Let's plot the solution:

export_plot(all_data, ("ρT",), joinpath(output_dir, "solution_vs_time.png"))
# ![](solution_vs_time.png)

# The results look as we would expect: a fixed temperature at the bottom is resulting in heat flux that propagates up the domain. To run this file, and inspect the solution in `all_data`, include this tutorial in the Julia REPL with:

# ```julia
# include(joinpath("tutorials", "Land", "Heat", "heat_equation.jl"))
# ```

