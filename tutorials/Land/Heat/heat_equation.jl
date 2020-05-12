# # Heat equation tutorial

# In this tutorial, we'll be solving the [heat equation](https://en.wikipedia.org/wiki/Heat_equation):

# ``
# \frac{∂ ρcT}{∂ t} + ∇ ⋅ (-α(T) ∇ρcT) = S(α(T), ∇T)
# \frac{∂ ρcT}{∂ t} + ∇ ⋅ (-α(T,∇T) ∇ρcT) = S(α(T,∇T), ∇T)
# ``

# where
#  - `t` is time
#  - `α(T)` is the thermal diffusivity
#  - `S(T)` α(T,∇T) * ∇T
#  - `T` is the temperature
#  - `ρ` is the density
#  - `c` is the heat capacity
#  - `ρcT` is the thermal energy

# To put this in the form of ClimateMachine's [`BalanceLaw`](@ref ClimateMachine.DGMethods.BalanceLaw), we'll re-write the equation as:

# ``
# \frac{∂ ρcT}{∂ t} + ∇ ⋅ (F(ρcT,t)) = 0
# ``

# where
#  - `F(ρcT,t) = -α(T) ∇ρcT` is the second-order flux

# with boundary conditions
#  - Fixed temperature ``T_{surface}`` at ``z_{min}`` (non-zero Dirichlet)
#  - No thermal flux at ``z_{min}`` (zero Neumann)

# Solving these equations is broken down into the following steps:
# 1) Preliminary configuration
# 2) PDEs
# 3) Space discretization
# 4) Time discretization
# 5) Solver hooks / callbacks
# 6) Solve
# 7) Post-processing

# # Preliminary configuration

# ## Loading code

# First, we'll load our pre-requisites
#  - load external packages:
using MPI
using NCDatasets
using OrderedCollections
using Plots
using StaticArrays

#  - load necessary ClimateMachine modules:
using ClimateMachine
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.Writers
using ClimateMachine.DGmethods
using ClimateMachine.DGmethods.NumericalFluxes
using ClimateMachine.DGmethods: BalanceLaw, LocalGeometry
using ClimateMachine.MPIStateArrays
using ClimateMachine.GenericCallbacks
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates

#  - import necessary ClimateMachine modules: (`import` indicates that we must provide implementations of these structs/methods)
import ClimateMachine.DGmethods:
    vars_state_auxiliary,
    vars_state_conservative,
    vars_state_gradient,
    vars_state_gradient_flux,
    source!,
    flux_second_order!,
    flux_first_order!,
    compute_gradient_argument!,
    compute_gradient_flux!,
    update_auxiliary_state!,
    nodal_update_auxiliary_state!,
    init_state_auxiliary!,
    init_state_conservative!,
    boundary_state!

# ## Initialization

# Define the float type (`Float64` or `Float32`)
FT = Float64;
# Initialize ClimateMachine for CPU.
ClimateMachine.init(; disable_gpu = true);

const clima_dir = dirname(dirname(pathof(ClimateMachine)));
# Load some helper functions (soon to be incorporated into `ClimateMachine/src`)
include(joinpath(clima_dir, "tutorials", "Land", "helper_funcs.jl"));
include(joinpath(clima_dir, "tutorials", "Land", "plotting_funcs.jl"));

# # Define the set of Partial Differential Equations (PDEs)

# ## Define the model

# Model parameters can be stored in the particular [`BalanceLaw`](@ref ClimateMachine.DGMethods.BalanceLaw), in this case, a `HeatModel`:

Base.@kwdef struct HeatModel{FT,Tα} <: BalanceLaw
    "Heat capacity"
    ρc::FT = 1
    "Thermal diffusivity"
    α::Tα = (T, ∇T) -> 0.000001*∇T/T
    "Thermal diffusivity"
    α_0::FT = 295.15*0.00001
    "Initial conditions for temperature"
    initialT::FT = 295.15
    "Bottom boundary value for temperature (Dirichlet boundary conditions)"
    T_bottom::FT = 300.0
    "Top flux (α(T)∇T) at top boundary (Neumann boundary conditions)"
    flux_top::FT = 0.0
end

# Create an instance of the `HeatModel`:
function compute_α(T, ∇T)
    return 0.000001*∇T/T
end

Tα = typeof(compute_α)
m = HeatModel{FT,Tα}(;α=compute_α);

# This model dictates the flow control, using [Dynamic Multiple Dispatch](https://en.wikipedia.org/wiki/Multiple_dispatch), for which kernels are executed.

# ## Define the variables

# All of the methods defined below, in this section, were `import`ed in [Loading code](@ref), and we must provide our own implementation as these methods are called inside `solve!` in the [Solve](@ref) section below. Each of these methods have a fall-back of "do nothing" if we don't, or fail to, implement these methods after `import`ing.

# Specify auxiliary variables for `HeatModel`
vars_state_auxiliary(::HeatModel, FT) = @vars(z::FT, T::FT);

# Specify state variables, the variables solved for in the PDEs, for `HeatModel`
vars_state_conservative(::HeatModel, FT) = @vars(ρcT::FT);

# Specify state variables whose gradients are needed for `HeatModel`
vars_state_gradient(::HeatModel, FT) = @vars(T::FT, ρcT::FT);

# Specify gradient variables for `HeatModel`
vars_state_gradient_flux(::HeatModel, FT) = @vars(α∇ρcT::SVector{3, FT}, ∇T::SVector{3, FT});

# ## Define the compute kernels

# Specify the initial values in `aux::Vars`, which are available in `init_state_conservative!`. Note that
# - this method is only called at `t=0`
# - `aux.z` and `aux.T` are available here because we've specified `z` and `T` in `vars_state_auxiliary`
function init_state_auxiliary!(m::HeatModel, aux::Vars, geom::LocalGeometry)
    aux.z = geom.coord[3]
    aux.T = m.initialT
end;

# Specify the initial values in `state::Vars`. Note that
# - this method is only called at `t=0`
# - `state.ρcT` is available here because we've specified `ρcT` in `vars_state_conservative`
function init_state_conservative!(
    m::HeatModel,
    state::Vars,
    aux::Vars,
    coords,
    t::Real,
)
    state.ρcT = m.ρc * aux.T
end;

# The remaining methods, defined in this section, are called at every time-step in the solver by the [`BalanceLaw`](@ref ClimateMachine.DGMethods.BalanceLaw) framework.

# Overload `update_auxiliary_state!` to call `heat_eq_nodal_update_aux!`, or any other auxiliary methods
function update_auxiliary_state!(
    dg::DGModel,
    m::HeatModel,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    nodal_update_auxiliary_state!(heat_eq_nodal_update_aux!, dg, m, Q, t, elems)
    return true # TODO: remove return true
end;

# Compute/update all auxiliary variables at each node. Note that
# - `aux.T` is available here because we've specified `T` in `vars_state_auxiliary`
function heat_eq_nodal_update_aux!(
    m::HeatModel,
    state::Vars,
    aux::Vars,
    t::Real,
)
    aux.T = state.ρcT / m.ρc
end;

# Since we have second-order fluxes, we must tell `ClimateMachine` to compute the gradient of `T`. Here, we specify how `T` is computed. Note that
#  - `transform.T` is available here because we've specified `T` in `vars_state_gradient`
function compute_gradient_argument!(
    m::HeatModel,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    transform.ρcT = state.ρcT
    transform.T = aux.T
end;

# Specify where in `diffusive::Vars` to store the computed gradient from `compute_gradient_argument!`. Note that:
#  - `diffusive.α∇T` is available here because we've specified `α∇T` in `vars_state_gradient_flux`
#  - `∇transform.T` is available here because we've specified `T`  in `vars_state_gradient`
function compute_gradient_flux!(
    m::HeatModel,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    α_val = m.α(aux.T, ∇transform.T)
    diffusive.α∇ρcT = hypot(α_val...) * ∇transform.ρcT'
    diffusive.∇T = ∇transform.T
end;

# We do no have sources, nor non-diffusive fluxes.
function source!(
    m::HeatModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    # α_val = m.α(aux.T, ∇transform.T)
    # diffusive.α∇ρcT = hypot(α_val...) * ∇transform.ρcT'
    # source.ρcT += m.α(aux.T, diffusive.∇T) * diffusive.∇T
end

function flux_first_order!(
    m::HeatModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
) end;

# \frac{∂ ρcT}{∂ t} + ∇ ⋅ (-α(T,∇T) ∇ρcT) = S(α(T,∇T), ∇T)

# Compute diffusive flux (``F(T,t) = -α ∇T`` in the original PDE). Note that:
# - `diffusive.α∇T` is available here because we've specified `α∇T` in `vars_state_gradient_flux`
function flux_second_order!(
    m::HeatModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
)
    # flux.ρcT -= m.α(state.T, diffusive.∇T) * diffusive.ρcT
    flux.ρcT -= diffusive.α∇ρcT
end;

# ### Boundary conditions

# Second-order terms in our equations, ``∇⋅(G)`` where ``G = α∇T``, are internally reformulated to first-order unknowns.
# Boundary conditions must be specified for all unknowns, both first-order and second-order unknowns which have been reformulated.

# The boundary conditions for `ρcT` (first order unknown)
function boundary_state!(
    nf,
    m::HeatModel,
    state⁺::Vars,
    aux⁺::Vars,
    n⁻,
    state⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
    _...,
)
    if bctype == 1 # bottom
        state⁺.ρcT = m.ρc * m.T_bottom
    elseif bctype == 2 # top
        nothing
    end
end;

# The boundary conditions for `ρcT` are specified here for second-order unknowns
function boundary_state!(
    nf,
    m::HeatModel,
    state⁺::Vars,
    diff⁺::Vars,
    aux⁺::Vars,
    n⁻,
    state⁻::Vars,
    diff⁻::Vars,
    aux⁻::Vars,
    bctype,
    t,
    _...,
)
    if bctype == 1 # bottom
        state⁺.ρcT = m.ρc * m.T_bottom
    elseif bctype == 2 # top
        diff⁺.α∇ρcT = -n⁻ * m.flux_top
    end
end;

# # Spatial discretization

# Prescribe vector of vertical elements (in meters)
velems = collect(0:10) / 10;

# Prescribe polynomial order of basis functions in finite elements
N_poly = 5;

# Define a grid
grid = SingleStackGrid(MPI, velems, N_poly, FT, Array);

# Configure the DG model, based on the PDEs, grid and penalty terms
dg = DGModel(
    m,
    grid,
    CentralNumericalFluxFirstOrder(),
    CentralNumericalFluxSecondOrder(),
    CentralNumericalFluxGradient(),
);

# # Time discretization

# We'll define the time-step based on the [Fourier number](https://en.wikipedia.org/wiki/Fourier_number)
Δ = min_node_distance(grid)

given_Fourier = FT(0.08);
Fourier_bound = given_Fourier * Δ^2 / m.α_0;
dt = Fourier_bound

# ## Initialize the state vector
# This initializes the state vector and allocates memory for the solution in space (`dg` has the model `m`, which describes the PDEs as well as the function used for initialization):
Q = init_ode_state(dg, FT(0));

# ## Initialize the ODE solver
# Here, we use an explicit Low-Storage [Runge-Kutta](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods) method
lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0);

# ## Inspect the initial conditions

# Let's export a plot of the initial state
output_dir = @__DIR__;

mkpath(output_dir);

state_vars = get_vars_from_stack(grid, Q, m, vars_state_conservative);
aux_vars =
    get_vars_from_stack(grid, dg.state_auxiliary, m, vars_state_auxiliary);
all_vars = OrderedDict(state_vars..., aux_vars...);
export_plot_snapshot(
    all_vars,
    ("ρcT",),
    joinpath(output_dir, "initial_condition.png"),
);
# ![](initial_condition.png)

# It matches what we have in `init_state_conservative!(m::HeatModel, ...)`, so let's continue.

# # Solver hooks / callbacks

# Define simulation time (all units are SI) and number of outputs over this interval
const timeend = 40;
const n_outputs = 5;

# This equates to exports every ceil(Int, timeend/n_outputs) time-step:
const every_x_simulation_time = ceil(Int, timeend / n_outputs);

# Create a dictionary for `z` coordinate (and convert to cm) NCDatasets IO:
z_scale = 100 # convert from meters to cm
dims = OrderedDict("z" => collect(get_z(grid, z_scale)));

# Create a DataFile, which is callable to get the name of each file given a step
output_data = DataFile(joinpath(output_dir, "output_data"));

# ClimateMachine's time-steppers provide hooks, or callbacks, which allow users to inject code to be executed at specified intervals. In this callback, the state and aux variables are collected, combined into a single `OrderedDict` and written to a NetCDF file (for each output step `step`).
step = [0];
callback = GenericCallbacks.EveryXSimulationTime(
    every_x_simulation_time,
    lsrk,
) do (init = false)
    state_vars = get_vars_from_stack(grid, Q, m, vars_state_conservative)
    aux_vars = get_vars_from_stack(
        grid,
        dg.state_auxiliary,
        m,
        vars_state_auxiliary;
        exclude = ["z"],
    )
    all_vars = OrderedDict(state_vars..., aux_vars...)
    write_data(
        NetCDFWriter(),
        output_data(step[1]),
        dims,
        all_vars,
        gettime(lsrk),
    )
    step[1] += 1
    nothing
end;

# # Solve

# This is the main "solve" method call. While users do not have access to the time-stepping loop, code and function calls may be injected via `callbacks` keyword argument, which is a `Tuple` of [`GenericCallbacks`](@ref).
solve!(Q, lsrk; timeend = timeend, callbacks = (callback,));

# # Post-processing

# Our solution has now been calculated and exported to NetCDF files in `output_dir`. Let's collect them all into a dictionary whose keys are output interval and values of NCDataset's:

all_data = collect_data(output_data, step[1]);

# To get `T` at ``t=0``, we can use `T_at_t_0 = all_data[0]["T"][:]`

# Let's plot the solution:

export_plot(all_data, ("ρcT",), joinpath(output_dir, "solution_vs_time.png"));
# ![](solution_vs_time.png)

# The results look as we would expect: a fixed temperature at the bottom is resulting in heat flux that propagates up the domain. To run this file, and inspect the solution in `all_data`, include this tutorial in the Julia REPL with:

# ```julia
# include(joinpath("tutorials", "Land", "Heat", "heat_equation.jl"))
# ```
