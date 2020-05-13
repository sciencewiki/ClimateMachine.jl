# # Single stack test

# Equations solved:

# ``
# Balance law form:
# \frac{∂ F}{∂ t} + ∇ ⋅ (F) = S
# \frac{∂ F}{∂ t} = - ∇ ⋅ (F) + S

# \frac{∂ ρ}{∂ t} = - ∇ ⋅ (ρu)
# \frac{∂ ρu}{∂ t} = - ∇ ⋅ (-μ ∇u) - ∇ ⋅ (ρu u')
# \frac{∂ ρcT}{∂ t} = - ∇ ⋅ (-α ∇ρcT) - ∇ ⋅ (u ρcT)

# z_min: ρ = 1
# z_min: ρu = 0
# z_min: ρcT = T=T_fixed

# z_max: ρ = 1
# z_max: ρu = 0
# z_max: ρcT = no flux


# ``

# where
#  - `t` is time
#  - `α` is the thermal diffusivity
#  - `μ` is the dynamic viscosity
#  - `T` is the temperature
#  - `ρ` is the density
#  - `c` is the heat capacity
#  - `ρcT` is the thermal energy

# To put this in the form of ClimateMachine's [`BalanceLaw`](@ref
# ClimateMachine.DGMethods.BalanceLaw), we'll re-write the equation as:

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
using Distributions
using NCDatasets
using OrderedCollections
using Plots
using StaticArrays

#  - load CLIMAParameters and set up to use it:

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

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

#  - import necessary ClimateMachine modules: (`import`ing enables us to
#  provide implementations of these structs/methods)
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
# Load some helper functions (soon to be incorporated into
# `ClimateMachine/src`)
include(joinpath(clima_dir, "tutorials", "Land", "helper_funcs.jl"));
include(joinpath(clima_dir, "tutorials", "Land", "plotting_funcs.jl"));

# # Define the set of Partial Differential Equations (PDEs)

# ## Define the model

# Model parameters can be stored in the particular [`BalanceLaw`](@ref
# ClimateMachine.DGMethods.BalanceLaw), in this case, a `SingleStack`:

Base.@kwdef struct SingleStack{FT} <: BalanceLaw
    "Parameters"
    param_set::AbstractParameterSet = param_set
    "Heat capacity"
    c::FT = 1
    "Dynamic viscosity"
    μ::FT = 0.001
    "Thermal diffusivity"
    α::FT = 0.01
    "IC variance"
    σ::FT = 1e-4
    "Domain height"
    zmax::FT = 1
    "Initial conditions for temperature"
    initialT::FT = 295.15
    "Bottom boundary value for temperature (Dirichlet boundary conditions)"
    T_bottom::FT = 300.0
    "Top flux (α∇ρcT) at top boundary (Neumann boundary conditions)"
    flux_top::FT = 0.0
end

# Create an instance of the `SingleStack`:
m = SingleStack{FT}();

# This model dictates the flow control, using [Dynamic Multiple
# Dispatch](https://en.wikipedia.org/wiki/Multiple_dispatch), for which
# kernels are executed.

# ## Define the variables

# All of the methods defined in this section were `import`ed in # [Loading
# code](@ref) to let us provide implementations for our `SingleStack` as they
# will be used by the solver.

# Specify auxiliary variables for `SingleStack`
vars_state_auxiliary(::SingleStack, FT) = @vars(z::FT, T::FT);

# Specify state variables, the variables solved for in the PDEs, for
# `SingleStack`
vars_state_conservative(::SingleStack, FT) = @vars(ρ::FT, ρu::SVector{3, FT}, ρcT::FT);

# Specify state variables whose gradients are needed for `SingleStack`
vars_state_gradient(::SingleStack, FT) = @vars(u::SVector{3, FT}, ρcT::FT);

# Specify gradient variables for `SingleStack`
vars_state_gradient_flux(::SingleStack, FT) = @vars(μ∇u::SMatrix{3, 3, FT, 9}, α∇ρcT::SVector{3, FT});

# ## Define the compute kernels

# Specify the initial values in `aux::Vars`, which are available in
# `init_state_conservative!`. Note that
# - this method is only called at `t=0`
# - `aux.z` and `aux.T` are available here because we've specified `z` and `T`
# in `vars_state_auxiliary`
function init_state_auxiliary!(m::SingleStack, aux::Vars, geom::LocalGeometry)
    aux.z = geom.coord[3]
    aux.T = m.initialT
end;

# Specify the initial values in `state::Vars`. Note that
# - this method is only called at `t=0`
# - `state.ρcT` is available here because we've specified `ρcT` in
# `vars_state_conservative`
function init_state_conservative!(
    m::SingleStack,
    state::Vars,
    aux::Vars,
    coords,
    t::Real,
)
    z = aux.z
    ε1 = rand(Normal(0, m.σ))
    ε2 = rand(Normal(0, m.σ))
    state.ρ = 1
    ρu = 1 - 4*(z - m.zmax/2)^2 + ε1
    ρv = 1 - 4*(z - m.zmax/2)^2 + ε2
    ρw = 0
    state.ρu = SVector(ρu,ρv,ρw)

    state.ρcT = state.ρ*m.c * aux.T
end;

# The remaining methods, defined in this section, are called at every
# time-step in the solver by the [`BalanceLaw`](@ref
# ClimateMachine.DGMethods.BalanceLaw) framework.

# Overload `update_auxiliary_state!` to call `heat_eq_nodal_update_aux!`, or
# any other auxiliary methods
function update_auxiliary_state!(
    dg::DGModel,
    m::SingleStack,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    nodal_update_auxiliary_state!(heat_eq_nodal_update_aux!, dg, m, Q, t, elems)
    return true # TODO: remove return true
end;

# Compute/update all auxiliary variables at each node. Note that
# - `aux.T` is available here because we've specified `T` in
# `vars_state_auxiliary`
function heat_eq_nodal_update_aux!(
    m::SingleStack,
    state::Vars,
    aux::Vars,
    t::Real,
)
    aux.T = state.ρcT / (state.ρ*m.c)
end;

# Since we have second-order fluxes, we must tell `ClimateMachine` to compute
# the gradient of `ρcT`. Here, we specify how `ρcT` is computed. Note that
#  - `transform.ρcT` is available here because we've specified `ρcT` in
#  `vars_state_gradient`
function compute_gradient_argument!(
    m::SingleStack,
    transform::Vars,
    state::Vars,
    aux::Vars,
    t::Real,
)
    transform.ρcT = state.ρcT
    transform.u = state.ρu/state.ρ
end;

# Specify where in `diffusive::Vars` to store the computed gradient from
# `compute_gradient_argument!`. Note that:
#  - `diffusive.α∇ρcT` is available here because we've specified `α∇ρcT` in
#  `vars_state_gradient_flux`
#  - `∇transform.ρcT` is available here because we've specified `ρcT`  in
#  `vars_state_gradient`
function compute_gradient_flux!(
    m::SingleStack,
    diffusive::Vars,
    ∇transform::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    diffusive.α∇ρcT = m.α * ∇transform.ρcT
    diffusive.μ∇u = m.μ * ∇transform.u
end;

# We have no sources, nor non-diffusive fluxes.
function source!(m::SingleStack, _...) end;
function flux_first_order!(
    m::SingleStack,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    flux.ρ = state.ρu

    u = state.ρu / state.ρ
    flux.ρu = state.ρu * u'
    flux.ρcT = u * state.ρcT
end;

# Compute diffusive flux (``F(α, ρcT, t) = -α ∇ρcT`` in the original PDE).
# Note that:
# - `diffusive.α∇ρcT` is available here because we've specified `α∇ρcT` in
# `vars_state_gradient_flux`
function flux_second_order!(
    m::SingleStack,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
)
    flux.ρcT -= diffusive.α∇ρcT
    flux.ρu -= diffusive.μ∇u
end;

# ### Boundary conditions

# Second-order terms in our equations, ``∇⋅(G)`` where ``G = α∇ρcT``, are
# internally reformulated to first-order unknowns.
# Boundary conditions must be specified for all unknowns, both first-order and
# second-order unknowns which have been reformulated.

# The boundary conditions for `ρcT` (first order unknown)
function boundary_state!(
    nf,
    m::SingleStack,
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
        state⁺.ρ = 1
        state⁺.ρu = SVector(0,0,0)
        state⁺.ρcT = state⁺.ρ*m.c * m.T_bottom
    elseif bctype == 2 # top
        state⁺.ρ = 1
        state⁺.ρu = SVector(0,0,0)
    end
end;

# The boundary conditions for `ρcT` are specified here for second-order
# unknowns
function boundary_state!(
    nf,
    m::SingleStack,
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
        state⁺.ρ = 1
        state⁺.ρu = SVector(0,0,0)
        state⁺.ρcT = state⁺.ρ*m.c * m.T_bottom
    elseif bctype == 2 # top
        state⁺.ρ = 1
        state⁺.ρu = SVector(0,0,0)
        diff⁺.α∇ρcT = -n⁻ * m.flux_top
    end
end;

# # Spatial discretization

# Prescribe polynomial order of basis functions in finite elements
N_poly = 5;

# Specify the number of vertical elements
nelem_vert = 20;

# Specify the domain height
zmax = m.zmax;

# Establish a `ClimateMachine` single stack configuration
driver_config = ClimateMachine.SingleStackConfiguration(
    "SingleStack",
    N_poly,
    nelem_vert,
    zmax,
    param_set,
    m,
    numerical_flux_first_order = CentralNumericalFluxFirstOrder(),
);

# # Time discretization

# Specify simulation time (SI units)
t0 = FT(0)
timeend = FT(10)

# We'll define the time-step based on the [Fourier
# number](https://en.wikipedia.org/wiki/Fourier_number)
Δ = min_node_distance(driver_config.grid)

given_Fourier = FT(0.08);
Fourier_bound = given_Fourier * Δ^2 / m.α;
dt = Fourier_bound

# # Configure a `ClimateMachine` solver.

# This initializes the state vector and allocates memory for the solution in
# space (`dg` has the model `m`, which describes the PDEs as well as the
# function used for initialization). This additionally initializes the ODE
# solver, by default an explicit Low-Storage
# [Runge-Kutta](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)
# method.

solver_config =
    ClimateMachine.SolverConfiguration(t0, timeend, driver_config, ode_dt = dt);

# ## Inspect the initial conditions

# Let's export a plot of the initial state
output_dir = @__DIR__;

mkpath(output_dir);

z_scale = 100 # convert from meters to cm
z_key = "z"
z_label = "z [cm]"
z = get_z(driver_config.grid, z_scale)
state_vars = get_vars_from_stack(
    driver_config.grid,
    solver_config.Q,
    m,
    vars_state_conservative,
);
aux_vars = get_vars_from_stack(
    driver_config.grid,
    solver_config.dg.state_auxiliary,
    m,
    vars_state_auxiliary;
    exclude = [z_key]
);
all_vars = OrderedDict(state_vars..., aux_vars...);
# all_vars = prep_for_io(z_label, all_vars)
export_plot_snapshot(
    z,
    all_vars,
    ("ρcT",),
    joinpath(output_dir, "initial_condition.png"),
    z_label,
);
# ![](initial_condition.png)

# It matches what we have in `init_state_conservative!(m::SingleStack, ...)`, so
# let's continue.

# # Solver hooks / callbacks

# Define the number of outputs from `t0` to `timeend`
const n_outputs = 5;

# This equates to exports every ceil(Int, timeend/n_outputs) time-step:
const every_x_simulation_time = ceil(Int, timeend / n_outputs);

# Create a dictionary for `z` coordinate (and convert to cm) NCDatasets IO:
dims = OrderedDict(z_key => collect(z));

# Create a DataFile, which is callable to get the name of each file given a step
output_data = DataFile(joinpath(output_dir, "output_data"));

all_data = Dict([k => Dict() for k in 0:n_outputs]...)
all_data[0] = deepcopy(all_vars)

# The `ClimateMachine`'s time-steppers provide hooks, or callbacks, which
# allow users to inject code to be executed at specified intervals. In this
# callback, the state and aux variables are collected, combined into a single
# `OrderedDict` and written to a NetCDF file (for each output step `step`).
step = [0];
callback = GenericCallbacks.EveryXSimulationTime(
    every_x_simulation_time,
    solver_config.solver,
) do (init = false)
    state_vars = get_vars_from_stack(
        driver_config.grid,
        solver_config.Q,
        m,
        vars_state_conservative,
    )
    aux_vars = get_vars_from_stack(
        driver_config.grid,
        solver_config.dg.state_auxiliary,
        m,
        vars_state_auxiliary;
        exclude = [z_key],
    )
    all_vars = OrderedDict(state_vars..., aux_vars...)
    step[1] += 1
    all_data[step[1]] = deepcopy(all_vars)
    nothing
end;

# # Solve

# This is the main `ClimateMachine` solver invocation. While users do not have
# access to the time-stepping loop, code may be injected via `user_callbacks`,
# which is a `Tuple` of [`GenericCallbacks`](@ref).
ClimateMachine.invoke!(solver_config; user_callbacks = (callback,))

# # Post-processing

# Our solution has now been calculated and exported to NetCDF files in
# `output_dir`. Let's collect them all into a nested dictionary whose keys are
# the output interval. The next level keys are the variable names, and the
# values are the values along the grid:

# all_data = collect_data(output_data, step[1]);

# To get `T` at ``t=0``, we can use `T_at_t_0 = all_data[0]["T"][:]`
# @show keys(all_data[0])

# Let's plot the solution:

export_plot(
    z,
    all_data,
    ("ρu[1]","ρu[2]",),
    joinpath(output_dir, "solution_vs_time.png"),
    z_label,
);
# ![](solution_vs_time.png)

# The results look as we would expect: a fixed temperature at the bottom is
# resulting in heat flux that propagates up the domain. To run this file, and
# inspect the solution in `all_data`, include this tutorial in the Julia REPL
# with:

# ```julia
# include(joinpath("tutorials", "Land", "Heat", "heat_equation.jl"))
# ```
