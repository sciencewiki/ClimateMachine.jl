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

import CLIMA.DGmethods: vars_aux,
                        vars_state,
                        vars_gradient,
                        vars_diffusive,
                        source!,
                        flux_diffusive!,
                        flux_nondiffusive!,
                        gradvariables!,
                        diffusive!,
                        update_aux!,
                        nodal_update_aux!,
                        init_aux!,
                        init_state!,
                        boundary_state!

FT = Float64;

CLIMA.init(; disable_gpu=true);

include(joinpath("..","helper_funcs.jl"))
include(joinpath("..","plotting_funcs.jl"))

Base.@kwdef struct HeatModel{FT} <: BalanceLaw
  "Heat capacity"
  ρc::FT = 1
  "Thermal diffusivity"
  α::FT = 0.01
  "Initial conditions for temperature"
  initialT::FT = 295.15
  "Surface boundary value for temperature (Dirichlet boundary conditions)"
  surfaceT::FT = 300.0
end

m = HeatModel{FT}();

vars_aux(::HeatModel, FT) = @vars(z::FT, T::FT);

vars_state(::HeatModel, FT) = @vars(ρcT::FT);

vars_gradient(::HeatModel, FT) = @vars(T::FT);

vars_diffusive(::HeatModel, FT) = @vars(∇T::SVector{3,FT});

function init_aux!(m::HeatModel, aux::Vars, geom::LocalGeometry)
  aux.z = geom.coord[3]
  aux.T = m.initialT
end;

function init_state!(m::HeatModel, state::Vars, aux::Vars, coords, t::Real)
  state.ρcT = m.ρc * aux.T
end;

function update_aux!(dg::DGModel, m::HeatModel, Q::MPIStateArray, t::Real, elems::UnitRange)
  nodal_update_aux!(soil_nodal_update_aux!, dg, m, Q, t, elems)
  return true # TODO: remove return true
end;

function soil_nodal_update_aux!(m::HeatModel, state::Vars, aux::Vars, t::Real)
  aux.T = state.ρcT / m.ρc
end;

function gradvariables!(m::HeatModel, transform::Vars, state::Vars, aux::Vars, t::Real)
  transform.T = state.ρcT / m.ρc
end;

function diffusive!(m::HeatModel, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real)
  diffusive.∇T = ∇transform.T
end;

function source!(m::HeatModel, _...); end;
function flux_nondiffusive!(m::HeatModel, _...); end;

function flux_diffusive!(m::HeatModel, flux::Grad, state::Vars, diffusive::Vars, hyperdiffusive::Vars, aux::Vars, t::Real)
   flux.ρcT -= m.α * diffusive.∇T
end;

function boundary_state!(nf, m::HeatModel, state⁺::Vars, aux⁺::Vars,
                         nM, state⁻::Vars, aux⁻::Vars, bctype, t, _...)
  if bctype == 1 # surface
    state⁺.ρcT = m.ρc * m.surfaceT
  elseif bctype == 2 # bottom
    nothing
  end
end;

function boundary_state!(nf, m::HeatModel, state⁺::Vars, diff⁺::Vars,
                         aux⁺::Vars, nM, state⁻::Vars, diff⁻::Vars, aux⁻::Vars,
                         bctype, t, _...)
  if bctype == 1 # surface
    state⁺.ρcT = m.ρc * m.surfaceT
  elseif bctype == 2 # bottom
    diff⁺.∇T = -diff⁻.∇T
  end
end;

velems = collect(0:10) / 10;

N_poly = 5;

topl = StackedBrickTopology(MPI.COMM_WORLD, (0.0:1,0.0:1,velems);
    periodicity = (true,true,false),
    boundary=((0,0),(0,0),(1,2)));

grid = DiscontinuousSpectralElementGrid(
  topl,
  FloatType = FT,
  DeviceArray = Array,
  polynomialorder = N_poly);

dg = DGModel(
  m,
  grid,
  CentralNumericalFluxNonDiffusive(), # penalty terms for discretizations
  CentralNumericalFluxDiffusive(),
  CentralNumericalFluxGradient())

Δ = min_node_distance(grid)

given_Fourier = 0.08;
Fourier_bound = given_Fourier*Δ^2 / m.α;
dt = Fourier_bound

Q = init_ode_state(dg, Float64(0))

lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0);

output_dir = pwd();
mkpath(output_dir)

state_vars = get_vars_from_stack(grid, Q, m, vars_state)
aux_vars = get_vars_from_stack(grid, dg.auxstate, m, vars_aux)
all_vars = OrderedDict(state_vars..., aux_vars...)
export_plot_snapshot(all_vars, ("ρcT",), joinpath(output_dir, "initial_condition.png"))

const timeend = 40;
const n_outputs = 5;

const every_x_simulation_time = ceil(Int, timeend/n_outputs);

z_scale = 100 # convert from meters to cm
dims = OrderedDict("z" => collect(get_z(grid, z_scale, N_poly)))

output_data = DataFile(joinpath(output_dir, "output_data"))

step = [0]
callback = GenericCallbacks.EveryXSimulationTime(every_x_simulation_time, lsrk) do (init = false)
  state_vars = get_vars_from_stack(grid, Q, m, vars_state; exclude=[])
  aux_vars = get_vars_from_stack(grid, dg.auxstate, m, vars_aux; exclude=["z"])
  all_vars = OrderedDict(state_vars..., aux_vars...)
  write_data(NetCDFWriter(), output_data(step[1]), dims, all_vars, gettime(lsrk))
  step[1]+=1
  nothing
end

solve!(Q, lsrk; timeend=timeend, callbacks=(callback,))

all_data = collect_data(output_data, step[1])

export_plot(all_data, ("ρcT",), joinpath(output_dir, "solution_vs_time.png"))

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

