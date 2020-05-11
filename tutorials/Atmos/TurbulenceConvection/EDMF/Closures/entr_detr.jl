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



include(joinpath("Models","entr_detr.jl"))
include(joinpath("Models","buoyancy.jl"))
include(joinpath("Models","mixing_length.jl"))
include(joinpath("Models","microphysics.jl"))
include(joinpath("Models","subdomain_statistics.jl"))
include(joinpath("Models","surface.jl"))
include(joinpath("Models","pressure.jl"))
include(joinpath("Models","eddy_diffusivity.jl"))


##### Entrainment-Detrainment models

abstract type EntrDetrModel end

struct MoistureDeficit{FT} <: EntrDetrModel
  entr_factor::FT
  detr_factor::FT
  detr_RH_power::FT
  sigmoid_slope_param::FT
  upd_mixing_frac::FT
  turb_entr_fac::FT
  entr_tke_fac::FT
end

function compute_entrainment_detrainment! end

function compute_entrainment_detrainment!(grid::Grid{FT}, UpdVar, tmp, q, params, model::MoistureDeficit) where FT
  gm, en, ud, sd, al = allcombinations(q)
  Δzi = grid.Δzi
  k_1 = first_interior(grid, Zmin())
  @inbounds for i in ud
    @inbounds for k in over_elems_real(grid)
      # get parameters
      @unpack params param_set
      c_ε = model.entr_factor
      c_turb = model.turb_entr_fac
      if tmp[:q_liq, k, en]+tmp[:q_liq, k, i]>0.0
        c_δ = model.detr_factor
      else
        c_δ = 0.0
      end
      β = model.detr_RH_power
      μ_0 = model.sigmoid_slope_param
      χ = model.upd_mixing_frac
      c_λ = model.entr_tke_fac

      # get subdomain properties
      b_up = tmp[:buoy, k, i]
      b_en = tmp[:buoy, k, en]
      w_up = max(q[:w, k, i],1e-4)
      w_en = q[:w, k, en]
      dw = max(w_up - w_en,1e-4)
      db = b_up - b_en
      sqrt_tke = sqrt(max(q[:tke, k, en],0.0))
      ts = ActiveThermoState(param_set, q, tmp, k, en)
      RH_en = relative_humidity(ts)
      ts = ActiveThermoState(param_set, q, tmp, k, i)
      RH_up = relative_humidity(ts)

      # compute aux functions
      D_ϵ = 1/(1+exp(-db/dw/μ_0*(χ - q[:a, k, i]/(q[:a, k, i]+q[:a, k, en]))))
      D_δ = 1/(1+exp( db/dw/μ_0*(χ - q[:a, k, i]/(q[:a, k, i]+q[:a, k, en]))))
      M_δ = ( max((RH_up^β-RH_en^β),0.0) )^(1/β)
      M_ϵ = ( max((RH_en^β-RH_up^β),0.0) )^(1/β)
      λ = min(abs(db/dw),c_λ*abs(db/(sqrt_tke+1e-8)))

      # compute entrainment/detrainmnet components
      tmp[:εt_model, k, i] = 2*q[:a, k, i]*c_turb*sqrt_tke/(w_up*q[:a, k, i]*UpdVar[i].cloud.updraft_top)
      tmp[:ε_model, k, i] = λ/w_up*(c_ε*D_ϵ + c_δ*M_ϵ)
      tmp[:δ_model, k, i] = λ/w_up*(c_ε*D_δ + c_δ*M_δ)
    end
    tmp[:εt_model, k_1, i] = FT(0)
    tmp[:ε_model, k_1, i] = 2 * Δzi
    tmp[:δ_model, k_1, i] = FT(0)
  end
end

function compute_cv_entr!(grid::Grid{FT}, q, tmp, tmp_O2, ϕ, ψ, cv, tke_factor) where FT
  gm, en, ud, sd, al = allcombinations(q)
  @inbounds for k in over_elems_real(grid)
    tmp_O2[cv][:entr_gain, k] = FT(0)
    @inbounds for i in ud
      Δϕ      = q[ϕ, k, i] - q[ϕ, k, en]
      Δψ      = q[ψ, k, i] - q[ψ, k, en]
      Δψ_star = q[ψ, k, i] - q[ψ, k, gm]
      Δϕ_star = q[ϕ, k, i] - q[ϕ, k, gm]
      tmp_O2[cv][:entr_gain, k] +=  (tke_factor*q[:a, k, i]*abs(q[:w, k, i]) *
          tmp[:δ_model, k, i]*Δϕ*Δψ + tmp[:εt_model, k, i]*(Δϕ_star*Δϕ + Δψ_star*Δψ))
    end
    tmp_O2[cv][:entr_gain, k] *= tmp[:ρ_0, k]
  end
end
