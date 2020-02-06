using StaticArrays
using LinearAlgebra
using CLIMA.VariableTemplates
import CLIMA.DGmethods: BalanceLaw,
                        vars_aux, vars_state, vars_gradient, vars_diffusive,
                        flux_nondiffusive!, flux_diffusive!, source!,
                        gradvariables!, diffusive!,
                        init_aux!, init_state!,
                        boundary_state!, wavespeed, LocalGeometry,
                        vars_gradient_laplacian, vars_hyperdiffusive,
                        hyperdiffusive!

abstract type HyperDiffusionProblem end
struct HyperDiffusion{dim, P} <: BalanceLaw
  problem::P
  function HyperDiffusion{dim}(problem::P) where {dim, P <: HyperDiffusionProblem}
    new{dim, P}(problem)
  end
end

vars_aux(::HyperDiffusion, FT) = @vars(k::SVector{3, FT})
#
# Density is only state
vars_state(::HyperDiffusion, FT) = @vars(ρ::FT)

# Take the gradient of density
vars_gradient(::HyperDiffusion, FT) = @vars(ρ::FT)
# Take the gradient of laplacian of density
vars_gradient_laplacian(::HyperDiffusion, FT) = @vars(ρ::FT)

vars_diffusive(::HyperDiffusion, FT) = @vars(ω::SVector{3,FT})
# The DG auxiliary variable: D ∇ Δρ
vars_hyperdiffusive(::HyperDiffusion, FT) = @vars(σ::SVector{3,FT})

function flux_nondiffusive!(m::HyperDiffusion, flux::Grad, state::Vars,
                            aux::Vars, t::Real)
  u = aux.k
  flux.ρ += u * state.ρ
end

"""
    flux_diffusive!(m::HyperDiffusion, flux::Grad, state::Vars,
                     auxDG::Vars, aux::Vars, t::Real)

Computes diffusive flux `F` in:

```
∂ρ
-- = - ∇ • (σ)
∂t
```
Where

 - `σ` is DG auxiliary variable (`σ = D ∇ Δρ` with D being the hyperdiffusion tensor)
"""
function flux_diffusive!(m::HyperDiffusion, flux::Grad, state::Vars,
                         auxDG::Vars, auxHDG::Vars, aux::Vars, t::Real)
  σ = auxHDG.σ
  ω = auxDG.ω
 
  k = aux.k
  Pk = I - k * k'
  flux.ρ += σ
end

"""
    gradvariables!(m::HyperDiffusion, transform::Vars, state::Vars,
                   aux::Vars, t::Real)

Set the variable to take the gradient of (`ρ` in this case)
"""
function gradvariables!(m::HyperDiffusion, transform::Vars, state::Vars,
                        aux::Vars, t::Real)
  transform.ρ = state.ρ
end

function diffusive!(m::HyperDiffusion, auxDG::Vars, gradvars::Grad,
                    state::Vars, aux::Vars, t::Real)
  ∇ρ = gradvars.ρ
  auxDG.ω = ∇ρ / 10
end
function hyperdiffusive!(m::HyperDiffusion, auxHDG::Vars, gradvars::Grad,
                         state::Vars, aux::Vars, t::Real)
  ∇Δρ = gradvars.ρ
  auxHDG.σ = ∇Δρ / 100
end

"""
    source!(m::HyperDiffusion, _...)

There is no source in the advection-diffusion model
"""
source!(m::HyperDiffusion, _...) = nothing

"""
    init_aux!(m::HyperDiffusion, aux::Vars, geom::LocalGeometry)

initialize the auxiliary state
"""
function init_aux!(m::HyperDiffusion, aux::Vars, geom::LocalGeometry)
  aux.k = geom.coord / norm(geom.coord)
end

function init_state!(m::HyperDiffusion, state::Vars, aux::Vars,
                     coords, t::Real)
  initial_condition!(m.problem, state, aux, coords, t)
end

boundary_state!(nf, ::HyperDiffusion, _...) = nothing