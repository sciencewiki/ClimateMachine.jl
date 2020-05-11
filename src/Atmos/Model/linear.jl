using CLIMAParameters.Planet: R_d, cv_d, T_0, e_int_v0, e_int_i0

"""
    linearized_air_pressure(ρ, ρe_tot, ρe_pot, ρq_tot=0, ρq_liq=0, ρq_ice=0)

The air pressure, linearized around a dry rest state, from the equation of state
(ideal gas law) where:

 - `ρ` (moist-)air density
 - `ρe_tot` total energy density
 - `ρe_pot` potential energy density
and, optionally,
 - `ρq_tot` total water density
 - `ρq_liq` liquid water density
 - `ρq_ice` ice density
"""
function linearized_air_pressure(
    param_set::AbstractParameterSet,
    ρ::FT,
    ρe_tot::FT,
    ρe_pot::FT,
    ρq_tot::FT = FT(0),
    ρq_liq::FT = FT(0),
    ρq_ice::FT = FT(0),
) where {FT <: Real, PS}
    _R_d::FT = R_d(param_set)
    _cv_d::FT = cv_d(param_set)
    _T_0::FT = T_0(param_set)
    _e_int_v0::FT = e_int_v0(param_set)
    _e_int_i0::FT = e_int_i0(param_set)
    return ρ * _R_d * _T_0 +
           _R_d / _cv_d * (
        ρe_tot - ρe_pot - (ρq_tot - ρq_liq) * _e_int_v0 +
        ρq_ice * (_e_int_i0 + _e_int_v0)
    )
end

@inline function linearized_pressure(
    ::DryModel,
    param_set::AbstractParameterSet,
    orientation::Orientation,
    state::Vars,
    aux::Vars,
)
    ρe_pot = state.ρ * gravitational_potential(orientation, aux)
    return linearized_air_pressure(param_set, state.ρ, state.ρe, ρe_pot)
end
@inline function linearized_pressure(
    ::EquilMoist,
    param_set::AbstractParameterSet,
    orientation::Orientation,
    state::Vars,
    aux::Vars,
)
    ρe_pot = state.ρ * gravitational_potential(orientation, aux)
    linearized_air_pressure(
        param_set,
        state.ρ,
        state.ρe,
        ρe_pot,
        state.moisture.ρq_tot,
    )
end

abstract type AtmosLinearModel <: BalanceLaw end

vars_state_conservative(lm::AtmosLinearModel, FT) =
    vars_state_conservative(lm.atmos, FT)
vars_state_gradient(lm::AtmosLinearModel, FT) = @vars()
vars_state_gradient_flux(lm::AtmosLinearModel, FT) = @vars()
vars_state_auxiliary(lm::AtmosLinearModel, FT) =
    vars_state_auxiliary(lm.atmos, FT)
vars_integrals(lm::AtmosLinearModel, FT) = @vars()
vars_reverse_integrals(lm::AtmosLinearModel, FT) = @vars()


function update_auxiliary_state!(
    dg::DGModel,
    lm::AtmosLinearModel,
    Q::MPIStateArray,
    t::Real,
    elems::UnitRange,
)
    return false
end
function flux_second_order!(
    lm::AtmosLinearModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    hyperdiffusive::Vars,
    aux::Vars,
    t::Real,
)
    nothing
end
integral_load_auxiliary_state!(
    lm::AtmosLinearModel,
    integ::Vars,
    state::Vars,
    aux::Vars,
) = nothing
integral_set_auxiliary_state!(lm::AtmosLinearModel, aux::Vars, integ::Vars) =
    nothing
reverse_integral_load_auxiliary_state!(
    lm::AtmosLinearModel,
    integ::Vars,
    state::Vars,
    aux::Vars,
) = nothing
reverse_integral_set_auxiliary_state!(
    lm::AtmosLinearModel,
    aux::Vars,
    integ::Vars,
) = nothing
flux_second_order!(
    lm::AtmosLinearModel,
    flux::Grad,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
) = nothing
function wavespeed(lm::AtmosLinearModel, nM, state::Vars, aux::Vars, t::Real)
    ref = aux.ref_state
    return soundspeed_air(lm.atmos.param_set, ref.T)
end

function boundary_state!(
    nf::NumericalFluxFirstOrder,
    atmoslm::AtmosLinearModel,
    args...,
)
    atmos_boundary_state!(nf, AtmosBC(), atmoslm, args...)
end
function boundary_state!(
    nf::NumericalFluxSecondOrder,
    atmoslm::AtmosLinearModel,
    args...,
)
    nothing
end
init_state_auxiliary!(lm::AtmosLinearModel, aux::Vars, geom::LocalGeometry) =
    nothing
init_state_conservative!(
    lm::AtmosLinearModel,
    state::Vars,
    aux::Vars,
    coords,
    t,
) = nothing


struct AtmosAcousticLinearModel{M} <: AtmosLinearModel
    atmos::M
    function AtmosAcousticLinearModel(atmos::M) where {M}
        if atmos.ref_state === NoReferenceState()
            error("AtmosAcousticLinearModel needs a model with a reference state")
        end
        new{M}(atmos)
    end
end

function flux_first_order!(
    lm::AtmosAcousticLinearModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    FT = eltype(state)
    ref = aux.ref_state
    e_pot = gravitational_potential(lm.atmos.orientation, aux)

    flux.ρ = state.ρu
    pL = linearized_pressure(
        lm.atmos.moisture,
        lm.atmos.param_set,
        lm.atmos.orientation,
        state,
        aux,
    )
    flux.ρu += pL * I
    flux.ρe = ((ref.ρe + ref.p) / ref.ρ - e_pot) * state.ρu
    nothing
end
source!(::AtmosAcousticLinearModel, _...) = nothing

struct AtmosAcousticGravityLinearModel{M} <: AtmosLinearModel
    atmos::M
    function AtmosAcousticGravityLinearModel(atmos::M) where {M}
        if atmos.ref_state === NoReferenceState()
            error("AtmosAcousticGravityLinearModel needs a model with a reference state")
        end
        new{M}(atmos)
    end
end
function flux_first_order!(
    lm::AtmosAcousticGravityLinearModel,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    FT = eltype(state)
    ref = aux.ref_state
    e_pot = gravitational_potential(lm.atmos.orientation, aux)

    flux.ρ = state.ρu
    pL = linearized_pressure(
        lm.atmos.moisture,
        lm.atmos.param_set,
        lm.atmos.orientation,
        state,
        aux,
    )
    flux.ρu += pL * I
    flux.ρe = ((ref.ρe + ref.p) / ref.ρ) * state.ρu
    nothing
end
function source!(
    lm::AtmosAcousticGravityLinearModel,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    if direction isa VerticalDirection || direction isa EveryDirection
        ∇Φ = ∇gravitational_potential(lm.atmos.orientation, aux)
        source.ρu -= state.ρ * ∇Φ
    end
    nothing
end



struct AtmosAcousticLinearModelMomentum{M} <: AtmosLinearModel
  atmos::M
  function AtmosAcousticLinearModelMomentum(atmos::M) where {M}
    if atmos.ref_state === NoReferenceState()
      error("AtmosAcousticLinearModelMomentum needs a model with a reference state")
    end
    new{M}(atmos)
  end
end

function flux_first_order!(
    lm::AtmosAcousticLinearModelMomentum,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    FT = eltype(state)
    ref = aux.ref_state

    pL = linearized_pressure(
        lm.atmos.moisture,
        lm.atmos.param_set,
        lm.atmos.orientation,
        state,
        aux,
    )
    flux.ρu += pL * I
    nothing
end
source!(::AtmosAcousticLinearModelMomentum, _...) = nothing

struct AtmosAcousticLinearModelThermo{M} <: AtmosLinearModel
  atmos::M
  function AtmosAcousticLinearModelThermo(atmos::M) where {M}
    if atmos.ref_state === NoReferenceState()
      error("AtmosAcousticLinearModelThermo needs a model with a reference state")
    end
    new{M}(atmos)
  end
end

function flux_first_order!(
    lm::AtmosAcousticLinearModelThermo,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    FT = eltype(state)
    ref = aux.ref_state
    e_pot = gravitational_potential(lm.atmos.orientation, aux)

    flux.ρ = state.ρu
    flux.ρe = ((ref.ρe + ref.p) / ref.ρ - e_pot) * state.ρu
    nothing
end
source!(::AtmosAcousticLinearModelThermo, _...) = nothing



struct AtmosAcousticGravityLinearModelMomentum{M} <: AtmosLinearModel
  atmos::M
  function AtmosAcousticGravityLinearModelMomentum(atmos::M) where {M}
    if atmos.ref_state === NoReferenceState()
      error("AtmosAcousticGravityLinearModelMomentum needs a model with a reference state")
    end
    new{M}(atmos)
  end
end
function flux_first_order!(
    lm::AtmosAcousticGravityLinearModelMomentum,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    FT = eltype(state)
    ref = aux.ref_state

    pL = linearized_pressure(
        lm.atmos.moisture,
        lm.atmos.param_set,
        lm.atmos.orientation,
        state,
        aux,
    )
    flux.ρu += pL * I
    nothing
end
function source!(
    lm::AtmosAcousticGravityLinearModelMomentum,
    source::Vars,
    state::Vars,
    diffusive::Vars,
    aux::Vars,
    t::Real,
    direction,
)
    if direction isa VerticalDirection || direction isa EveryDirection
        ∇Φ = ∇gravitational_potential(lm.atmos.orientation, aux)
        source.ρu -= state.ρ * ∇Φ
    end
    nothing
end

struct AtmosAcousticGravityLinearModelThermo{M} <: AtmosLinearModel
  atmos::M
  function AtmosAcousticGravityLinearModelThermo(atmos::M) where {M}
    if atmos.ref_state === NoReferenceState()
      error("AtmosAcousticGravityLinearModelThermo needs a model with a reference state")
    end
    new{M}(atmos)
  end
end
function flux_first_order!(
    lm::AtmosAcousticGravityLinearModelThermo,
    flux::Grad,
    state::Vars,
    aux::Vars,
    t::Real,
)
    FT = eltype(state)
    ref = aux.ref_state
    e_pot = gravitational_potential(lm.atmos.orientation, aux)

    flux.ρ = state.ρu
    flux.ρe = ((ref.ρe + ref.p) / ref.ρ) * state.ρu
    nothing
end
source!(::AtmosAcousticGravityLinearModelThermo, _...) = nothing



abstract type AtmosLinearModelSplit <: AtmosLinearModel end

struct AtmosAcousticLinearModelSplit{M} <: AtmosLinearModelSplit
  linear::AtmosAcousticLinearModel{M}
  momentum::AtmosAcousticLinearModelMomentum{M}
  thermo::AtmosAcousticLinearModelThermo{M}
  function AtmosAcousticLinearModelSplit(atmos::M) where {M}
    if atmos.ref_state === NoReferenceState()
      error("AtmosAcousticLinearModelSplit needs a model with a reference state")
    end
    new{M}(
        AtmosAcousticLinearModel(atmos),
        AtmosAcousticLinearModelMomentum(atmos),
        AtmosAcousticLinearModelThermo(atmos),
    )
  end
end

struct AtmosAcousticGravityLinearModelSplit{M} <: AtmosLinearModelSplit
  linear::AtmosAcousticGravityLinearModel{M}
  momentum::AtmosAcousticGravityLinearModelMomentum{M}
  thermo::AtmosAcousticGravityLinearModelThermo{M}
  function AtmosAcousticGravityLinearModelSplit(atmos::M) where {M}
    if atmos.ref_state === NoReferenceState()
      error("AtmosAcousticGravityLinearModelSplit needs a model with a reference state")
    end
    new{M}(
        AtmosAcousticGravityLinearModel(atmos),
        AtmosAcousticGravityLinearModelMomentum(atmos),
        AtmosAcousticGravityLinearModelThermo(atmos),
    )
  end
end
