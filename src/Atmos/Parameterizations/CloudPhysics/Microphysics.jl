"""
    one-moment bulk Microphysics scheme

Included processes
  - condensation/evaporation as relaxation to equilibrium
  - autoconversion
  - accretion
  - evaporation and sublimation
  - terminal velocity
"""
module Microphysics

using SpecialFunctions

using ClimateMachine.MoistThermodynamics

using CLIMAParameters
using CLIMAParameters.Planet: ρ_cloud_liq, R_v, grav
using CLIMAParameters.Atmos.Microphysics

#abstract type AbstractMicrophysicsParameterSet end
#abstract type AbstractFallingWaterParameterSet   <: AbstractMicrophysicsParameterSet end
#abstract type AbstractSuspendedWaterParameterSet <: AbstractMicrophysicsParameterSet end
#
#abstract type AbstractCloudParameterSet <: AbstractSuspendedWaterParameterSet end
#abstract type AbstractIceParameterSet   <: AbstractSuspendedWaterParameterSet end
#
#abstract type AbstractRainParameterSet  <: AbstractFallingWaterParameterSet end
#abstract type AbstractSnowParameterSet  <: AbstractFallingWaterParameterSet end

const APS = AbstractParameterSet
const ASuspPS = AbstractSuspendedWaterParameterSet
const AFallPS = AbstractFallingWaterParameterSet
const ACPS = AbstractCloudParameterSet
const AIPS = AbstractIceParameterSet
const ARPS = AbstractRainParameterSet
const ASPS = AbstractSnowParameterSet

export ζ_rai
export n0_sno
export lambda
export unpack_parameters

export supersaturation
export G_func

export terminal_velocity

export conv_q_vap_to_q_liq_ice

export conv_q_liq_to_q_rai
export conv_q_ice_to_q_sno

export accretion
export accretion_rain_sink
export accretion_snow_rain

export evaporation_sublimation
export snow_melt

"""
    ζ_rai(param_set, ρ)

 - `param_set` - abstract set with earth parameters
 - `ρ` air density

Returns the proportionality coefficient between terminal velocity of an
individual water drop and the square root of its radius.
"""
function ζ_rai(param_set::APS, ρ::FT) where {FT<:Real}

    _ρ_cloud_liq::FT = ρ_cloud_liq(param_set)
    _C_drag::FT      = Microphysics.C_drag(param_set)
    _grav::FT        = grav(param_set)

    return sqrt(_grav * FT(8/3) / _C_drag * (_ρ_cloud_liq / ρ - FT(1)))
end

"""
    n0_sno(snow_param_set, q_sno, ρ)

 - `snow_param_set` - abstract set with snow microphysics parameters
 - `q_sno` -  snow specific humidity
 - `ρ` - air density

Returns the intercept parameter of the assumed Marshal Palmer distribution of
snow particles.
"""
function n0_sno(snow_param_set::ASPS, q_sno::FT, ρ::FT) where {FT<:Real}

    _ν_sno::FT = ν_sno(snow_param_set)
    _μ_sno::FT = μ_sno(snow_param_set)

    return _μ_sno * (ρ * q_sno)^_ν_sno
end

"""
    lambda(q, ρ)

 - `q` - specific humidity of rain, ice or snow
 - `ρ` - air density
 - `n0` - size disyribution parameter
 - `α`, `β` - mass(radius) parameters

Returns the rate parameter of the assumed size distribution of
particles (rain drops, ice crystals, snow crystals).
"""
function lambda(q::FT, ρ::FT, n0::FT, α::FT, β::FT) where {FT<:Real}

    λ::FT = FT(0)

    if q > FT(0)
        λ = (α * n0 * gamma(β + FT(1)) / ρ / q)^FT(1/(β + 1))
    end
    return λ
end

"""
    unpack_params(param_set, microphysics_param_set, ρ, q_)

 - `param_set` - abstract set with earth parameters
 - `microphysics_param_set` - abstract set with microphysics parameters
 - `q_` - specific humidity
 - `ρ` - air density

Utility function that unpacks microphysics parameters.
"""
function unpack_params(param_set::APS, ice_param_set::AIPS, ρ::FT,
                       q_ice::FT) where {FT<:Real}
    #TODO - make ρ and q_ice optional
    _n0::FT = n0(ice_param_set)
    _α::FT  = α(param_set, ice_param_set)
    _β::FT  = β(ice_param_set)

    return (_n0, _α, _β)
end
function unpack_params(param_set::APS, rain_param_set::ARPS, ρ::FT,
                       q_rai::FT) where {FT<:Real}
    #TODO - make q_rai optional
    _n0::FT = n0(rain_param_set)
    _α::FT  = α(param_set, rain_param_set)
    _β::FT  = β(rain_param_set)
    _γ::FT  = γ(rain_param_set)
    _δ::FT  = δ(rain_param_set)
    _ζ::FT  = ζ_rai(param_set, ρ)
    _η::FT  = η(rain_param_set)

    return (_n0, _α, _β, _γ, _δ, _ζ, _η)
end
function unpack_params(param_set::APS, snow_param_set::ASPS, ρ::FT,
                       q_sno::FT) where {FT<:Real}

    _n0::FT = n0_sno(snow_param_set, q_sno, ρ)
    _α::FT  = α(snow_param_set)
    _β::FT  = β(snow_param_set)
    _γ::FT  = γ(snow_param_set)
    _δ::FT  = δ(snow_param_set)
    _ζ::FT  = ζ(snow_param_set)
    _η::FT  = η(snow_param_set)

    return (_n0, _α, _β, _γ, _δ, _ζ, _η)
end

"""
    supersaturation(param_set, q, ρ, T, Liquid())
    supersaturation(param_set, q, ρ, T, Ice())

 - `param_set` - abstract set with earth parameters
 - `q` - phase partition
 - `ρ` - air density,
 - `T` - air temperature
 - `Liquid()`, `Ice()` - liquid or ice phase to dispatch over.

Returns supersaturation (qv/qv_sat -1) over water or ice.
"""
function supersaturation(param_set::APS, q::PhasePartition{FT}, ρ::FT, T::FT,
                         ::Liquid) where {FT<:Real}

    q_sat::FT = q_vap_saturation_generic(param_set, T, ρ)
    q_vap::FT = q.tot - q.liq - q.ice

    return q_vap/q_sat - FT(1)
end
function supersaturation(param_set::APS, q::PhasePartition{FT}, ρ::FT, T::FT,
                         ::Ice) where {FT<:Real}

    q_sat::FT = q_vap_saturation_generic(param_set, T, ρ, Ice())
    q_vap::FT = q.tot - q.liq - q.ice

    return q_vap/q_sat - FT(1)
end

"""
    G_func(param_set, T, Liquid())
    G_func(param_set, T, Ice())

 - `param_set` - abstract set with earth parameters
 - `T` - air temperature
 - `Liquid()`, `Ice()` - liquid or ice phase to dispatch over.

Utility function combining thermal conductivity and vapor diffusivity effects.
"""
function G_func(param_set::APS, T::FT, ::Liquid) where {FT<:Real}

    _K_therm::FT = K_therm(param_set)
    _R_v::FT     = R_v(param_set)
    _D_vapor::FT = D_vapor(param_set)

    L = latent_heat_vapor(param_set, T)
    p_vs = saturation_vapor_pressure(param_set, T, Liquid())

    return FT(1) / (
              L / _K_therm / T * (L / _R_v / T - FT(1))
              + _R_v * T / _D_vapor / p_vs
           )
end
function G_func(param_set::APS, T::FT, ::Ice) where {FT<:Real}

    _K_therm::FT = K_therm(param_set)
    _R_v::FT     = R_v(param_set)
    _D_vapor::FT = D_vapor(param_set)

    L = latent_heat_sublim(param_set, T)
    p_vs = saturation_vapor_pressure(param_set, T, Ice())

    return FT(1) / (
              L / _K_therm / T * (L / _R_v / T - FT(1))
              + _R_v * T / _D_vapor / p_vs
           )
end

"""
    terminal_velocity(param_set, microphysics_param_set, ρ, q_)

 - `param_set` - abstract set with earth parameters
 - `microphysics_param_set` - abstract set with rain or snow microphysics parameters
 - `ρ` - air density
 - `q_` - rain or snow specific humidity

Returns the mass weighted average terminal velocity assuming
Marshall Palmer 1948 distribution of rain drops and snow crystals.
"""
function terminal_velocity(param_set::APS, fall_param_set::AFallPS,
                           ρ::FT, q_::FT) where {FT <: Real}
    fall_w = FT(0)
    if q_ > FT(0)

        (_n0, _α, _β, _γ, _δ, _ζ, _η) = unpack_params(param_set, fall_param_set, ρ, q_)
        _λ::FT = lambda(q_, ρ, _n0, _α, _β)

        fall_w = _ζ * _λ^(-_η) * gamma(_η + _β + FT(1)) / gamma(_β + FT(1))
    end

    return fall_w
end

"""
    conv_q_vap_to_q_liq_ice(cloud_param_set::ACPS, q_sat, q)
    conv_q_vap_to_q_liq_ice(ice_param_set::AIPS, q_sat, q)

 - `cloud_param_set` - abstract set with cloud microphysics parameters
 - `ice_param_set` - abstract set with ice microphysics parameters
 - `q_sat` - PhasePartition at equilibrium
 - `q` - current PhasePartition

Returns the cloud water tendency due to condensation/evaporation
or cloud ice tendency due to sublimation/resublimation.
The tendency is obtained assuming a relaxation to equilibrium with
constant timescale.
"""
function conv_q_vap_to_q_liq_ice(cloud_param_set::ACPS,
                                 q_sat::PhasePartition{FT},
                                 q::PhasePartition{FT}) where {FT<:Real}

    _τ_cond_evap::FT = τ_cond_evap(cloud_param_set)

    return (q_sat.liq - q.liq) / _τ_cond_evap
end
function conv_q_vap_to_q_liq_ice(ice_param_set::AIPS,
                                 q_sat::PhasePartition{FT},
                                 q::PhasePartition{FT}) where {FT<:Real}

    _τ_sub_resub::FT = τ_sub_resub(ice_param_set)

    return (q_sat.ice - q.ice) / _τ_sub_resub
end

"""
    conv_q_liq_to_q_rai(rain_param_set, q_liq)

 - `rain_param_set` - abstract set with rain microphysics parameters
 - `q_liq` - liquid water specific humidity

Returns the q_rai tendency due to collisions between cloud droplets
(autoconversion) parametrized following Kessler 1995.
"""
function conv_q_liq_to_q_rai(rain_param_set::ARPS, q_liq::FT) where {FT <: Real}

    _τ_acnv::FT = τ_acnv(rain_param_set)
    _q_liq_threshold::FT = q_liq_threshold(rain_param_set)

    return max(FT(0), q_liq - _q_liq_threshold) / _τ_acnv
end

"""
    conv_q_ice_to_q_sno(param_set, ice_param_set, q, ρ)

 - `param_set` - abstract set with earth parameters
 - `ice_param_set` - abstract set with ice microphysics parameters
 - `q` - phase partition
 - `ρ` - air density
 - `T` - air temperature

Returns the q_sno tendency due to autoconversion from ice.
Parameterized following Harrington et al 1996 and Kaul et al 2015
"""
function conv_q_ice_to_q_sno(param_set::APS, ice_param_set::AIPS,
                             q::PhasePartition{FT}, ρ::FT, T::FT) where {FT<:Real}
    acnv_rate = FT(0)

    if q.ice > FT(0)

        _S::FT = supersaturation(param_set, q, ρ, T, Ice())
        _G::FT = G_func(param_set, T, Ice())

        _r_ice_snow::FT = r_ice_snow(ice_param_set)
        (_n0, _α, _β) = unpack_params(param_set, ice_param_set, ρ, q.ice)
        _λ::FT = lambda(q.ice, ρ, _n0, _α, _β)

        acnv_rate = FT(4) * π * _S * _G * _n0 / ρ * exp(-_λ * _r_ice_snow) *
                    (_r_ice_snow^FT(2) / _β
                      +
                     (_r_ice_snow * _λ + FT(1)) / _λ^FT(2)
                    )
    end
    return acnv_rate
end

"""
    accretion(param_set, suspended_param_set, falling_param_set, q_susp, q_fall, ρ)

 - `param_set` - abstract set with earth parameters
 - `suspended_param_set` - abstract set with cloud microphysics or cloud ice parameters
 - `falling_param_set` - abstract set with rain or snow microphysics parameters
 - `q_susp` - cloud water or cloud ice specific humidity
 - `q_fall` - rain water or snow specific humidity
 - `ρ` - rain water or snow specific humidity

Returns the sink to suspended water (cloud water or cloud ice) due to collisions
with falling water (rain or snow).
"""
function accretion(param_set::APS,
                   suspended_param_set::ASuspPS,
                   falling_param_set::AFallPS,
                   q_susp::FT, q_fall::FT, ρ::FT) where {FT<:Real}

    #TODO - have another subtype for suspended and falling parameter types?
    accr_rate = FT(0)
    if (q_susp > FT(0) && q_fall > FT(0))

        (_n0, _α, _β, _γ, _δ, _ζ, _η) = unpack_params(param_set, falling_param_set, ρ, q_fall)
        _λ::FT = lambda(q_fall, ρ, _n0, _α, _β)
        _E::FT  = E(suspended_param_set, falling_param_set)

        accr_rate = q_susp * _E * _n0 * _γ * _ζ * gamma(_δ + _η + FT(1)) /
                    _λ^(_δ + _η + FT(1))
    end
    return accr_rate
end

"""
    accretion_rain_sink(param_set, ice_param_set, rain_param_set, q_ice, q_rai, ρ)

 - `param_set` - abstract set with earth parameters
 - `ice_param_set` - abstract set with ice microphysics parameters
 - `rain_param_set` - abstract set with rain microphysics parameters
 - `q_ice` - ice water specific humidity
 - `q_rai` - rain water specific humidity
 - `ρ` - air density

Returns the sink of rain water (partial source of snow) due to collisions
with cloud ice.
"""
function accretion_rain_sink(param_set::APS, ice_param_set::AIPS,
                             rain_param_set::ARPS, q_ice::FT,
                             q_rai::FT, ρ::FT) where {FT<:Real}

    accr_rate = FT(0)
    if (q_ice > FT(0) && q_rai > FT(0))

        (_n0_ice, _α_ice, _β_ice) =
            unpack_params(param_set, ice_param_set, ρ, q_ice)
        (_n0_rai, _α_rai, _β_rai, _γ_rai, _δ_rai, _ζ_rai, _η_rai) =
            unpack_params(param_set, rain_param_set, ρ, q_rai)
        _E::FT  = E(ice_param_set, rain_param_set)

        _λ_ice::FT = lambda(q_ice, ρ, _n0_ice, _α_ice, _β_ice)
        _λ_rai::FT = lambda(q_rai, ρ, _n0_rai, _α_rai, _β_rai)

        accr_rate = _E * _n0_rai * _n0_ice * _α_rai * _γ_rai * _ζ_rai / _λ_ice *
                    gamma(_β_rai + _δ_rai + _η_rai + FT(1)) /
                    _λ_rai^(_β_rai + _δ_rai + _η_rai + FT(1))
    end
    return accr_rate
end

"""
    accretion_snow_rain(param_set, i_param_set, j_param_set, q_i, q_j, ρ)

 - `i` - snow for temperatures below freezing or rain for temperatures above freezing
 - `j` - rain for temperatures below freezing or rain for temperatures above freezing
 - `param_set` - abstract set with earth parameters
 - `_param_set` - abstract set with snow or rain microphysics parameters
 - `q_` - specific humidity of snow or rain
 - `ρ` - air density

Returns the accretion rate between rain and snow.
Collisions between rain and snow result in
snow in temperatures below freezing andin rain in temperatures above freezing.
"""
function accretion_snow_rain(param_set::APS, i_param_set::AFallPS, j_param_set::AFallPS,
                             q_i::FT, q_j::FT, ρ::FT) where {FT<:Real}

    accr_rate = FT(0)
    if (q_i > FT(0) && q_j > FT(0))

        (_n0_i, _α_i, _β_i, _γ_i, _δ_i, _ζ_i, _η_i) =
            unpack_params(param_set, i_param_set, ρ, q_i)
        (_n0_j, _α_j, _β_j, _γ_j, _δ_j, _ζ_j, _η_j) =
            unpack_params(param_set, j_param_set, ρ, q_j)

        _E_ij::FT  = E(i_param_set, j_param_set)

        _λ_i::FT = lambda(q_i, ρ, _n0_i, _α_i, _β_i)
        _λ_j::FT = lambda(q_j, ρ, _n0_j, _α_j, _β_j)

        _v_ti = terminal_velocity(param_set, i_param_set, ρ, q_i, ρ)
        _v_tj = terminal_velocity(param_set, j_param_set, ρ, q_j, ρ)

        accr_rate = π * _n0_i * _n0_j * _α_j * _E_ij * abs(_v_ti - _v_tj) *
                    (
                      FT(2) * gamma(_β_j + FT(1)) / _λ_i^FT(3) / _λ_j^(β_j + FT(1)) +
                      FT(2) * gamma(_β_j + FT(2)) / _λ_i^FT(2) / _λ_j^(β_j + FT(2)) +
                      gamma(_β_j + FT(3)) / _λ_i /_λ_j^(_β_j + FT(3))
                    )
    end
    return accr_rate
end

"""
    evaporation_sublimation(param_set, rain_param_set, q_rai, ρ, T)
    evaporation_sublimation(param_set, snow_param_set, q_sno, ρ, T)

 - `param_set` - abstract set with earth parameters
 - `rain_param_set` - abstract set with rain microphysics parameters
 - `snow_param_set` - abstract set with snow microphysics parameters
 - `q` - phase partition
 - `q_rai` - rain specific humidity
 - `q_sno` - snow specific humidity
 - `ρ` - air density
 - `T` - air temperature

Returns the tendency due to rain evaporation or snow sublimation.
"""
function evaporation_sublimation(param_set::APS, rain_param_set::ARPS,
                                 q::PhasePartition{FT}, q_rai::FT, ρ::FT,
                                 T::FT) where {FT<:Real}
    evap_subl_rate = FT(0)
    if q_rai  > FT(0)

        _a_vent::FT  = a_vent(rain_param_set)
        _b_vent::FT  = b_vent(rain_param_set)
        _ν_air::FT   = ν_air(param_set)
        _D_vapor::FT = D_vapor(param_set)

        _S::FT = supersaturation(param_set, q, ρ, T, Liquid())
        _G::FT = G_func(param_set, T, Liquid())

        (_n0, _α, _β, _γ, _δ, _ζ, _η) =
            unpack_params(param_set, rain_param_set, ρ, q_rai)
        _λ::FT = lambda(q_rai, ρ, _n0, _α, _β)

        evap_subl_rate = 4 * π * _n0 / ρ * _S * _G / _λ^FT(2) * (
          _a_vent +
          _b_vent * (_ν_air / _D_vapor)^FT(1/3) * (FT(2) * _ζ / _ν_air)^FT(1/2) *
            gamma((_η + FT(5)) / FT(2)) / _λ^((_η + FT(1))/FT(2))
        )
    end
    return evap_subl_rate
end
function evaporation_sublimation(param_set::APS, snow_param_set::ASPS,
                                 q::PhasePartition{FT}, q_sno::FT, ρ::FT,
                                 T::FT) where {FT<:Real}
    evap_subl_rate = FT(0)
    if q_sno  > FT(0)

        _a_vent::FT  = a_vent(snow_param_set)
        _b_vent::FT  = b_vent(snow_param_set)
        _ν_air::FT   = ν_air(param_set)
        _D_vapor::FT = D_vapor(param_set)

        _S::FT = supersaturation(param_set, q, ρ, T, Ice())
        _G::FT = G_func(param_set, T, Ice())

        (_n0, _α, _β, _γ, _δ, _ζ, _η) =
            unpack_params(param_set, snow_param_set, ρ, q_sno)

        _λ::FT = lambda(q_sno, ρ, _n0, _α, _β)

        evap_subl_rate = 4 * π * _n0 / ρ * _S * _G / _λ^FT(2) * (
          _a_vent +
          _b_vent * (_ν_air / _D_vapor)^FT(1/3) * (FT(2) * _ζ / _ν_air)^FT(1/2) *
            gamma((_η + FT(5)) / FT(2)) / _λ^((_η + FT(1))/FT(2))
        )
    end
    return evap_subl_rate
end

"""
    snow_melt(param_set, snow_param_set, q_sno, ρ)

 - `param_set` - abstract set with earth parameters
 - `snow_param_set` - abstract set with snow microphysics parameters
 - `q_sno` - snow water specific humidity
 - `ρ` - air density

Returns the tendency due to snow melt.
"""
function snow_melt(param_set::APS, snow_param_set::ASPS, q_sno::FT,
                   ρ::FT) where {FT<:Real}

    snow_melt_rate = FT(0)
    if q_sno  > FT(0)

        _a_vent::FT  = a_vent(snow_param_set)
        _b_vent::FT  = b_vent(snow_param_set)
        _ν_air::FT   = ν_air(param_et)
        _D_vapor::FT = D_vapor(param_set)
        _K_therm::FT = K_therm(param_set)

        _T_freeze = T_freeze(param_set)
        L = latent_heat_fusion(param_set, T)

        (_n0, _α, _β, _γ, _δ, _ζ, _η) =
            unpack_params(param_set, snow_param_set, ρ, q)

        _λ::FT = lambda(q_sno, ρ, _n0, _α, _β)

        snow_melt_rate = 4 * π * _n0 / ρ *
          _K_therm / L * (T - _T_freeze) / _λ^FT(2) * (
          _a_vent_sno +
          _b_vent_sno * (_ν_air / _D_vapor)^FT(1/3) *
            (FT(2) * _ζ / _ν_air)^FT(1/2) *
            gamma((_η + FT(5)) / FT(2)) / _λ^((_η + FT(1))/FT(2))
        )
    end
    return snow_melt_rate
end

end #module Microphysics.jl
