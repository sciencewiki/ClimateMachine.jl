using Test
using ClimateMachine.Microphysics
using ClimateMachine.MoistThermodynamics

using CLIMAParameters
using CLIMAParameters.Planet: ρ_cloud_liq, R_v, grav, R_d, molmass_ratio
using CLIMAParameters.Atmos.Microphysics

struct LiquidParameterSet <: AbstractLiquidParameterSet end
struct IceParameterSet    <: AbstractIceParameterSet end
struct RainParameterSet   <: AbstractRainParameterSet end
struct SnowParameterSet   <: AbstractSnowParameterSet end

struct MicropysicsParameterSet{L,I,R,S} <: AbstractMicrophysicsParameterSet
    liquid ::L
    ice ::I
    rain ::R
    snow ::S
end

struct EarthParameterSet{M} <: AbstractEarthParameterSet
    microphys_param_set::M
end

microphys_param_set = MicropysicsParameterSet(
    LiquidParameterSet(),
    IceParameterSet(),
    RainParameterSet(),
    SnowParameterSet(),
)

param_set = EarthParameterSet(microphys_param_set)
liquid_param_set = param_set.microphys_param_set.liquid
ice_param_set = param_set.microphys_param_set.ice
rain_param_set = param_set.microphys_param_set.rain
snow_param_set = param_set.microphys_param_set.snow

@testset "RainFallSpeed" begin
    # eq. 5d in Smolarkiewicz and Grabowski 1996
    # https://doi.org/10.1175/1520-0493(1996)124<0487:TTLSLM>2.0.CO;2
    function terminal_velocity_empir(
        q_rai::FT,
        q_tot::FT,
        ρ::FT,
        ρ_air_ground::FT,
    ) where {FT <: Real}
        rr = q_rai / (1 - q_tot)
        vel = FT(14.34) * ρ_air_ground^FT(0.5) * ρ^-FT(0.3654) * rr^FT(0.1346)
        return vel
    end

    # some example values
    q_rain_range = range(1e-8, stop = 5e-3, length = 10)
    ρ_air, q_tot, ρ_air_ground = 1.2, 20 * 1e-3, 1.22

    for q_rai in q_rain_range

        @test terminal_velocity(param_set, rain_param_set, ρ_air, q_rai) ≈
              terminal_velocity_empir(q_rai, q_tot, ρ_air, ρ_air_ground) atol =
            0.2 * terminal_velocity_empir(q_rai, q_tot, ρ_air, ρ_air_ground)

    end
end

@testset "CloudCondEvap" begin

    q_liq_sat = 5e-3
    frac = [0.0, 0.5, 1.0, 1.5]

    _τ_cond_evap = τ_relax(liquid_param_set)

    for fr in frac

        q_liq = q_liq_sat * fr

        @test conv_q_vap_to_q_liq_ice(
            liquid_param_set,
            PhasePartition(0.0, q_liq_sat, 0.0),
            PhasePartition(0.0, q_liq, 0.0),
        ) ≈ (1 - fr) * q_liq_sat / _τ_cond_evap
    end
end

@testset "RainAutoconversion" begin

    _q_liq_threshold = q_liq_threshold(rain_param_set)
    _τ_acnv = τ_acnv(rain_param_set)

    q_liq_small = 0.5 * _q_liq_threshold
    @test conv_q_liq_to_q_rai(rain_param_set, q_liq_small) == 0.0

    q_liq_big = 1.5 * _q_liq_threshold
    @test conv_q_liq_to_q_rai(rain_param_set, q_liq_big) ==
          0.5 * _q_liq_threshold / _τ_acnv
end

@testset "RainAccretion" begin

    # eq. 5b in Smolarkiewicz and Grabowski 1996
    # https://doi.org/10.1175/1520-0493(1996)124<0487:TTLSLM>2.0.CO;2
    function accretion_empir(q_rai::FT, q_liq::FT, q_tot::FT) where {FT <: Real}
        rr = q_rai / (FT(1) - q_tot)
        rl = q_liq / (FT(1) - q_tot)
        return FT(2.2) * rl * rr^FT(7 / 8)
    end

    # some example values
    q_rain_range = range(1e-8, stop = 5e-3, length = 10)
    ρ_air, q_liq, q_tot = 1.2, 5e-4, 20e-3

    for q_rai in q_rain_range

        @test accretion(param_set, liquid_param_set, rain_param_set, q_liq,
                q_rai, ρ_air) ≈ accretion_empir(q_rai, q_liq,
                q_tot) atol = (0.1 * accretion_empir(q_rai, q_liq, q_tot))
    end
end

@testset "RainEvaporation" begin

    # eq. 5c in Smolarkiewicz and Grabowski 1996
    # https://doi.org/10.1175/1520-0493(1996)124<0487:TTLSLM>2.0.CO;2
    function rain_evap_empir(
        param_set::AbstractParameterSet,
        q_rai::FT,
        q::PhasePartition,
        T::FT,
        p::FT,
        ρ::FT,
    ) where {FT <: Real}

        q_sat = q_vap_saturation_generic(param_set, T, ρ)
        q_vap = q.tot - q.liq
        rr = q_rai / (1 - q.tot)
        rv_sat = q_sat / (1 - q.tot)
        S = q_vap / q_sat - 1

        ag, bg = FT(5.4 * 1e2), FT(2.55 * 1e5)
        G = FT(1) / (ag + bg / p / rv_sat) / ρ

        av, bv = FT(1.6), FT(124.9)
        F =
            av * (ρ / FT(1e3))^FT(0.525) * rr^FT(0.525) +
            bv * (ρ / FT(1e3))^FT(0.7296) * rr^FT(0.7296)

        return 1 / (1 - q.tot) * S * F * G
    end

    # example values
    T, p = 273.15 + 15, 90000.0
    ϵ = 1.0 / molmass_ratio(param_set)
    p_sat = saturation_vapor_pressure(param_set, T, Liquid())
    q_sat = ϵ * p_sat / (p + p_sat * (ϵ - 1.0))
    q_rain_range = range(1e-8, stop = 5e-3, length = 10)
    q_tot = 15e-3
    q_vap = 0.15 * q_sat
    q_ice = 0.0
    q_liq = q_tot - q_vap - q_ice
    q = PhasePartition(q_tot, q_liq, q_ice)
    R = gas_constant_air(param_set, q)
    ρ = p / R / T

    for q_rai in q_rain_range

        @test evaporation_sublimation(param_set, rain_param_set, q, q_rai, ρ, T) ≈
              rain_evap_empir(param_set, q_rai, q, T, p, ρ) atol =
            -0.5 * rain_evap_empir(param_set, q_rai, q, T, p, ρ)
    end
end
