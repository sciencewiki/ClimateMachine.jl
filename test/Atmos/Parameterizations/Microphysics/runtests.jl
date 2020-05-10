using Test
include(joinpath("../../../", "testhelpers.jl"))

@testset "Microphysics tests" begin
    tests = [
        (1, "unit_tests.jl")
        (1, "saturation_adjustment.jl")
        (1, "warm_rain.jl")
    ]

    runmpi(tests, @__FILE__)
end
