using MPI
using Test
using CLIMA
using CLIMA.MPIStateArrays

let
    CLIMA.init()
    ArrayType = CLIMA.array_type()
    mpicomm = MPI.COMM_WORLD
    FT = Float32
    Q = MPIStateArray{FT}(mpicomm, ArrayType, 4, 4, 4)
    Qb = reshape(Q, (16, 4, 1));

    Q .= 1
    Qb .= 1

    @testset "MPIStateArray Reshape" begin
        @test minimum(Q[:] .== 1)
        @test minimum(Qb[:] .== 1)
    end

end
