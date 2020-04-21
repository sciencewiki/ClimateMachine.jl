using MPI
using CLIMA
using CLIMA.MPIStateArrays
CLIMA.init()
const ArrayType = CLIMA.array_type()
const mpicomm = MPI.COMM_WORLD
Q = MPIStateArray{Float32}(mpicomm, ArrayType, 4, 4, 4)
println(size(Array(Q)))
Qb = reshape(Q, (16, 4, 1))
# this works
Q .= Q
# this provides an error
Qb .= Qb
