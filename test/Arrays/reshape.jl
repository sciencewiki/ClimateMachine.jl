using MPI
using CLIMA
using CLIMA.MPIStateArrays

#CLIMA.init(disable_gpu = true)
CLIMA.init()
const ArrayType = CLIMA.array_type()
const mpicomm = MPI.COMM_WORLD
FT = Float32
Q = MPIStateArray{FT}(mpicomm, ArrayType, 4, 4, 4)
println(size(Array(Q)))
Qb = reshape(Q, (16, 4, 1))

# this works
Q .= Q

# this provides an error
Qb .= Qb
