#####
##### Helper functions
#####

function get_z(grid, z_scale, N_poly)

    return reshape(grid.vgeo[(1:(N_poly+1)^2:(N_poly+1)^3),CLIMA.Mesh.Grids.vgeoid.x3id,:],:)*z_scale
end

struct DataFile{S<:AbstractString}
    filename::S
    prop_name::S
    DataFile(filename::S) where {S} = new{S}(filename, "num")
end
function (data_file::DataFile)(num)
    return "$(data_file.filename)_$(data_file.prop_name)=$(num)"
end

"""
    collect_data(output_data::DataFile, n_steps::Int)

Get all data from the entire simulation run.
Data is in the form of a dictionary, whose keys
are integers corresponding to the output number.
The values are a NetCDF file, which includes the
time information.
"""
function collect_data(output_data::DataFile, n_steps::Int)
  all_data = Dict()
  for i in 0:n_steps-1
    f = output_data(i)
    all_data[i] = NCDataset(f*".nc", "r")
  end
  return all_data
end

function get_vars_from_stack(grid::DiscontinuousSpectralElementGrid{T,dim,N},
                             Q::MPIStateArray, # dg.auxstate or state.data
                             bl::BalanceLaw, # SoilModelHeat
                             vars_fun::F;
                             exclude=[]) where {T,dim,N,F<:Function}
    D = Dict()
    FT = eltype(Q)
    vf = vars_fun(bl,FT)
    nz = size(Q, 3)
    R = (1:(N+1)^2:(N+1)^3)
    fn = flattenednames(vf)
    for e in exclude
      filter!(x->!(x == e), fn)
    end
    for v in fn
        D[v] = reshape(FT[getproperty(Vars{vf}(Q[i, :, e]), Symbol(v)) for i in R, e in 1:nz],:)
    end
    return D
end

"""
    TimeContinuousData{FT<:AbstractFloat,A<:AbstractArray}

Creates a time-continuous representation
of temporally discrete data. Example:

```julia
FT = Float64

data_discrete = FT[1, 2, 3, 4]
time_discrete = FT[0, 10, 20, 30]

data_continuous = TimeContinuousData(time_discrete, data_discrete)

data_at_40 = data_continuous(40)
```
"""
struct TimeContinuousData{FT<:AbstractFloat,A<:AbstractArray}
  itp
  ext
  bounds::Tuple{FT,FT}
  function TimeContinuousData(time_data::A, data::A) where {A<:AbstractArray}
    FT = eltype(A)
    itp = interpolate((time_data,), data, Gridded(Linear()))
    ext = extrapolate(itp, Flat())
    bounds = (first(data), last(data))
    return new{FT,A}(itp, ext, bounds)
  end
end
function (cont_data::TimeContinuousData)(x::A) where {A<:AbstractArray}
  return [ cont_data.bounds[1] < x_i < cont_data.bounds[2] ?
  cont_data.itp(x_i) : cont_data.ext(x_i) for x_i in x]
end

function (cont_data::TimeContinuousData)(x_i::FT) where {FT<:AbstractFloat}
  return cont_data.bounds[1] < x_i < cont_data.bounds[2] ? cont_data.itp(x_i) : cont_data.ext(x_i)
end

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

