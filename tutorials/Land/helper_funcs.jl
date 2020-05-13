#####
##### Helper functions
#####

using DocStringExtensions: FIELDS

"""
    get_z(grid, z_scale)

Gauss-Lobatto points along the z-coordinate

 - `grid` DG grid
 - `z_scale` multiplies `z-coordinate`
"""
function get_z(
    grid::DiscontinuousSpectralElementGrid{T, dim, N},
    z_scale,
) where {T, dim, N}
    # TODO: this currently uses some internals: provide a better way to do this
    return reshape(
        grid.vgeo[
            (1:((N + 1)^2):((N + 1)^3)),
            ClimateMachine.Mesh.Grids.vgeoid.x3id,
            :,
        ],
        :,
    ) * z_scale
end

"""
    SingleStackGrid(MPI, velems, N_poly)

Returns a single-stack grid
 - `MPI` mpi communicator
 - `velems` range of vertical elements
 - `N_poly` polynomial order
 - `FT` float type
 - `DeviceArray` polynomial order
"""
function SingleStackGrid(MPI, velems, N_poly, FT, DeviceArray)
    topl = StackedBrickTopology(
        MPI.COMM_WORLD,
        # Does it ever make sense to have Float32 data with a Float64 mesh?
        # i.e., should we grab FT from eltype(elemrange), or pass in FT separately?
        (FT.(0.0:1), FT.(0.0:1), FT.(velems));
        periodicity = (true, true, false),
        boundary = ((0, 0), (0, 0), (1, 2)),
    )
    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = DeviceArray,
        polynomialorder = N_poly,
    )
    return grid
end

"""
    DataFile

Used to ensure generation and reading of
a set of files are automatically synchronized.

# Example:
```julia
julia> output_data = DataFile("my_data")
DataFile{String}("my_data", "num")

julia> output_data(1)
"my_data_num=1"
```
# Fields
$(FIELDS)
"""
struct DataFile{S <: AbstractString}
    "filename"
    filename::S
    "property name"
    prop_name::S
    DataFile(filename::S) where {S} = new{S}(filename, "num")
end
function (data_file::DataFile)(num)
    return "$(data_file.filename)_$(data_file.prop_name)=$(num)"
end

"""
    prep_for_io(z_label, vars_from_stack)

Prepares `vars_from_stack` to
be passed into `write_data`.
"""
prep_for_io(z_label, vars_from_stack) =
    Dict(k => ([z_label], v) for (k, v) in vars_from_stack)

"""
    collect_data(output_data::DataFile, n_steps::Int)

Get all data from the entire simulation run.
Data is in the form of a nested dictionary,
whose keys are integers corresponding to the
output number. The next level keys are the
variable names, and the values are the
corresponding values along the stack.
"""
function collect_data(output_data::DataFile, n_steps::Int)
    all_keys = NCDataset(output_data(0) * ".nc", "r") do ds
        keys(ds)
    end
    all_data = Dict((
        i => Dict((k => [] for k in all_keys)...) for i in 0:(n_steps - 1)
    )...)
    for i in 0:(n_steps - 1)
        f = output_data(i)
        NCDataset(f * ".nc", "r") do ds
            for k in all_keys
                all_data[i][k] = ds[k][:]
            end
        end
    end
    return all_data
end

"""
    get_prop_recursive(var, v::String)

Recursive `getproperty` on struct `var` based on dot chain (`a.b`).

# Example:
```
struct A; a; end
struct B; b; end
struct C; c; end
var = A(B(C(1)))
c = get_prop_recursive(var, "a.b.c")
```
"""
function get_prop_recursive(var, v::String)
    if occursin(".", v)
        s = split(v, ".")
        if length(s) == 1
            return getproperty(var, Symbol(s[1]))
        else
            return get_prop_recursive(
                getproperty(var, Symbol(s[1])),
                join(s[2:end], "."),
            )
        end
    else
        return getproperty(var, Symbol(v))
    end
end

"""
    get_vars_from_stack(
        grid::DiscontinuousSpectralElementGrid{T,dim,N},
        Q::MPIStateArray,
        bl::BalanceLaw,
        vars_fun::F;
        exclude=[]
        ) where {T,dim,N,F<:Function}

Returns a dictionary whose keys are computed
from `vars_fun` (e.g., `vars_state_conservative`)
and values are arrays of each variable along
the z-coordinate. Skips variables whose keys
are listed in `exclude`.
"""
function get_vars_from_stack(
    grid::DiscontinuousSpectralElementGrid{T, dim, N},
    Q::MPIStateArray,
    bl::BalanceLaw,
    vars_fun::F;
    exclude = [],
) where {T, dim, N, F <: Function}
    D = Dict()
    FT = eltype(Q)
    vf = vars_fun(bl, FT)
    nz = size(Q, 3)
    R = (1:((N + 1)^2):((N + 1)^3))
    fn = flattenednames(vf)
    for e in exclude
        filter!(x -> !(x == e), fn)
    end
    for v in fn
        vars = [Vars{vf}(Q[i, :, e]) for i in R, e in 1:nz]
        D[v] = get_prop_recursive.(vars, Ref(v))
        D[v] = reshape(D[v], :)
    end
    return D
end
