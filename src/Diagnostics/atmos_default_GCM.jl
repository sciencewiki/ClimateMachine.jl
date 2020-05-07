
# # Dry Atmosphere GCM Config File
# 
# This file computes selected diagnostics for the GCM and outputs them on the
# spherical interpolated diagnostic grid.
#
# It is called by setting ```dgn_config``` in the GCM run file
# (e.g. heldsuarez.jl) to ```setup_atmos_default_GCM_diagnostics```
#
# TODO:
# - the struct functions need to be used to generalise the choices
#   - these require Array(FT, 1) but interpolation requires FT
#   - maybe will need to define a conversion function? struct(num_thermo(FT), Nel)(vari) --> Array(num_thermo(FT), vari, Nel)
# - elementwise aggregation of interpolated vars very slow
# - enable zonal means and calculation of covariances using those means
# - add more variables, including hioriz streamfunction from laplacial of vorticity (LN)
# - density weighting
# - maybe change thermo/dun separation to local/nonlocal vars?


using Statistics
using ..Atmos
using ..Atmos: thermo_state, turbulence_tensors
using ..Mesh.Topologies
using ..Mesh.Grids
using ..MoistThermodynamics
using CLIMAParameters.Atmos.SubgridScale: inv_Pr_turb
using LinearAlgebra
using ..DGmethods
using ..DGmethods:
    vars_state, vars_aux

using Printf

# read in definitions of all possible fields
include("diagnostic_vars_GCM.jl")

# read in functions for calculating variable not in state/aux/diff
include("diagnostic_fields.jl")

# define init function
function atmos_default_GCM_init(dgngrp::DiagnosticsGroup, currtime)
    @assert dgngrp.interpol !== nothing
end


# Set up thermodynamic vars (user-selected) - this func needs to be integrated into ```compute_thermo!```
function vars_thermo(FT)
    @vars begin
        T::FT
        θ_dry::FT
    end
end
num_thermo(FT) = varsize(vars_thermo(FT))
thermo_vars(array) = Vars{vars_thermo(eltype(array))}(array)

function compute_thermo!(bl, state, aux, ijk, e, thermoQ)
    ts = thermo_state(bl, state, aux)
    #th = thermo_vars(Array{Float32,1}(thermoQ[ijk, :, e]))
    #th.T = air_temperature(ts) # this df works
    #th.θ_dry = dry_pottemp(ts)

    thermoQ[ijk, 1, e] = air_temperature(ts)
    thermoQ[ijk, 2, e] = dry_pottemp(ts)
    return nothing
end

# Set up dynamic vars (user-selected) - this func needs to be integrated into ```compute_dyn!```
function vars_dyn(FT)
    @vars begin
        Ω₁::FT
        Ω₂::FT
        Ω₃::FT
    end
end
num_dyn(FT) = varsize(vars_dyn(FT))
dyn_vars(array) = Vars{vars_dyn(eltype(array))}(array)

function compute_dyn!( ijk, e, dynQ, vortvec)
    #ts = thermo_state(bl, state, aux) # thermodynamic state
    #dy = dyn_vars(Array{Float32,1}(dynQ[ijk, : , e])) # init dynQ
    ##@info @sprintf("""vortvec %s""", vortvec)
    #dy.Ω₁= vortvec[1]
    #dy.Ω₂= vortvec[2]
    #dy.Ω₃= vortvec[3]

    dynQ[ijk, 1 , e] = vortvec[1]
    dynQ[ijk, 2 , e] = vortvec[2]
    dynQ[ijk, 3 , e] = vortvec[3]
    return nothing
end

# Set up state vars on interpolated grid - this func needs to be integrated into ```compute_diagnosticsums_GCM!```
function vars_statei(FT)
    @vars begin
        ρ::FT
        ρu::FT
        ρv::FT
        ρw::FT
        ρe::FT
    end
end
statei_vars(array) = Vars{vars_statei(eltype(array))}(array)

# Set up thermo vars on interpolated grid - this func needs to be integrated into ```compute_diagnosticsums_GCM!```
function vars_thermoi(FT)
    @vars begin
        T::FT
        θ_dry::FT
    end
end
thermoi_vars(array) = Vars{vars_thermoi(eltype(array))}(array)

# Set up dynamic vars on interpolated grid - this func needs to be integrated into ```compute_diagnosticsums_GCM!```
function vars_dyni(FT)
    @vars begin
        Ω₁::FT
        Ω₂::FT
        Ω₃::FT
    end
end
dyni_vars(array) = Vars{vars_dyni(eltype(array))}(array)

# aggregate all diags, including calculations done after interpolation to sphere
function compute_diagnosticsums_GCM!(
    all_state_data,
    all_thermo_data,
    all_dyn_data,
    dsumsi,
    lo,
    la,
    le
    )

    # set up empty variable structs with interpolated arrays as templates
    #state_i = statei_vars(Array{Float32,1}(all_state_data[le,la,lo,:]))
    #th_i = thermoi_vars(Array{Float32,1}(all_thermo_data[le,la,lo,:]))
    #dyn_i = dyni_vars(Array{Float32,1}(all_dyn_data[le,la,lo,:]))

    state_i = statei_vars(all_state_data[lo,la,le,:])
    th_i = thermoi_vars(all_thermo_data[lo,la,le,:])
    dyn_i = dyni_vars(all_dyn_data[lo,la,le,:])

    # calculate ds. vars that will be saved (these need to be selected from the diagnostic_vars(_GCM) file)
    """ds = diagnostic_vars(dsumsi[lo,la,le,:])
    ds.u = state_i.ρu / state_i.ρ
    ds.v = state_i.ρv / state_i.ρ
    ds.w = state_i.ρw / state_i.ρ
    ds.T     = th_i.T
    ds.thd = th_i.θ_dry
    ds.vortrel = dyn_i.Ω₃"""

    dsumsi[lo,la,le,1] = state_i.ρu / state_i.ρ
    dsumsi[lo,la,le,2] = state_i.ρv / state_i.ρ
    dsumsi[lo,la,le,3] = state_i.ρw / state_i.ρ
    dsumsi[lo,la,le,4] = th_i.T
    dsumsi[lo,la,le,5] = th_i.θ_dry
    dsumsi[lo,la,le,6] = dyn_i.Ω₃
    #@info @sprintf("""size(dyn_i.Ω₃) %s""", size(dyn_i.Ω₃))

    # zonal means
    #ds.T_zm = mean(.*1., ds.T; dims = 3)
    #ds.u_zm = mean((ds.u); dims = 3 )
    #v_zm = mean(ds.v; dims = 3)
    #w_zm = mean(ds.w; dims = 3)

    # (co)variances
    #ds.uvcovariance = (ds.u .- ds.u_zm) * (ds.v .- v_zm)
    #ds.vTcovariance = (ds.v .- v_zm) * (ds.T .- ds.T_zm)

    return nothing
end

# get diment=sions for the interpolated grid
function get_dims(dgngrp)
    if dgngrp.interpol !== nothing
        if dgngrp.interpol isa InterpolationBrick
            if Array ∈ typeof(dgngrp.interpol.x1g).parameters
                h_x1g = dgngrp.interpol.x1g
                h_x2g = dgngrp.interpol.x2g
                h_x3g = dgngrp.interpol.x3g
            else
                h_x1g = Array(dgngrp.interpol.x1g)
                h_x2g = Array(dgngrp.interpol.x2g)
                h_x3g = Array(dgngrp.interpol.x3g)
            end
            dims = OrderedDict("x" => h_x1g, "y" => h_x2g, "z" => h_x3g)
        elseif dgngrp.interpol isa InterpolationCubedSphere
            if Array ∈ typeof(dgngrp.interpol.rad_grd).parameters
                h_rad_grd = dgngrp.interpol.rad_grd
                h_lat_grd = dgngrp.interpol.lat_grd
                h_long_grd = dgngrp.interpol.long_grd
            else
                h_rad_grd = Array(dgngrp.interpol.rad_grd)
                h_lat_grd = Array(dgngrp.interpol.lat_grd)
                h_long_grd = Array(dgngrp.interpol.long_grd)
            end
            dims = OrderedDict(
                "rad" => h_rad_grd,
                "lat" => h_lat_grd,
                "long" => h_long_grd,
            )
        else
            error("Unsupported interpolation topology $(dgngrp.interpol)")
        end
    else
        error("Dump of non-interpolated data currently unsupported")
    end

    return dims
end


"""
    atmos_default_GCM_collect(bl, currtime)

    Master function that performs a global grid traversal to compute various
    diagnostics using the above functions.
"""
function atmos_default_GCM_collect(dgngrp::DiagnosticsGroup, currtime)
    DA = CLIMA.array_type()
    mpicomm = Settings.mpicomm
    dg = Settings.dg
    Q = Settings.Q
    mpirank = MPI.Comm_rank(mpicomm)
    current_time = string(currtime)

    # make sure this time step is not already recorded
    dprefix = @sprintf(
        "%s_%s-%s-num%04d",
        dgngrp.out_prefix,
        dgngrp.name,
        Settings.starttime,
        dgngrp.num
    )
    dfilename = joinpath(Settings.output_dir, dprefix)
    docollect = [false]
    if mpirank == 0
        dfullname = full_name(dgngrp.writer, dfilename)
        if isfile(dfullname)
            @warn """
            Diagnostics $(dgngrp.name) collection
            output file $dfullname exists
            skipping collection at $current_time"""
        else
            docollect[1] = true
        end
    end
    MPI.Bcast!(docollect, 0, mpicomm)
    if !docollect[1]
        return nothing
    end

    # extract grid information
    bl = dg.balancelaw
    grid = dg.grid
    topology = grid.topology
    N = polynomialorder(grid)
    Nq = N + 1
    Nqk = dimensionality(grid) == 2 ? 1 : Nq
    npoints = Nq * Nq * Nqk
    nrealelem = length(topology.realelems)
    nvertelem = topology.stacksize
    nhorzelem = div(nrealelem, nvertelem)

    # get the state and geo variables onto the host if needed
    if Array ∈ typeof(Q).parameters
        localQ = Q.realdata
        localaux = dg.auxstate.realdata
        localvgeo = grid.vgeo
    else
        localQ = Array(Q.realdata)
        localaux = dg.auxstate.realdata
        localvgeo = Array(grid.vgeo)
    end
    FT = eltype(localQ)
    Nel = size(localQ, 3) # # of spectral elements
    Npl = size(localQ, 1) # # of dof per element
    nstate = num_state(bl, FT)

    # Initiate local variables (e.g. thermoQ - obtained from elementwise)
    #thermoQ = [zeros(FT, num_thermo(FT)) for _ in 1:npoints, _ in 1:nrealelem]
    thermoQ =DA{FT}(undef, Npl, num_thermo(FT), Nel)

    # nonlocal vars
    # e.g. Relative vorticity (NB: computationally intensive, so on gpu)
    #dynQ = [zeros(FT, num_dyn(FT)) for _ in 1:npoints, _ in 1:nrealelem]
    dynQ =DA{FT}(undef, Npl, num_dyn(FT), Nel)
    vgrad = compute_vec_grad(Settings.dg.balancelaw, Settings.Q, Settings.dg)
    vort_all = compute_vorticity(Settings.dg, vgrad)

    # compute thermo and dynamical vars element-wise
    @visitQ nhorzelem nvertelem Nqk Nq begin
        state = extract_state(dg, localQ, ijk, e)
        aux = extract_aux(dg, localaux, ijk, e)

        compute_thermo!(bl, state, aux, ijk, e, thermoQ)
        compute_dyn!(ijk, e, dynQ, Array(vort_all.data)[ijk, : ,e])
    end

    # interpolate and project variables onto a sphere
    all_state_data = nothing
    all_thermo_data = nothing
    all_dyn_data = nothing
    if dgngrp.interpol !== nothing
        # interpolate the state, thermo and dyn vars to sphere (u and vorticity need projection to zonal, merid)
        istate = DA(Array{FT}(undef, dgngrp.interpol.Npl, num_state(bl, FT))) # empty on cubed sphere structure (here, Npl = # of interpolation points on the local process)
        interpolate_local!(dgngrp.interpol, localQ, istate)

        ithermo = DA(Array{FT}(undef, dgngrp.interpol.Npl, num_thermo(FT))) # empty on cubed sphere structure
        interpolate_local!(dgngrp.interpol, thermoQ, ithermo)

        idyn = DA(Array{FT}(undef, dgngrp.interpol.Npl, num_dyn(FT) )) # empty on cubed sphere structure

        interpolate_local!(dgngrp.interpol, dynQ , idyn)

        if dgngrp.project
            if dgngrp.interpol isa InterpolationCubedSphere
                # TODO: get indices here without hard-coding them
                _ρu, _ρv, _ρw = 2, 3, 4
                project_cubed_sphere!(dgngrp.interpol, istate, (_ρu, _ρv, _ρw))
                _Ω₁, _Ω₂, _Ω₃ = 1, 2, 3
                project_cubed_sphere!(dgngrp.interpol, idyn, (_Ω₁, _Ω₂, _Ω₃))
            else
                error("Can only project for InterpolationCubedSphere")
            end
        end

        all_state_data =
            accumulate_interpolated_data(mpicomm, dgngrp.interpol, istate)

        all_thermo_data =
            accumulate_interpolated_data(mpicomm, dgngrp.interpol, ithermo)

        all_dyn_data =
            accumulate_interpolated_data(mpicomm, dgngrp.interpol, idyn)
    else
        error("Dump of non-interpolated data currently unsupported")
    end

    # combine state, thermo and dyn variables, and their manioulations on the interpolated grid

    nlon = size(all_thermo_data,1) # actually diff order [lon, lat, rad] - but doesnt matter for now
    nlat = size(all_thermo_data,2)
    nrad = size(all_thermo_data,3)

    #dsumsi = [zeros(FT, num_diagnostic(FT)) for  _ in 1:nlon, _ in 1:nlat, _ in 1:nrad]
    dsumsi =  DA(Array{FT}(undef, nlon, nlat, nrad, num_diagnostic(FT)))

    @visitIQ nlon nlat nrad begin
        compute_diagnosticsums_GCM!(all_state_data, all_thermo_data, all_dyn_data, dsumsi, lo, la, le)
    end

    # attribute names to the vars in dsumsi and collect in a dict
    varvals = OrderedDict()
    varnames = flattenednames(vars_diagnostic(FT))
    for vari in 1:length(varnames)
        varvals[varnames[vari]] = dsumsi[:,:,:,vari]
    end


    # get dimensions for the interpolated grid
    dims = get_dims(dgngrp)

    # write diagnostics
    write_data(dgngrp.writer, dfilename, dims, varvals, currtime)

    return nothing
end # atmos_default_GCM_collect

function atmos_default_GCM_fini(dgngrp::DiagnosticsGroup, currtime) end
