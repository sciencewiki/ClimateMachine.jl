# # Dry atmosphere GCM Config filter
# 
# This file computes selected diagnostics for the GCM and outputs them on the
# spherical interpolated diagnostic grid.
#
# It is called by setting ```dgn_config``` in the GCM run file
# (e.g. heldsuarez.jl) to ```setup_atmos_default_GCM_diagnostics```

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

# Set up thermodynamic vars (user-selected).
# These must be names in ```diagnostic_vars_GCM.jl`` and calculated in ```compute_thermo!```
function vars_thermo(FT)
    @vars begin
        T::FT
        θ_dry::FT
    end
end
num_thermo(FT) = varsize(vars_thermo(FT))
thermo_vars(array) = Vars{vars_thermo(eltype(array))}(array)

# Compute thermodynamic vars
function compute_thermo!(bl, state, auxstate, thermoQ, dgngrp ,FT)
    bl = Settings.dg.balancelaw
    FTa = eltype(auxstate)
    FTs = eltype(state)

    ts = thermo_state( bl, Vars{vars_state(bl, FTs)}(state[:,:,:]), Vars{vars_aux(bl, FTa)}(auxstate[:,:,:]) ) # thermodynamic state
    th = thermo_vars(thermoQ) # init thermoQ
    th.T = air_temperature(ts)
    th.θ_dry = dry_pottemp(ts)

    return nothing
end

# Set up dynamic vars (user-selected).
# These must be names in ```diagnostic_vars_GCM.jl`` and calculated in ```compute_dyn!```
function vars_dyn(FT)
    @vars begin
        vort_rel::SVector{3, FT}
    end
end
num_dyn(FT) = varsize(vars_dyn(FT))
dyn_vars(array) = Vars{vars_dyn(eltype(array))}(array)

# Compute dynamic vars
function compute_dyn!(bl, state, aux, dynQ, dgngrp, FT)
    #ts = thermo_state(bl, state, aux) # thermodynamic state
    dy = dyn_vars(dynQ) # init dynQ

    # Relative vorticity (NB: computationally intensive, so on gpu)
    vgrad = compute_vec_grad(Settings.dg.balancelaw, Settings.Q, Settings.dg)
    vort_all = compute_vorticity(Settings.dg, vgrad)

    @info @sprintf("""vort_all.data %s""", size(vort_all.data))
    @info @sprintf("""dynq %s""", size(dynQ))
    dy.vort_rel = vort_all.data

    return nothing
end

# set up struct for interpolated state vars
function vars_statei(FT)
    @vars begin
        ρ::FT
        ρu::SVector{3, FT}
        ρe::FT
    end
end
statei_vars(array) = Vars{vars_statei(eltype(array))}(array)

# aggregate all diags, including calculations done after interpolation to sphere
function compute_diagnosticsums_GCM!(
    all_state_data,
    all_thermo_data,
    all_dyn_data,
    dsumsi,
    )

    # set up variable structs with empty arrays
    state_i  = statei_vars(all_state_data)
    th_i = thermo_vars(all_thermo_data)
    dyn_i = dyn_vars(all_dyn_data)

    ds = diagnostic_vars(dsumsi)

    # calculate ds. vars that will be saved (these need to be selected from the diagnostic_vars(_GCM) file)
    ds.u = state_i.ρu[1] / state_i.ρ
    ds.v = state_i.ρu[2] / state_i.ρ
    ds.w = state_i.ρu[3] / state_i.ρ

    ds.T     = th_i.T
    ds.thd = th_i.θ_dry
    ds.vort_rel = dyn_i.vort_rel[3]
    #ds.vort_rel = all_dyn_data[3] # temporary fix

    # zonal means
    #@info @sprintf("""size u is %s""", size(ds.u[:,:,:]))
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

    @info @sprintf("""localQ %s""", size(localQ))

    # Compute and aggregate local thermo variables
    thermoQ = DA{FT}(undef, Npl, num_thermo(FT), Nel)
    compute_thermo!(bl, Q, dg.auxstate, thermoQ, dgngrp, FT)

    # Compute and aggregate local dynamic variables
    dynQ = DA{FT}(undef, Npl, num_dyn(FT), Nel)
    compute_dyn!(bl, Q, dg.auxstate, dynQ, dgngrp, FT)
    # this is a temporary fix - better to use a struct using compute_dyn!
    #vgrad = compute_vec_grad(Settings.dg.balancelaw, Settings.Q, Settings.dg)
    #vort_all = compute_vorticity(Settings.dg, vgrad)
    #dynQ =  Array(vort_all.data)

    @info @sprintf("""Ignre this %s""", size(dynQ))
    @info @sprintf("""num_dyn %s""", num_dyn(FT))

    # interpolate and project local variables onto a sphere
    all_state_data = nothing
    all_thermo_data = nothing
    if dgngrp.interpol !== nothing
        # interpolate the state, thermo and dyn vars to sphere (u and vorticity need projection to zonal, merid)
        istate = DA(Array{FT}(undef, dgngrp.interpol.Npl, num_state(bl, FT))) # empty on interpolated grid (here, Npl = # of interpolation points on the local process)
        interpolate_local!(dgngrp.interpol, localQ, istate)

        ithermo = DA(Array{FT}(undef, dgngrp.interpol.Npl, num_thermo(FT))) # empty on interpolated grid
        interpolate_local!(dgngrp.interpol, thermoQ, ithermo)

        idyn = DA(Array{FT}(undef, dgngrp.interpol.Npl, 3 )) # empty on interpolated grid
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
    dsumsi =  DA(Array{FT}(undef, size(all_state_data,1), size(all_state_data,2), size(all_state_data,3), num_diagnostic(FT))) # sets up the output array (as in diagnostic_vars_GCM.jl)
    compute_diagnosticsums_GCM!(all_state_data, all_thermo_data, all_dyn_data, dsumsi)

    @info @sprintf("""all_state_data: %s""", size(all_state_data))
    @info @sprintf("""dsumsi %s""", size(dsumsi))

    # get dimensions for the interpolated grid
    dims = get_dims(dgngrp)

    # collect all vars into one Dict
    varvals = OrderedDict()
    varnames = flattenednames(vars_diagnostic(FT))
    for i in 1:length(varnames)
        varvals[varnames[i]] = dsumsi[:, :, :, i]
    end

    # write diagnostics
    write_data(dgngrp.writer, dfilename, dims, varvals, currtime)

    return nothing
end # atmos_default_GCM_collect

function atmos_default_GCM_fini(dgngrp::DiagnosticsGroup, currtime) end
