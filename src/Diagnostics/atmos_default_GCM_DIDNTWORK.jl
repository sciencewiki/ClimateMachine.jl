# # Dry Atmosphere GCM Config File
# 
# This file computes selected diagnostics for the GCM and outputs them on the
# spherical interpolated diagnostic grid.
#
# It is called by setting ```dgn_config``` in the GCM run file
# (e.g. heldsuarez.jl) to ```setup_atmos_default_GCM_diagnostics```
#
# TODO:
# - fix compute_dyn! [weird dimensions error] - temporary fix for now
# - add more GCM vars following the vorticity template
# - enable zonal means and calculation of covatiances (commented out now)
# - do mass weighting of vars

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
        T::Array{FT,2}
        θ_dry::Array{FT,2}
    end
end
num_thermo(FT) = varsize(vars_thermo(FT),2)
thermo_vars(array) = Vars{vars_thermo(eltype(array))}((array[:,1,:], array[:,2,:]))

# Compute thermodynamic vars
function compute_thermo!(bl, state, auxstate, thermoQ, dgngrp ,FT)
    bl = Settings.dg.balancelaw
    FTa = eltype(auxstate)
    FTs = eltype(state)

    ts = thermo_state( bl, Vars{vars_state(bl, FTs)}(state[:,:,:]), Vars{vars_aux(bl, FTa)}(auxstate[:,:,:]) ) # thermodynamic state
    th = thermo_vars(Array(thermoQ)) # init thermoQ
    th.T = air_temperature(ts)
    th.θ_dry = dry_pottemp(ts)

    @info @sprintf("""size(th.θ_dry) %s""", size(th.θ_dry))
    @info @sprintf("""typeof(th.θ_dry)  %s""", typeof(th.θ_dry) )
    return nothing
end

# Set up dynamic vars (user-selected).
# These must be names in ```diagnostic_vars_GCM.jl`` and calculated in ```compute_dyn!```
function vars_dyn(FT)
    @vars begin
        Ω₁::Array{FT,3}
        Ω₂::Array{FT,3}
        Ω₃::Array{FT,3}
    end
end
num_dyn(FT) = varsize(vars_dyn(FT))
dyn_vars(array) = Vars{vars_dyn(eltype(array))}((squeeze(array[:,1,:],2), squeeze(array[:,2,:],2),squeeze(array[:,3,:],2)))

# Compute dynamic vars
function compute_dyn!(bl, state, aux, dynQ,dgngrp,FT)
    #ts = thermo_state(bl, state, aux) # thermodynamic state
    dy = dyn_vars(dynQ) # init dynQ

    # Relative vorticity (NB: computationally intensive, so on gpu)
    vgrad = compute_vec_grad(Settings.dg.balancelaw, Settings.Q, Settings.dg)
    vort_all = compute_vorticity(Settings.dg, vgrad)
    dy.vort_rel = Array(vort_all.data)

    return nothing
end

# set up struct for interpolated state vars
function vars_statei(FT)
    @vars begin
        ρ::Array{FT,3}
        ρu::Array{FT,3}
        ρv::Array{FT,3}
        ρw::Array{FT,3}
        ρe::Array{FT,3}
    end
end
statei_vars(array) = Vars{vars_statei(eltype(array[1]))}((array[:,:,:,1],array[:,:,:,2],array[:,:,:,3],array[:,:,:,4],array[:,:,:,5]))

function vars_thermoi(FT)
    @vars begin
        T::Array{FT,3}
        θ_dry::Array{FT,3}
    end
end
thermoi_vars(array) = Vars{vars_thermoi(eltype(array))}((array[:,:,:,1], array[:,:,:,2]))

function vars_dyni(FT)
    @vars begin
        Ω₁::Array{FT,3}
        Ω₂::Array{FT,3}
        Ω₃::Array{FT,3}
    end
end
dyni_vars(array) = Vars{vars_dyni(eltype(array))}(array)

# aggregate all diags, including calculations done after interpolation to sphere
function compute_diagnosticsums_GCM!(
    all_state_data,
    all_thermo_data,
    all_dyn_data,
    dsumsi,
    )

    # set up empty variable structs with interpolated arrays as templates
    #state_i  = statei_vars(Array([all_state_data[1],all_state_data[2],all_state_data[3],all_state_data[4],all_state_data[5]]))
    #th_i = thermo_vars(all_thermo_data)
    #dyn_i = dyn_vars(all_dyn_data)

    state_i = statei_vars(all_state_data)
    th_i = thermoi_vars(all_thermo_data)
    dyn_i = dyni_vars(all_thermo_data)

    # calculate ds. vars that will be saved (these need to be selected from the diagnostic_vars(_GCM) file)
    @info @sprintf("""vars_statei(FT) %s""", vars_statei(Float32) )
    @info @sprintf("""typeof(state_i.ρ) %s""", typeof(state_i.ρ) )
    @info @sprintf("""typeof(state_i.ρu) %s""", typeof(state_i.ρu) )
    @info @sprintf("""typeof(Array(state_i.ρu)) %s""", typeof(Array(state_i.ρu)) )
    ds = diagnostic_vars(dsumsi)
    ds.u = state_i.ρu ./ state_i.ρ
    ds.v = state_i.ρv ./ state_i.ρ
    ds.w = state_i.ρw ./ state_i.ρ

    @info @sprintf("""typeof(all_state_data) %s""", typeof(all_state_data) )
    @info @sprintf("""typeof(state_i.ρ) %s""", typeof(state_i.ρ) )
    @info @sprintf("""typeof(state_i.ρu[2]) %s""", typeof(state_i.ρu[2]) )
    @info @sprintf("""typeof(state_i.ρu) %s""", typeof(state_i.ρu) )
    @info @sprintf("""typeof(all_dyn_data) %s""", typeof(all_dyn_data) )
    ds.T     = th_i.T
    ds.thd = th_i.θ_dry

    rv_struct = dyn_vertvor_vars(Array(all_dyn_data))
    ds.vortrel = rv_struct.Ω₃ .*1000000.

    #ds.vortrel = Array(all_dyn_data[:,:,:,3])
    @info @sprintf("""size(ds.thd) %s""", size(ds.thd))
    @info @sprintf("""size(th_i.θ_dry) %s""", size(th_i.θ_dry))

    @info @sprintf("""size(ds.vortrel) %s""", size(ds.vortrel))
    @info @sprintf("""size(rv_struct.vort_rel) %s""", size(rv_struct.vort_rel))
    @info @sprintf("""size(Array(all_dyn_data[:,:,:,3])) %s""", size(Array(all_dyn_data[:,:,:,3])))
    #ds.vort_rel = dyn_i.vort_rel[3] #1val

    @info @sprintf("""ds.u %s""", ds.u)

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

    @info @sprintf("""size(localQ) %s""", size(localQ))
    @info @sprintf("""size(localaux) %s""", size(localaux))

    nstate = num_state(bl, FT)

    # Compute and aggregate local thermo variables
    thermoQ = DA{FT}(undef, Npl, num_thermo(FT), Nel)
    compute_thermo!(bl, localQ, localaux, thermoQ, dgngrp, FT)

    @info @sprintf("""size(thermoQ) %s""", size(thermoQ))
    # Compute and aggregate local dynamic variables
    #dynQ = DA{FT}(undef, Npl, 3, Nel)
    #compute_dyn!(bl, Q, dg.auxstate, dynQ, dgngrp, FT)

    # this is a temporary fix - better to use a struct using compute_dyn!
    vgrad = compute_vec_grad(Settings.dg.balancelaw, Settings.Q, Settings.dg)
    vort_all = compute_vorticity(Settings.dg, vgrad)
    dynQ =  Array(vort_all.data)

    @info @sprintf(""" size(dynQ) %s""", size(dynQ))
    #@info @sprintf("""num_dyn %s""", num_dyn(FT))

    # interpolate and project local variables onto a sphere
    all_state_data = nothing
    all_thermo_data = nothing
    if dgngrp.interpol !== nothing
        # interpolate the state, thermo and dyn vars to sphere (u and vorticity need projection to zonal, merid)
        istate = DA(Array{FT}(undef, dgngrp.interpol.Npl, num_state(bl, FT))) # empty on cubed sphere structure (here, Npl = # of interpolation points on the local process)
        interpolate_local!(dgngrp.interpol, localQ, istate)

        ithermo = DA(Array{FT}(undef, dgngrp.interpol.Npl, num_thermo(FT))) # empty on cubed sphere structure
        interpolate_local!(dgngrp.interpol, thermoQ, ithermo)

        idyn = DA(Array{FT}(undef, dgngrp.interpol.Npl, 3 )) # empty on cubed sphere structure
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

    @info @sprintf("""size(all_state_data): %s""", size(all_state_data))
    @info @sprintf("""size(all_thermo_data): %s""", size(all_thermo_data))
    @info @sprintf("""size(all_aux_data): %s""", size(all_dyn_data))
    @info @sprintf("""size(all_dyn_data[:,:,:,3]) %s""", size(all_dyn_data[:,:,:,3]))
    @info @sprintf("""all_dyn_data[3] %s""", all_dyn_data[3])

    @info @sprintf("""dsumsi size %s""", size(dsumsi))
    @info @sprintf("""sum(dsumsi[:, :, :, 1]) %s""", sum(dsumsi[:, :, :, 1]))
    @info @sprintf("""sum(dsumsi[:, :, :, 6]) %s""", sum(dsumsi[:, :, :, 6]))
    @info @sprintf("""dsumsi[:, :, 5, 1] %s""", dsumsi[:, :, 5, 1])
    @info @sprintf("""typeof(dsumsi) %s""", typeof(dsumsi) )
    #@info @sprintf("""all_dyn_data %s""", all_dyn_data[:, :, :, 3]) # full arrays e-6 vals

    # get dimensions for the interpolated grid
    dims = get_dims(dgngrp)

    # collect all vars into one Dict
    varvals = OrderedDict()
    varnames = flattenednames(vars_diagnostic(FT))
    for i in 1:length(varnames)
        varvals[varnames[i]] = dsumsi[:, :, :, i]
        @info @sprintf("""varname %s""", varnames[i])
    end

    # write diagnostics
    write_data(dgngrp.writer, dfilename, dims, varvals, currtime)

    return nothing
end # atmos_default_GCM_collect

function atmos_default_GCM_fini(dgngrp::DiagnosticsGroup, currtime) end
