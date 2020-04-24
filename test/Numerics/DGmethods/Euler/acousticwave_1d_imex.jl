using CLIMA
using CLIMA.ConfigTypes
using CLIMA.Mesh.Topologies: StackedCubedSphereTopology, cubedshellwarp, grid1d
using CLIMA.Mesh.Grids:
    DiscontinuousSpectralElementGrid,
    VerticalDirection,
    HorizontalDirection,
    min_node_distance
using CLIMA.Mesh.Filters
using CLIMA.DGmethods: DGModel, init_ode_state
using CLIMA.DGmethods.NumericalFluxes:
    Rusanov, CentralNumericalFluxGradient, CentralNumericalFluxDiffusive
using CLIMA.ODESolvers
using CLIMA.GeneralizedMinimalResidualSolver
using CLIMA.ColumnwiseLUSolver: ManyColumnLU
using CLIMA.VTK: writevtk, writepvtu
using CLIMA.GenericCallbacks: EveryXWallTimeSeconds, EveryXSimulationSteps
using CLIMA.MoistThermodynamics:
    air_density,
    soundspeed_air,
    internal_energy,
    PhaseDry_given_pT,
    PhasePartition
using CLIMA.Atmos:
    AtmosModel,
    SphericalOrientation,
    DryModel,
    NoPrecipitation,
    NoRadiation,
    ConstantViscosityWithDivergence,
    vars_state,
    vars_aux,
    Gravity,
    HydrostaticState,
    IsothermalProfile,
    AtmosAcousticGravityLinearModel,
    AtmosAcousticLinearModel,
    altitude,
    latitude,
    longitude,
    gravitational_potential,
    RemainderModel
using CLIMA.VariableTemplates: flattenednames

using CLIMAParameters
using CLIMAParameters.Planet: planet_radius
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

using MPI, Logging, StaticArrays, LinearAlgebra, Printf, Dates, Test

const output_vtk = true

"""
    main()

Run this test problem
"""
function main()
    CLIMA.init()
    ArrayType = CLIMA.array_type()

    mpicomm = MPI.COMM_WORLD

    polynomialorder = 5
    numelem_horz = 10
    numelem_vert = 5

    # Real test should be run for 33 hour, which is approximate time for the
    # pulse to go around the whole sphere
    # but for CI we only run 1 hour
    timeend = 60 * 60 * 33

    # Do the output every hour
    outputtime = 60 * 60

    # Expected result is L2-norm of the final solution
    expected_result = Dict()
    expected_result[Float32] = 9.5064378310656000e+13
    expected_result[Float64] = 9.5073559883839516e+13

    @testset "acoustic wave" begin
        for FT in (Float64, )# Float32)
            result = run(
                mpicomm,
                polynomialorder,
                numelem_horz,
                numelem_vert,
                timeend,
                outputtime,
                ArrayType,
                FT,
            )
            # @test result ≈ expected_result[FT]
        end
    end
end

"""
    run(
        mpicomm,
        polynomialorder,
        numelem_horz,
        numelem_vert,
        timeend,
        outputtime,
        ArrayType,
        FT,
    )

Run the actual simulation.
"""
function run(
    mpicomm,
    polynomialorder,
    numelem_horz,
    numelem_vert,
    timeend,
    outputtime,
    ArrayType,
    FT,
)

    # Structure to pass around to setup the simulation
    setup = AcousticWaveSetup{FT}()

    # Create the cubed sphere mesh
    _planet_radius::FT = planet_radius(param_set)
    vert_range = grid1d(
        _planet_radius,
        FT(_planet_radius + setup.domain_height),
        nelem = numelem_vert,
    )
    topology = StackedCubedSphereTopology(mpicomm, numelem_horz, vert_range)

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = polynomialorder,
        meshwarp = cubedshellwarp,
    )
    hmnd = min_node_distance(grid, HorizontalDirection())
    vmnd = min_node_distance(grid, VerticalDirection())

    # This is the base model which defines all the data (all other DGModels
    # for substepping components will piggy-back off of this models data)
    fullmodel = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        orientation = SphericalOrientation(),
        ref_state = HydrostaticState(IsothermalProfile(setup.T_ref), FT(0)),
        turbulence = ConstantViscosityWithDivergence(FT(0)),
        moisture = DryModel(),
        source = Gravity(),
        init_state = setup,
    )
    dg = DGModel(
        fullmodel,
        grid,
        Rusanov(),
        CentralNumericalFluxDiffusive(),
        CentralNumericalFluxGradient(),
    )
    Q = init_ode_state(dg, FT(0))

    # The linear model which contains the fast modes
    # acousticmodel = AtmosAcousticLinearModel(fullmodel)
    acousticmodel = AtmosAcousticGravityLinearModel(fullmodel)

    # Vertical acoustic model will be handle with implicit time stepping
    lineardg = DGModel(
        acousticmodel,
        grid,
        Rusanov(),
        CentralNumericalFluxDiffusive(),
        CentralNumericalFluxGradient();
        direction = VerticalDirection(),
        auxstate = dg.auxstate,
    )

    # determine the time step
    element_size = (setup.domain_height / numelem_vert)
    acoustic_speed = soundspeed_air(fullmodel.param_set, FT(setup.T_ref))
    dt_factor = 445
    dt = 100
    # Adjust the time step so we exactly hit 1 hour for VTK output
    @show dt = 60 * 60 / ceil(60 * 60 / dt)
    nsteps = ceil(Int, timeend / dt)
    Q = init_ode_state(dg, FT(0))
    nsteps_output = ceil(outputtime / dt)

    linearsolver = ManyColumnLU()

    odesolver = ARK2GiraldoKellyConstantinescu(
        dg,
        lineardg,
        LinearBackwardEulerSolver(linearsolver; isadjustable = false),
        Q;
        dt = dt,
        t0 = 0,
        split_explicit_implicit = false,
    )

    filterorder = 18
    filter = ExponentialFilter(grid, 0, filterorder)
    cbfilter = EveryXSimulationSteps(1) do
        Filters.apply!(Q, 1:size(Q, 2), grid, filter, VerticalDirection())
        nothing
    end

    eng0 = norm(Q)
    @info @sprintf """Starting
                      ArrayType       = %s
                      FT              = %s
                      polynomialorder = %d
                      numelem_horz    = %d
                      numelem_vert    = %d
                      dt              = %.16e
                      norm(Q₀)        = %.16e
                      """ "$ArrayType" "$FT" polynomialorder numelem_horz numelem_vert dt eng0

    # Set up the information callback
    starttime = Ref(now())
    # cbinfo = EveryXWallTimeSeconds(60, mpicomm) do (s = false)
    cbinfo = EveryXSimulationSteps(nsteps_output) do (s = false)
        if s
            starttime[] = now()
        else
            energy = norm(Q)
            runtime = Dates.format(
                convert(DateTime, now() - starttime[]),
                dateformat"HH:MM:SS",
            )
            @info @sprintf """Update
                              simtime = %.16e
                              runtime = %s
                              norm(Q) = %.16e
                              """ gettime(odesolver) runtime energy
        end
    end
    callbacks = (cbinfo, cbfilter)

    # Setup the vtk callback
    if output_vtk
        # create vtk dir
        vtkdir =
            "vtk_acousticwave" *
            "_poly$(polynomialorder)_horz$(numelem_horz)_vert$(numelem_vert)" *
            "_dt$(dt_factor)x_$(ArrayType)_$(FT)"
        mkpath(vtkdir)

        vtkstep = 0
        # output initial step
        do_output(mpicomm, vtkdir, vtkstep, dg, Q, fullmodel)

        # setup the output callback
        cbvtk = EveryXSimulationSteps(floor(outputtime / dt)) do
            vtkstep += 1
            Qe = init_ode_state(dg, gettime(odesolver))
            do_output(mpicomm, vtkdir, vtkstep, dg, Q, fullmodel)
        end
        callbacks = (callbacks..., cbvtk)
    end

    # Solve the ode
    solve!(
        Q,
        odesolver;
        numberofsteps = nsteps,
        adjustfinalstep = false,
        callbacks = callbacks,
    )

    # final statistics
    engf = norm(Q)
    @info @sprintf """Finished
    norm(Q)                 = %.16e
    norm(Q) / norm(Q₀)      = %.16e
    norm(Q) - norm(Q₀)      = %.16e
    """ engf engf / eng0 engf - eng0
    engf
end

Base.@kwdef struct AcousticWaveSetup{FT}
    domain_height::FT = 10e3
    T_ref::FT = 300
    α::FT = 3
    γ::FT = 100
    nv::Int = 1
end

function (setup::AcousticWaveSetup)(bl, state, aux, coords, t)
    # callable to set initial conditions
    FT = eltype(state)

    λ = longitude(bl, aux)
    φ = latitude(bl, aux)
    z = altitude(bl, aux)

    β = min(FT(1), setup.α * acos(cos(φ) * cos(λ)))
    f = (1 + cos(FT(π) * β)) / 2
    g = sin(setup.nv * FT(π) * z / setup.domain_height)
    Δp = setup.γ * f * g
    p = aux.ref_state.p + Δp

    ts = PhaseDry_given_pT(bl.param_set, p, setup.T_ref)
    q_pt = PhasePartition(ts)
    e_pot = gravitational_potential(bl.orientation, aux)
    e_int = internal_energy(ts)

    state.ρ = air_density(ts)
    state.ρu = SVector{3, FT}(0, 0, 0)
    state.ρe = state.ρ * (e_int + e_pot)
    nothing
end

function do_output(
    mpicomm,
    vtkdir,
    vtkstep,
    dg,
    Q,
    model,
    testname = "acousticwave",
)
    ## name of the file that this MPI rank will write
    filename = @sprintf(
        "%s/%s_mpirank%04d_step%04d",
        vtkdir,
        testname,
        MPI.Comm_rank(mpicomm),
        vtkstep
    )

    statenames = flattenednames(vars_state(model, eltype(Q)))
    auxnames = flattenednames(vars_aux(model, eltype(Q)))
    writevtk(filename, Q, dg, statenames, dg.auxstate, auxnames)

    ## Generate the pvtu file for these vtk files
    if MPI.Comm_rank(mpicomm) == 0
        ## name of the pvtu file
        pvtuprefix = @sprintf("%s/%s_step%04d", vtkdir, testname, vtkstep)

        ## name of each of the ranks vtk files
        prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
            @sprintf("%s_mpirank%04d_step%04d", testname, i - 1, vtkstep)
        end

        writepvtu(pvtuprefix, prefixes, (statenames..., auxnames...))

        @info "Done writing VTK: $pvtuprefix"
    end
end

main()
