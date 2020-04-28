
using CLIMA.VariableTemplates

function vars_diagnostic(FT)
    @vars begin
        # state and functions of state
        u::FT
        v::FT
        w::FT
<<<<<<< HEAD
        T::FT                  # air temperature
=======
        T::FT                   # air temperature
>>>>>>> 50668a47431717065f64a2e327a86dfb8a3401a4
        #q_tot::FT
        #e_tot::FT
        #q_liq::FT
        thd::FT                 # θ
        #thl::FT                # θ_liq
        #thv::FT                # θ_v
        #e_int::FT
        #h_m::FT
        #h_t::FT
        #qt_sgs::FT
        #ht_sgs::FT
<<<<<<< HEAD
=======

>>>>>>> 50668a47431717065f64a2e327a86dfb8a3401a4
        # vertical fluxes
        #vert_eddy_mass_flux::FT # <w′ρ′>
        #vert_eddy_u_flux::FT    # <w′u′>
        #vert_eddy_v_flux::FT    # <w′v′>
        #vert_eddy_qt_flux::FT   # <w'q_tot'>
        #vert_qt_flux::FT        # <w q_tot>
        #vert_eddy_ql_flux::FT   # <w′q_liq′>
        #vert_eddy_qv_flux::FT   # <w′q_vap′>
        #vert_eddy_thd_flux::FT  # <w′θ′>
        #vert_eddy_thv_flux::FT  # <w′θ_v′>
        #vert_eddy_thl_flux::FT  # <w′θ_liq′>
<<<<<<< HEAD
        # zonal means
        #u_zm::FT                 # zonal mean zonal wind
        #T_zm::FT                 # zonal mean air temperature
        # covariances
        #uvcovariance::FT          # u*v*
        #vtcovariance::FT          # v*T*
        # eddy kinetic energy
        #EKE::FT                   # 1/2 (u**2 +v**2)
        # other sample vars for now
        vortrel::FT               # relative vorticity
=======

        # zonal means
        #u_zm::FT                 # zonal mean zonal wind
        #T_zm::FT                 # zonal mean air temperature

        # covariances
        #uvcovariance::FT          # u*v*
        #vtcovariance::FT          # v*T*

        # eddy kinetic energy
        #EKE::FT                   # 1/2 (u**2 +v**2)

        # other sample vars for now
        vort_rel::FT               # relative vorticity
>>>>>>> 50668a47431717065f64a2e327a86dfb8a3401a4
        #streamf_hor::FT             # horizontal streamfunction

    end
end

num_diagnostic(FT) = varsize(vars_diagnostic(FT))
diagnostic_vars(array) = Vars{vars_diagnostic(eltype(array))}(array)
