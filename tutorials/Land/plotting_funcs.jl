export_plot(all_data, ϕ_all, filename) = nothing
export_plot_snapshot(all_data, ϕ_all, filename) = nothing

# using Requires
# @init @require Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80" begin
#   using .Plots
"""
    plot_friendly_name(ϕ)

Get plot-friendly string, since many Unicode
characters do not render in plot labels.
"""
function plot_friendly_name(ϕ)
    s = ϕ
    s = replace(s, "ρ" => "rho")
    s = replace(s, "α" => "alpha")
    s = replace(s, "∂" => "partial")
    s = replace(s, "∇" => "nabla")
    return s
end

"""
    export_plot(all_data, ϕ_all, filename)

Export plot of all variables, or all
available time-steps in `all_data`.

TODO: Assumes `z` in `all_data`, maybe passing `grid`
`all_data` would be better, and removing `z` from `all_data`.
"""
function export_plot(all_data, ϕ_all, filename)
    ϕ_all isa Tuple || (ϕ_all = (ϕ_all,))
    p = plot()
    z = all_data[1]["z"][:]
    for n in 0:(length(keys(all_data)) - 1)
        for ϕ in ϕ_all
            ϕ_string = String(ϕ)
            ϕ_name = plot_friendly_name(ϕ_string)
            ϕ_data = all_data[n][ϕ_string][:]
            plot!(ϕ_data, z, xlabel = ϕ_name, ylabel = "z [cm]")
        end
    end
    savefig(filename)
end

function export_plot_snapshot(all_data, ϕ_all, filename)
    ϕ_all isa Tuple || (ϕ_all = (ϕ_all,))
    p = plot()
    z = all_data["z"][:]
    for ϕ in ϕ_all
        ϕ_string = String(ϕ)
        ϕ_name = plot_friendly_name(ϕ_string)
        ϕ_data = all_data[ϕ_string][:]
        plot!(ϕ_data, z, xlabel = ϕ_name, ylabel = "z [cm]")
    end
    savefig(filename)
end