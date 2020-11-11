using PyPlot


function plot_on_circumference(angularposition, val; ylabel = "", title = "", ylims = [])
    fig, ax = PyPlot.subplots()
    ax.plot(angularposition, val)
    if length(ylims) > 0
        ax.set_ylim(ylims)
    end
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Angular position (deg)")
    ax.grid()
    fig.tight_layout()
    ax.set_title(title)
    return fig
end

function plot_stress_strain(
    angularposition,
    stresscomponent,
    straincomponent;
    ylabels = ["", ""],
    title = "",
)
    fig, ax = PyPlot.subplots(2, 1, sharex = true)
    ax[1].plot(angularposition, stresscomponent)
    ax[1].set_ylabel(ylabels[1])
    ax[1].grid()
    ax[2].plot(angularposition, straincomponent)
    ax[2].set_ylabel(ylabels[2])
    ax[2].grid()
    ax[2].set_xlabel("Angular position (deg)")
    fig.tight_layout()
    fig.suptitle(title)
    return fig
end

function plot_all_stress_strain(angularposition, stress, symmdispgrad, filename)
    fig11 = plot_stress_strain(
        angularposition,
        stress[1, :],
        symmdispgrad[1, :],
        ylabels = [L"\sigma_{11}", L"\epsilon_{11}"],
    )
    fig22 = plot_stress_strain(
        angularposition,
        stress[2, :],
        symmdispgrad[2, :],
        ylabels = [L"\sigma_{22}", L"\epsilon_{22}"],
    )
    fig12 = plot_stress_strain(
        angularposition,
        stress[3, :],
        symmdispgrad[3, :],
        ylabels = [L"\sigma_{12}", L"2\epsilon_{12}"],
    )

    s33 = stress[4, :]
    lowlim = 1.1min(0.0, minimum(s33))
    uplim = 1.1max(0.0, maximum(s33))
    fig33 = plot_on_circumference(
        angularposition,
        s33,
        ylabel = L"\sigma_{33}",
        ylims = [lowlim, uplim],
    )

    fig33.savefig(filename*"33.png")
    fig11.savefig(filename * "11.png")
    fig22.savefig(filename * "22.png")
    fig12.savefig(filename * "12.png")
end
