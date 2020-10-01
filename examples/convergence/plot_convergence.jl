using PyPlot
using CSV
using Printf

function mean_convergence_rate(dx,err)
    slope = diff(log.(err)) ./ diff(log.(dx))
    avgslope = sum(slope)/length(slope)
    return avgslope
end

function plot_convergence(polyorder,component)
    filename = "examples/convergence_polyorder_"*string(polyorder)*".csv"
    df = CSV.read(filename,DataFrame)
    dx = 1.0 ./ df[:,:NumElmts]
    err = df[:,component]
    fig,ax = PyPlot.subplots()
    ax.loglog(dx,err,"-o")
    ax.grid()
    rate = mean_convergence_rate(dx,err)
    annotation = @sprintf "Rate = %1.2f" rate
    ax.annotate(annotation,(0.5,0.3),xycoords = "axes fraction")
    filename = "examples/poly"*string(polyorder)*"_"*string(component)*".png"
    fig.savefig(filename)
end

plot_convergence(1,:ErrorU1)
plot_convergence(1,:ErrorU2)

plot_convergence(2,:ErrorU1)
plot_convergence(2,:ErrorU2)

plot_convergence(3,:ErrorU1)
plot_convergence(3,:ErrorU2)

plot_convergence(4,:ErrorU1)
plot_convergence(4,:ErrorU2)
