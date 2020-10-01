using CSV, DataFrames, Printf
using PyPlot

function plot_error_vs_position(position, err; title = "", filename = "")
    fig, ax = PyPlot.subplots()
    ax.plot(position / 2.0, err, linewidth = 2)
    ax.set_title(title)
    ax.grid()
    ax.set_xlabel("Normalized interface position")
    ax.set_ylabel("Normalized error")
    if length(filename) > 0
        fig.savefig(filename)
    else
        return fig
    end
end

function plot_errors_vs_position(positions, err, labels; filename = "")
    fig, ax = PyPlot.subplots()
    for (p, e, l) in zip(positions, err, labels)
        ax.plot(p / 2.0, e, linewidth = 2, label = l)
    end
    ax.grid()
    ax.set_ylabel("Normalized error")
    ax.set_xlabel("Normalized interface position")
    ax.legend()
    if length(filename) > 0
        fig.savefig(filename)
    else
        return fig
    end
end

function mean_log_slope(x, y)
    rate = diff(log.(y)) ./ diff(log.(x))
    return sum(rate) / length(rate)
end

function plot_errors_vs_penalties(penalties, err; title = "", filename = "")
    fig, ax = PyPlot.subplots()
    ax.plot(penalties, err, linewidth = 2)
    ax.set_title(title)
    # slope = mean_log_slope(penalties,err)
    # annotation = @sprintf "Mean slope = %1.2f" slope
    # ax.annotate(annotation,(0.2,0.2),xycoords = "axes fraction")
    ax.grid()
    ax.set_xlabel("Penalty parameter")
    ax.set_ylabel("Normalized error")
    if length(filename) > 0
        fig.savefig(filename)
    else
        return fig
    end
end

function plot_condition_vs_position(positions, cond; title = "", filename = "")
    fig, ax = PyPlot.subplots()
    ax.plot(positions / 2.0, cond, linewidth = 2)
    ax.set_title(title)
    # slope = mean_log_slope(penalties,err)
    # annotation = @sprintf "Mean slope = %1.2f" slope
    # ax.annotate(annotation,(0.2,0.2),xycoords = "axes fraction")
    ax.grid()
    ax.set_xlabel("Normalized interface position")
    ax.set_ylabel("Condition number")
    if length(filename) > 0
        fig.savefig(filename)
    else
        return fig
    end
end

condfile = "examples/interface/condition-number-vs-position.csv"
df = CSV.read(condfile, DataFrame)
plot_condition_vs_position(
    df[:, :position],
    df[:, :condition],
    title = "Condition no. vs. interface position for P = 1000",
    filename = "examples/interface/cond-vs-position.png",
)


# filename1 = "examples/interface/plane_theta-0-poly-1-penalty-1000.csv"
# df1 = CSV.read(filename1, DataFrame)
# filename2 = "examples/interface/plane_theta-0-poly-1-penalty-100.csv"
# df2 = CSV.read(filename2, DataFrame)
# filename3 = "examples/interface/plane_theta-0-poly-1-penalty-10.csv"
# df3 = CSV.read(filename3, DataFrame)
#
# penaltyfile = "examples/interface/error-vs-penalty.csv"
# dfp = CSV.read(penaltyfile, DataFrame)
# plot_errors_vs_penalties(
#     dfp[:, :penalty],
#     dfp[:, :ErrorU1],
#     title = "Error vs. penalty parameter",
#     filename = "examples/interface/error-vs-penalty.png",
# )

# plot_error_vs_position(
#     df1[:, :position],
#     df1[:, :ErrorU1],
#     title = "Error vs. interface position for P = 1000",
#     filename = "examples/interface/error-v-position-P-1000.png"
# )
