using PyPlot
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("../../test/useful_routines.jl")

function plot_elements!(ax, mesh)
    ncells = CutCell.number_of_cells(mesh)
    for cellid = 1:ncells
        cellmap = CutCell.cell_map(mesh, cellid)
        yL = cellmap.yL
        yR = cellmap.yR
        cellwidth = yR - yL
        patch = matplotlib.patches.Rectangle(
            (yL[1], yL[2]),
            cellwidth[1],
            cellwidth[2],
            ec = "black",
            fill = false,
        )
        ax.add_patch(patch)
    end
end

function interpolate_at_reference_points(
    refpoints,
    levelset,
    levelsetcoeffs,
    mesh,
)
    interpolatedvals = zeros(0)
    ncells = CutCell.number_of_cells(mesh)
    numrefpoints = size(refpoints)[2]
    for cellid = 1:ncells
        nodeids = CutCell.nodal_connectivity(mesh, cellid)
        update!(levelset, levelsetcoeffs[nodeids])
        append!(interpolatedvals, vec(mapslices(levelset, refpoints, dims = 1)))
    end
    return interpolatedvals
end

function map_reference_points_to_spatial(refpoints,mesh)
    spatialpoints = zeros(2,0)
    ncells = CutCell.number_of_cells(mesh)
    numrefpoints = size(refpoints)[2]
    for cellid = 1:ncells
        cellmap = CutCell.cell_map(mesh,cellid)
        spatialpoints = hcat(spatialpoints,cellmap(refpoints))
    end
    return spatialpoints
end

polyorder = 2
nelmts = 3
width = 1.0
delta = 1e-2width

interfacecenter = [width / 2, width / 2]
dx = width / nelmts
inradius = dx / sqrt(2) + delta

basis = TensorProductBasis(2, polyorder)
mesh = CutCell.Mesh([0.0, 0.0], [width, width], [nelmts, nelmts], basis)
levelset = InterpolatingPolynomial(1, basis)
levelsetcoeffs = CutCell.levelset_coefficients(
    x -> -circle_distance_function(x, interfacecenter, inradius),
    mesh,
)

refrange = -1:1e-1:1
refpoints = vcat(
    repeat(refrange, outer = length(refrange))',
    repeat(refrange, inner = length(refrange))',
)

levelsetvals =
    interpolate_at_reference_points(refpoints, levelset, levelsetcoeffs, mesh)
spatialpoints = map_reference_points_to_spatial(refpoints,mesh)


fig, ax = PyPlot.subplots()
ax.tricontour(
    spatialpoints[1, :],
    spatialpoints[2, :],
    levelsetvals,
    levels = [0.0],
)
plot_elements!(ax, mesh)
ax.set_xlim(0,1)
ax.set_ylim(0,1)
# ax.set_xlim(0.65, 0.75)
# ax.set_ylim(0.7, 0.8)
ax.set_aspect("equal")
fig
folderpath = "examples/circular-interface-pressure/"
fig.savefig(folderpath * "coarse-mesh.png")
