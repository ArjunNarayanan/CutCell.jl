using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("../../test/useful_routines.jl")

function cell_interface_error(quad, cellmap, facescale, distance_to_interface)
    err = 0.0
    for (idx, (p, w)) in enumerate(quad)
        x = cellmap(p)
        err += (distance_to_interface(x)[1])^2 * facescale[idx] * w
    end
    return err
end

function interface_error(levelset, levelsetcoeffs, cutmesh, numqp, distance_to_interface)
    interfacequads = CutCell.InterfaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
    cellids = findall(CutCell.cell_sign(cutmesh) .== 0)

    err = 0.0
    for cellid in cellids
        quad = interfacequads[+1, cellid]
        if length(quad) > 0
            normals = CutCell.interface_normals(interfacequads, cellid)
            cellmap = CutCell.cell_map(cutmesh, cellid)
            facescale = CutCell.scale_area(cellmap, normals)

            err += cell_interface_error(quad, cellmap, facescale, distance_to_interface)
        end
    end
    return sqrt(err)
end

function shrinking_circle_distance(x, xc, initialradius, speed, time)
    r = initialradius - speed * time
    return circle_distance_function(x, xc, r)
end

function interface_error_over_timesteps(
    levelset,
    mesh,
    speed,
    dt,
    nsteps,
    numqp,
    xc,
    initialradius,
)
    err = zeros(nsteps + 1)

    levelsetcoeffs = CutCell.levelset_coefficients(
        x -> shrinking_circle_distance(x, xc, initialradius, speed, 0.0),
        mesh,
    )
    cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)

    err[1] = interface_error(
        levelset,
        levelsetcoeffs,
        cutmesh,
        numqp,
        x -> shrinking_circle_distance(x, xc, initialradius, speed, 0.0),
    )/(2pi*initialradius)

    levelsetspeed = speed * ones(length(levelsetcoeffs))

    for t = 1:nsteps
        paddedmesh = CutCell.BoundaryPaddedMesh(cutmesh, 1)
        refseedpoints, spatialseedpoints, seedcellids =
            CutCell.seed_zero_levelset(2, levelset, levelsetcoeffs, cutmesh)
        paddedlevelset = CutCell.BoundaryPaddedLevelset(
            paddedmesh,
            refseedpoints,
            spatialseedpoints,
            seedcellids,
            levelset,
            levelsetcoeffs,
            cutmesh,
            1e-10,
        )
        levelsetcoeffs =
            CutCell.step_first_order_levelset(paddedlevelset, levelsetspeed, dt)

        cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
        err[t+1] = interface_error(
            levelset,
            levelsetcoeffs,
            cutmesh,
            numqp,
            x -> shrinking_circle_distance(x,xc,initialradius,speed,t*dt),
        )/(2pi*(initialradius - speed*t*dt))
    end
    return err
end

function grid_size(mesh)
    w = CutCell.widths(mesh)
    nn = CutCell.nodes_per_mesh_side(mesh)
    return w ./ (nn .- 1)
end

function time_step_size(speed, mesh; CFL = 0.9)
    dx = minimum(grid_size(mesh))
    return CFL * dx / abs(speed)
end



x0 = [0.0, 0.0]
L, W = 1.0, 1.0
nelmts = 10
numghostlayers = 1
polyorder = 2
numqp = 3

xc = [0.5, 0.5]
initialradius = 0.45
speed = 1.0
stoptime = 0.3


basis = TensorProductBasis(2, polyorder)
levelset = InterpolatingPolynomial(1, basis)

mesh = CutCell.Mesh(x0, [L, W], [nelmts, nelmts], basis)
dt = time_step_size(speed, mesh)
nsteps = ceil(Int, stoptime / dt)

err = interface_error_over_timesteps(
    levelset,
    mesh,
    speed,
    dt,
    nsteps,
    numqp,
    xc,
    initialradius,
)
