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

function shrinking_corner_distance(x,xc,speed,time)
    dx = speed*time
    return corner_distance_function(x,xc-[dx,dx])
end

function final_interface_error(
    levelset,
    mesh,
    speed,
    dt,
    nsteps,
    numqp,
    xc,
)

    levelsetcoeffs = CutCell.levelset_coefficients(
        x -> shrinking_corner_distance(x, xc, speed, 0.0),
        mesh,
    )
    cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
    levelsetspeed = speed * ones(length(levelsetcoeffs))

    for t = 1:nsteps
        println(t)

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
    end
    dx = speed*nsteps*dt
    cornerlocation = xc - [dx,dx]
    exact_interface_length = sum(cornerlocation)

    err = interface_error(
        levelset,
        levelsetcoeffs,
        cutmesh,
        numqp,
        x -> shrinking_corner_distance(x,xc,speed,nsteps*dt),
    )/exact_interface_length

    return err
end

function grid_size(mesh)
    w = CutCell.widths(mesh)
    nn = CutCell.nodes_per_mesh_side(mesh)
    return w ./ (nn .- 1)
end

function time_step_size(speed, mesh; CFL = 0.5)
    dx = minimum(grid_size(mesh))
    return CFL * dx / abs(speed)
end

function numsteps(stoptime,dt)
    r = stoptime/dt
    @assert isinteger(r)
    return round(Int,r)
end

function error_for_numelmts(nelmts)
    x0 = [0.0, 0.0]
    L, W = 1.0, 1.0
    numghostlayers = 1
    polyorder = 2
    numqp = 3

    xc = [0.87, 0.87]
    speed = 1.0
    stoptime = 0.2

    basis = TensorProductBasis(2, polyorder)
    levelset = InterpolatingPolynomial(1, basis)

    mesh = CutCell.Mesh(x0, [L, W], [nelmts, nelmts], basis)
    dt = time_step_size(speed, mesh)
    nsteps = numsteps(stoptime,dt)

    return final_interface_error(
        levelset,
        mesh,
        speed,
        dt,
        nsteps,
        numqp,
        xc,
    )
end

# nelmts = [5,10,20,40,80]
err = error_for_numelmts(80)
# err = [error_for_numelmts(ne) for ne in nelmts]
