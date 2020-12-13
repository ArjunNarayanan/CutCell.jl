struct BoundaryPaddedMesh
    mesh::Any
    gridsize::Any
    nfmside::Any
    numghostlayers::Any
    bottomghostcoords::Any
    rightghostcoords::Any
    topghostcoords::Any
    leftghostcoords::Any
end

function uniform_grid_coordinates(x0, widths, npts)
    xrange = range(x0[1], stop = x0[1] + widths[1], length = npts[1])
    yrange = range(x0[2], stop = x0[2] + widths[2], length = npts[2])

    xcoords = repeat(xrange, inner = npts[2])
    ycoords = repeat(yrange, outer = npts[1])

    return vcat(xcoords', ycoords')
end

function bottom_ghost_coordinates(x0, xwidth, npts, dy, numlayers)

    return uniform_grid_coordinates(
        x0 - [0.0, numlayers * dy],
        [xwidth, (numlayers - 1) * dy],
        [npts, numlayers],
    )
end

function right_ghost_coordinates(x0, widths, npts, dx, numlayers)

    return uniform_grid_coordinates(
        x0 + [widths[1] + dx, 0.0],
        [(numlayers - 1) * dx, widths[2]],
        [numlayers, npts],
    )
end

function top_ghost_coordinates(x0, widths, npts, dy, numlayers)

    return uniform_grid_coordinates(
        x0 + [0.0, widths[2] + dy],
        [widths[1], (numlayers - 1) * dy],
        [npts, numlayers],
    )
end

function left_ghost_coordinates(x0, ywidth, npts, dx, numlayers)

    return uniform_grid_coordinates(
        x0 - [numlayers * dx, 0.0],
        [(numlayers - 1) * dx, ywidth],
        [numlayers, npts],
    )
end

function BoundaryPaddedMesh(mesh, numghostlayers)
    x0 = reference_corner(mesh)
    nfmside = nodes_per_mesh_side(mesh)
    meshwidths = widths(mesh)

    gridsize = meshwidths ./ (nfmside .- 1)

    bottomghostcoords =
        bottom_ghost_coordinates(x0, meshwidths[1], nfmside[1], gridsize[2], numghostlayers)
    rightghostcoords =
        right_ghost_coordinates(x0, meshwidths, nfmside[2], gridsize[1], numghostlayers)
    topghostcoords =
        top_ghost_coordinates(x0, meshwidths, nfmside[1], gridsize[2], numghostlayers)
    leftghostcoords =
        left_ghost_coordinates(x0, meshwidths[2], nfmside[2], gridsize[1], numghostlayers)

    return BoundaryPaddedMesh(
        mesh,
        gridsize,
        nfmside,
        numghostlayers,
        bottomghostcoords,
        rightghostcoords,
        topghostcoords,
        leftghostcoords,
    )
end

struct BoundaryPaddedLevelset
    paddedmesh::Any
    bottomghostdist::Any
    rightghostdist::Any
    topghostdist::Any
    leftghostdist::Any
end

function BoundaryPaddedLevelset(
    paddedmesh,
    refseedpoints,
    spatialseedpoints,
    seedcellids,
    levelset,
    levelsetcoeffs,
    cutmesh,
    tol,
)

    bottomghostdist = distance_to_zero_levelset(
        paddedmesh.bottomghostcoords,
        refseedpoints,
        spatialseedpoints,
        seedcellids,
        levelset,
        levelsetcoeffs,
        cutmesh,
        tol,
    )
    rightghostdist = distance_to_zero_levelset(
        paddedmesh.rightghostcoords,
        refseedpoints,
        spatialseedpoints,
        seedcellids,
        levelset,
        levelsetcoeffs,
        cutmesh,
        tol,
    )
    topghostdist = distance_to_zero_levelset(
        paddedmesh.topghostcoords,
        refseedpoints,
        spatialseedpoints,
        seedcellids,
        levelset,
        levelsetcoeffs,
        cutmesh,
        tol,
    )
    leftghostdist = distance_to_zero_levelset(
        paddedmesh.leftghostcoords,
        refseedpoints,
        spatialseedpoints,
        seedcellids,
        levelset,
        levelsetcoeffs,
        cutmesh,
        tol,
    )

    return BoundaryPaddedLevelset(
        paddedmesh,
        bottomghostdist,
        rightghostdist,
        topghostdist,
        leftghostdist,
    )
end
