function reference_coordinates(coords, cellids, cutmesh)
    numpts = size(coords)[2]
    @assert length(cellids) == numpts
    refcoords = similar(coords)
    for i = 1:numpts
        cellmap = cell_map(cutmesh, cellids[i])
        refcoords[:, i] .= inverse(cellmap, coords[:, i])
    end
    return refcoords
end

function levelset_sign(refcoords, cellids, levelset, levelsetcoeffs, mesh)
    breakpoints = [i == 1 ? true : cellids[i] != cellids[i-1] for i = 1:length(cellids)]
    start = 1
    levelsetsign = zeros(Int, length(cellids))
    while start <= length(cellids)
        nodeids = nodal_connectivity(mesh, cellids[start])
        update!(levelset, levelsetcoeffs[nodeids])
        stop = findnext(breakpoints, start + 1)
        stop = isnothing(stop) ? length(cellids) + 1 : stop
        for idx = start:(stop-1)
            levelsetsign[idx] = sign(levelset(refcoords[:, idx]))
        end
        start = stop
    end
    return levelsetsign
end

function interpolate(coords, nodalvalues, basis, levelset, levelsetcoeffs, cutmesh)
    dim, numpts = size(coords)
    ndofs, numnodes = size(nodalvalues)

    cellids = [cell_id(cutmesh, coords[:, i]) for i = 1:numpts]
    refcoords = reference_coordinates(coords, cellids, cutmesh)
    bgmesh = background_mesh(cutmesh)
    levelsetsign = levelset_sign(refcoords, cellids, levelset, levelsetcoeffs, bgmesh)

    interpolater = InterpolatingPolynomial(ndofs, basis)
    interpvals = zeros(ndofs, numpts)

    breakpoints = [
        i == 1 ? true :
        (cellids[i] != cellids[i-1]) || (levelsetsign[i] != levelsetsign[i-1])
        for i = 1:numpts
    ]
    start = 1
    while start <= numpts
        s = levelsetsign[start]
        nodeids = nodal_connectivity(cutmesh,s,cellids[start])
        update!(interpolater,nodalvalues[:,nodeids])
        stop = findnext(breakpoints,start+1)
        stop = isnothing(stop) ? numpts+1 : stop
        for idx = start:(stop-1)
            interpvals[:,idx] .= interpolater(refcoords[:,idx])
        end
        start = stop
    end
    return interpvals
end
