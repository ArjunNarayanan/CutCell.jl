struct MergeMapper
    cellmaps::Any
    function MergeMapper()
        south = cell_map_to_south()
        east = cell_map_to_east()
        north = cell_map_to_north()
        west = cell_map_to_west()
        southeast = cell_map_to_south_east()
        northeast = cell_map_to_north_east()
        northwest = cell_map_to_north_west()
        southwest = cell_map_to_south_west()
        new([south, east, north, west, southeast, northeast, northwest, southwest])
    end
end

function Base.getindex(m::MergeMapper, i)
    return m.cellmaps[i]
end

function cell_map_to_south()
    cellmap = CellMap([-1.0, -3.0], [1.0, -1.0])
    return cellmap
end

function cell_map_to_east()
    cellmap = CellMap([1.0, -1.0], [3.0, 1.0])
    return cellmap
end

function cell_map_to_north()
    cellmap = CellMap([-1.0, 1.0], [1.0, 3.0])
    return cellmap
end

function cell_map_to_west()
    cellmap = CellMap([-3.0, -1.0], [-1.0, 1.0])
    return cellmap
end

function cell_map_to_south_east()
    cellmap = CellMap([1.0, -3.0], [3.0, -1.0])
    return cellmap
end

function cell_map_to_north_east()
    cellmap = CellMap([1.0, 1.0], [3.0, 3.0])
    return cellmap
end

function cell_map_to_north_west()
    cellmap = CellMap([-3.0, 1.0], [-1.0, 3.0])
    return cellmap
end

function cell_map_to_south_west()
    cellmap = CellMap([-3.0, -3.0], [-1.0, -1.0])
    return cellmap
end

function map_quadrature(quad, mergemapper, mapid)
    mappedpoints = mergemapper[mapid](quad.points)
    return QuadratureRule(mappedpoints, quad.weights)
end

function update_quadrature!(cellquads::CellQuadratures, s, cellid, quad)
    row = cell_sign_to_row(s)
    idx = cellquads.celltoquad[row, cellid]
    cellquads.quads[idx] = quad
end

function update_quadrature!(interfacequads::InterfaceQuadratures, s, cellid, quad)
    row = cell_sign_to_row(s)
    idx = interfacequads.celltoquad[cellid]
    interfacequads.quads[row, idx] = quad
end

function map_and_update_quadrature!(cellquads, s, cellid, mergemapper, mapid)
    quad = cellquads[s, cellid]
    newquad = map_quadrature(quad, mergemapper, mapid)
    update_quadrature!(cellquads, s, cellid, newquad)
end

struct MergeCutMesh
    mergedwithcell::Any
    cutmesh::Any
    function MergeCutMesh(cutmesh::CutMesh)
        ncells = number_of_cells(cutmesh)
        mergedwithcell = vcat((1:ncells)', (1:ncells)')
        new(mergedwithcell, cutmesh)
    end
end

function number_of_cells(mergecutmesh::MergeCutMesh)
    return number_of_cells(mergecutmesh.cutmesh)
end

function cell_sign(mergecutmesh::MergeCutMesh)
    return cell_sign(mergecutmesh.cutmesh)
end

function dimension(mergecutmesh::MergeCutMesh)
    return dimension(mergecutmesh.cutmesh)
end

function cell_map(mergecutmesh::MergeCutMesh, idx)
    return cell_map(mergecutmesh.cutmesh,idx)
end

function number_of_nodes(mergecutmesh::MergeCutMesh)
    cellsign = cell_sign(mergecutmesh)
    numnodes = 0
    for (cellid, s) in enumerate(cellsign)
        if s == +1 || s == 0
            nodeids = nodal_connectivity(mergecutmesh, +1, cellid)
            numnodes = max(numnodes, maximum(nodeids))
        end
        if s == -1 || s == 0
            nodeids = nodal_connectivity(mergecutmesh, -1, cellid)
            numnodes = max(numnodes, maximum(nodeids))
        end
    end
    return numnodes
end

function number_of_degrees_of_freedom(mergecutmesh::MergeCutMesh)
    dim = dimension(mergecutmesh)
    numnodes = number_of_nodes(mergecutmesh)
    return dim * numnodes
end

function merge_cells!(mergecutmesh::MergeCutMesh, s, mergeto, mergefrom)
    row = cell_sign_to_row(s)
    ncells = number_of_cells(mergecutmesh)
    @assert 1 <= mergeto <= ncells
    @assert 1 <= mergefrom <= ncells

    mergecutmesh.mergedwithcell[row, mergefrom] = mergeto
end

function Base.show(io::IO, mergecutmesh::MergeCutMesh)
    ncells = number_of_cells(mergecutmesh.cutmesh)
    str = "MergeCutMesh\n\tNum. Cells: $ncells"
    print(io, str)
end

function nodal_connectivity(mergecutmesh::MergeCutMesh, s, cellid)
    row = cell_sign_to_row(s)
    mergecellid = mergecutmesh.mergedwithcell[row, cellid]
    return nodal_connectivity(mergecutmesh.cutmesh, s, mergecellid)
end

function cell_connectivity(mergecutmesh::MergeCutMesh)
    return cell_connectivity(mergecutmesh.cutmesh)
end

function active_cells(mergecutmesh::MergeCutMesh)
    return active_cells(mergecutmesh.cutmesh)
end

function quadrature_areas(cellquads)
    ncells = number_of_cells(cellquads)
    areas = zeros(2, ncells)
    for cellid = 1:ncells
        for s in [+1, -1]
            if has_quadrature(cellquads, s, cellid)
                row = cell_sign_to_row(s)
                areas[row, cellid] = sum(cellquads[s, cellid].weights)
            end
        end
    end
    return areas
end

function is_tiny_cell(cellquads, quadareas; ratio = 0.2)
    ncells = number_of_cells(cellquads)
    istiny = zeros(Bool, 2, ncells)

    tinyarea = ratio * sum(uniform_cell_quadrature(cellquads).weights)

    for cellid = 1:ncells
        for s in [+1, -1]
            if has_quadrature(cellquads, s, cellid)
                row = cell_sign_to_row(s)
                if quadareas[row, cellid] <= tinyarea
                    istiny[row, cellid] = true
                end
            end
        end
    end

    return istiny
end

function merge_cells_in_mesh!(mergecutmesh, cellquads, interfacequads, mergemapper)

    cellconnectivity = cell_connectivity(mergecutmesh)
    activecells = active_cells(mergecutmesh)
    ncells = number_of_cells(mergecutmesh)

    quadareas = quadrature_areas(cellquads)
    istinycell = is_tiny_cell(cellquads, quadareas)

    for cellid = 1:ncells
        for s in [+1, -1]
            row = cell_sign_to_row(s)
            if istinycell[row, cellid]
                nbrcellids = cellconnectivity[:, cellid]
                nbractive =
                    [nbrid == 0 ? false : activecells[row, nbrid] for nbrid in nbrcellids]
                nbrnontiny =
                    [nbrid == 0 ? false : !istinycell[row, nbrid] for nbrid in nbrcellids]
                faceid = findfirst(nbractive .& nbrnontiny)
                !isnothing(faceid) ||
                    error("Could not find appropriate merge direction for cell $cellid, cellsign $s")
                oppositeface = opposite_face(faceid)
                mergecellid = nbrcellids[faceid]

                merge_cells!(mergecutmesh, s, mergecellid, cellid)
                map_and_update_quadrature!(cellquads, s, cellid, mergemapper, oppositeface)
                map_and_update_quadrature!(
                    interfacequads,
                    s,
                    cellid,
                    mergemapper,
                    oppositeface,
                )
            end
        end
    end
end

struct MergedMesh
    mergecutmesh::MergeCutMesh
    nodelabeltonodeid
    numnodes
    function MergedMesh(mergecutmesh::MergeCutMesh)
        activenodelabels = active_node_labels(mergecutmesh)
        maxlabel = maximum(activenodelabels)
        nodelabeltonodeid = zeros(Int,maxlabel)
        for (idx,label) in enumerate(activenodelabels)
            nodelabeltonodeid[label] = idx
        end
        numnodes = length(activenodelabels)
        new(mergecutmesh,nodelabeltonodeid,numnodes)
    end
end

function number_of_cells(mergedmesh::MergedMesh)
    return number_of_cells(mergedmesh.mergecutmesh)
end

function Base.show(io::IO, mergedmesh::MergedMesh)
    ncells = number_of_cells(mergedmesh)
    numnodes = mergedmesh.numnodes
    str = "MergedMesh\n\tNum. Cells: $ncells\n\tNum. Nodes: $numnodes"
    print(io,str)
end

function dimension(mergedmesh::MergedMesh)
    return dimension(mergedmesh.mergecutmesh)
end

function cell_sign(mergedmesh::MergedMesh)
    return cell_sign(mergedmesh.mergecutmesh)
end

function nodal_connectivity(mergedmesh::MergedMesh,s,cellid)
    labels = nodal_connectivity(mergedmesh.mergecutmesh,s,cellid)
    nodeids = mergedmesh.nodelabeltonodeid[labels]
    return nodeids
end

function cell_map(mergedmesh::MergedMesh, idx)
    return cell_map(mergedmesh.mergecutmesh,idx)
end

function number_of_nodes(mergedmesh::MergedMesh)
    return mergedmesh.numnodes
end

function number_of_degrees_of_freedom(mergedmesh::MergedMesh)
    dim = dimension(mergedmesh)
    numnodes = number_of_nodes(mergedmesh)
    return dim*numnodes
end

function active_node_labels(mergecutmesh::MergeCutMesh)
    numcells = number_of_cells(mergecutmesh)
    cellsign = cell_sign(mergecutmesh)

    activenodeids = Int[]

    cellids = findall(x -> x == 0 || x == +1, cellsign)
    for cellid in cellids
        append!(activenodeids,nodal_connectivity(mergecutmesh,+1,cellid))
    end

    cellids = findall(x -> x == 0 || x == -1, cellsign)
    for cellid in cellids
        append!(activenodeids,nodal_connectivity(mergecutmesh,-1,cellid))
    end

    sort!(activenodeids)
    unique!(activenodeids)
    return activenodeids
end
