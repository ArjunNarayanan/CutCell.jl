struct MergeMapper
    cellmaps
    function MergeMapper()
        south = cell_map_to_south()
        east = cell_map_to_east()
        north = cell_map_to_north()
        west = cell_map_to_west()
        southeast = cell_map_to_south_east()
        northeast = cell_map_to_north_east()
        northwest = cell_map_to_north_west()
        southwest = cell_map_to_south_west()
        new([south,east,north,west,southeast,northeast,northwest,southwest])
    end
end

function Base.getindex(m::MergeMapper,i)
    return m.cellmaps[i]
end

function cell_map_to_south()
    cellmap = CellMap([-1.0,-3.0],[1.0,-1.0])
    return cellmap
end

function cell_map_to_east()
    cellmap = CellMap([1.0,-1.0],[3.0,1.0])
    return cellmap
end

function cell_map_to_north()
    cellmap = CellMap([-1.0,1.0],[1.0,3.0])
    return cellmap
end

function cell_map_to_west()
    cellmap = CellMap([-3.0,-1.0],[-1.0,1.0])
    return cellmap
end

function cell_map_to_south_east()
    cellmap = CellMap([1.0,-3.0],[3.0,-1.0])
    return cellmap
end

function cell_map_to_north_east()
    cellmap = CellMap([1.0,1.0],[3.0,3.0])
    return cellmap
end

function cell_map_to_north_west()
    cellmap = CellMap([-3.0,1.0],[-1.0,3.0])
    return cellmap
end

function cell_map_to_south_west()
    cellmap = CellMap([-3.0,-3.0],[-1.0,-1.0])
    return cellmap
end

function map_quadrature(quad,mergemapper,mapid)
    mappedpoints = mergemapper[mapid](quad.points)
    return QuadratureRule(mappedpoints,quad.weights)
end

function update_quadrature!(cellquads::CellQuadratures,s,cellid,quad)
    row = cell_sign_to_row(s)
    idx = cellquads.celltoquad[row,cellid]
    cellquads.quads[idx] = quad
end

function update_quadrature!(interfacequads::InterfaceQuadratures,s,cellid,quad)
    row = cell_sign_to_row(s)
    idx = interfacequads.celltoquad[cellid]
    interfacequads.quads[row,idx] = quad
end

function map_and_update_quadrature!(cellquads,s,cellid,mergemapper,mapid)
    quad = cellquads[s,cellid]
    newquad = map_quadrature(quad,mergemapper,mapid)
    update_quadrature!(cellquads,s,cellid,newquad)
end

struct MergeCutMesh
    mergedwithcell
    cutmesh
    function MergeCutMesh(cutmesh::CutMesh)
        ncells = number_of_cells(cutmesh)
        mergedwithcell = vcat((1:ncells)',(1:ncells)')
        new(mergedwithcell,cutmesh)
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

function number_of_degrees_of_freedom(mergecutmesh::MergeCutMesh)
    dim = dimension(mergecutmesh)
    cellsign = cell_sign(mergecutmesh)
    numnodes = 0
    for (cellid,s) in enumerate(cellsign)
        if s == +1 || s == 0
            nodeids = nodal_connectivity(mergecutmesh,+1,cellid)
            numnodes = max(numnodes,maximum(nodeids))
        end
        if s == -1 || s == 0
            nodeids = nodal_connectivity(mergecutmesh,-1,cellid)
            numnodes = max(numnodes,maximum(nodeids))
        end
    end
    return dim*numnodes
end

function merge_cells!(mergecutmesh::MergeCutMesh,s,mergeto,mergefrom)
    row = cell_sign_to_row(s)
    ncells = number_of_cells(mergecutmesh)
    @assert 1 <= mergeto <= ncells
    @assert 1 <= mergefrom <= ncells

    mergecutmesh.mergedwithcell[row,mergefrom] = mergeto
end

function Base.show(io::IO,mergecutmesh::MergeCutMesh)
    ncells = number_of_cells(mergecutmesh.cutmesh)
    str = "MergeCutMesh\n\tNum. Cells: $ncells"
    print(io,str)
end

function nodal_connectivity(mergecutmesh::MergeCutMesh,s,cellid)
    row = cell_sign_to_row(s)
    mergecellid = mergecutmesh.mergedwithcell[row,cellid]
    return nodal_connectivity(mergecutmesh.cutmesh,s,mergecellid)
end
