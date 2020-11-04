struct CutMesh
    mesh::Mesh
    cellsign::Vector{Int}
    activecells::Matrix{Bool}
    cutmeshnodeids::Matrix{Int}
    ncells::Int
    numnodes::Int
    nelmts::Int
    function CutMesh(mesh::Mesh, cellsign::Vector{Int}, cutmeshnodeids::Matrix{Int})
        ncells = number_of_cells(mesh)
        nummeshnodes = number_of_nodes(mesh)
        @assert length(cellsign) == ncells
        @assert size(cutmeshnodeids) == (2, nummeshnodes)

        activecells = active_cells(cellsign)
        numnodes = maximum(cutmeshnodeids)
        nelmts = count(activecells)
        new(mesh, cellsign, activecells, cutmeshnodeids, ncells, numnodes, nelmts)
    end
end

function active_cells(cellsign)
    ncells = length(cellsign)
    activecells = zeros(Bool, 2, ncells)
    idx1 = findall((cellsign .== +1) .| (cellsign .== 0))
    idx2 = findall((cellsign .== -1) .| (cellsign .== 0))
    activecells[1, idx1] .= true
    activecells[2, idx2] .= true
    return activecells
end

function CutMesh(levelset::InterpolatingPolynomial, levelsetcoeffs, mesh)
    nodalconnectivity = nodal_connectivity(mesh)
    cellsign = cell_sign(levelset, levelsetcoeffs, nodalconnectivity)

    posactivenodeids = active_node_ids(+1, cellsign, nodalconnectivity)
    negactivenodeids = active_node_ids(-1, cellsign, nodalconnectivity)

    totalnumnodes = number_of_nodes(mesh)
    cutmeshnodeids = cut_mesh_nodeids(posactivenodeids, negactivenodeids, totalnumnodes)
    return CutMesh(mesh, cellsign, cutmeshnodeids)
end

function Base.show(io::IO, cutmesh::CutMesh)
    ncells = number_of_cells(cutmesh)
    nnodes = number_of_nodes(cutmesh)
    nelmts = number_of_elements(cutmesh)
    str = "CutMesh\n\tNum. Cells: $ncells\n\tNum. Elements: $nelmts\n\tNum. Nodes: $nnodes"
    print(io, str)
end

function nodal_connectivity(cutmesh::CutMesh, s, cellid)
    row = cell_sign_to_row(s)
    ncells = cutmesh.mesh.ncells
    @assert 1 <= cellid <= ncells
    cs = cell_sign(cutmesh, cellid)
    @assert cs == s || cs == 0

    nc = nodal_connectivity(cutmesh.mesh)
    ids = nc[:, cellid]
    nodeids = cutmesh.cutmeshnodeids[row, ids]
    return nodeids
end

function active_cells(cutmesh::CutMesh)
    return cutmesh.activecells
end

function dimension(cutmesh::CutMesh)
    return dimension(cutmesh.mesh)
end

function number_of_nodes(cutmesh::CutMesh)
    return cutmesh.numnodes
end

function number_of_degrees_of_freedom(cutmesh::CutMesh)
    dim = dimension(cutmesh)
    numnodes = number_of_nodes(cutmesh)
    return dim * numnodes
end

function number_of_cells(cutmesh::CutMesh)
    return cutmesh.ncells
end

function background_mesh(cutmesh::CutMesh)
    return cutmesh.mesh
end

function number_of_elements(cutmesh::CutMesh)
    return cutmesh.nelmts
end

function cell_sign(cutmesh::CutMesh)
    return cutmesh.cellsign
end

function cell_id(cutmesh::CutMesh,x)
    return cell_id(cutmesh.mesh,x)
end

function cell_sign(cutmesh::CutMesh, cellid)
    return cutmesh.cellsign[cellid]
end

function cell_map(cutmesh::CutMesh, cellid)
    return cell_map(cutmesh.mesh, cellid)
end

function cell_map(cutmesh::CutMesh)
    return cell_map(cutmesh,1)
end

function cell_map(cutmesh::CutMesh, s, cellid)
    return cell_map(cutmesh, cellid)
end

function jacobian(cutmesh::CutMesh)
    return jacobian(cell_map(cutmesh))
end

function face_determinant_jacobian(cutmesh::CutMesh)
    return face_determinant_jacobian(cutmesh.mesh)
end

function is_interior_cell(cutmesh::CutMesh)
    return is_interior_cell(cutmesh.mesh)
end

function cell_connectivity(cutmesh::CutMesh)
    return cell_connectivity(cutmesh.mesh)
end

function cell_sign(levelset, levelsetcoeffs, nodalconnectivity)
    ncells = size(nodalconnectivity)[2]
    cellsign = zeros(Int, ncells)
    box = IntervalBox(-1..1, 2)
    for cellid = 1:ncells
        nodeids = nodalconnectivity[:, cellid]
        update!(levelset, levelsetcoeffs[nodeids])
        s = sign_allow_perturbations(levelset, box)
        if (s == +1 || s == -1)
            cellsign[cellid] = s
        else
            cellsign[cellid] = 0
        end
    end
    return cellsign
end

function active_node_ids(s, cellsign, nodalconnectivity)
    @assert (s == -1 || s == +1)
    activenodeids = Int[]
    numcells = size(nodalconnectivity)[2]
    for cellid = 1:numcells
        if cellsign[cellid] == s || cellsign[cellid] == 0
            append!(activenodeids, nodalconnectivity[:, cellid])
        end
    end
    sort!(activenodeids)
    unique!(activenodeids)
    return activenodeids
end

function cut_mesh_nodeids(posactivenodeids, negactivenodeids, totalnumnodes)
    cutmeshnodeids = zeros(Int, 2, totalnumnodes)
    counter = 1
    for nodeid in posactivenodeids
        cutmeshnodeids[1, nodeid] = counter
        counter += 1
    end
    for nodeid in negactivenodeids
        cutmeshnodeids[2, nodeid] = counter
        counter += 1
    end
    return cutmeshnodeids
end
