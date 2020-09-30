struct CutMeshCellQuadratures
    quads::Any
    celltoquad::Any
    ncells::Any
    function CutMeshCellQuadratures(quads, celltoquad)
        nphase, ncells = size(celltoquad)
        @assert nphase == 2
        @assert all(celltoquad .>= 0)
        @assert all(celltoquad .<= length(quads))
        new(quads, celltoquad, ncells)
    end
end

function Base.getindex(vquads::CutMeshCellQuadratures, s, col)
    flag = (s == -1 || s == +1) && (1 <= col <= vquads.ncells)
    flag || throw(BoundsError(vquads.celltoquad, [s, col]))
    row = s == +1 ? 1 : 2
    return vquads.quads[vquads.celltoquad[row, col]]
end

function uniform_cell_quadrature(vquads::CutMeshCellQuadratures)
    return vquads.quads[1]
end

function cell_sign(levelset, levelsetcoeffs, nodalconnectivity)
    ncells = size(nodalconnectivity)[2]
    cellsign = zeros(Int, ncells)
    box = IntervalBox(-1..1, 2)
    for cellid = 1:ncells
        nodeids = nodalconnectivity[:, cellid]
        update!(levelset, levelsetcoeffs[nodeids])
        cellsign[cellid] = sign(levelset, box)
    end
    return cellsign
end

function CutMeshCellQuadratures(
    cellsign,
    levelset,
    levelsetcoeffs,
    nodalconnectivity,
    numqp,
)
    numcells = length(cellsign)
    @assert size(nodalconnectivity)[2] == numcells

    quad1d = ImplicitDomainQuadrature.ReferenceQuadratureRule(numqp)
    tpq = tensor_product_quadrature(2, numqp)
    quads = Vector{QuadratureRule{2}}(undef, 0)
    push!(quads, tpq)
    celltoquad = zeros(Int, 2, numcells)
    box = IntervalBox(-1..1, 2)

    for cellid = 1:numcells
        if cellsign[cellid] == +1
            celltoquad[1, cellid] = 1
        elseif cellsign[cellid] == -1
            celltoquad[2, cellid] = 1
        elseif cellsign[cellid] == 0
            nodeids = nodalconnectivity[:, cellid]
            update!(levelset, levelsetcoeffs[nodeids])

            pquad = quadrature(levelset, +1, false, box, quad1d)
            push!(quads, pquad)
            celltoquad[1, cellid] = length(quads)

            nquad = quadrature(levelset, -1, false, box, quad1d)
            push!(quads, nquad)
            celltoquad[2, cellid] = length(quads)
        end
    end
    return CutMeshCellQuadratures(quads, celltoquad)
end

function active_node_ids(s, cellsign, nodalconnectivity)
    @assert s ∈ [-1, 1]
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

struct CutMesh
    mesh::Mesh
    cellsign::Vector{Int}
    cutmeshnodeids::Matrix{Int}
    function CutMesh(mesh::Mesh,cellsign::Vector{Int},cutmeshnodeids::Matrix{Int})
        ncells = number_of_cells(mesh)
        numnodes = total_number_of_nodes(mesh)
        @assert length(cellsign) == ncells
        @assert size(cutmeshnodeids) == (2,numnodes)
        new(mesh,cellsign,cutmeshnodeids)
    end
end

function cell_sign(cutmesh::CutMesh)
    return cutmesh.cellsign
end

function cell_sign(cutmesh::CutMesh,cellid)
    return cutmesh.cellsign[cellid]
end

function CutMesh(levelset::InterpolatingPolynomial, levelsetcoeffs, mesh)
    nodalconnectivity = nodal_connectivity(mesh)
    cellsign = cell_sign(levelset, levelsetcoeffs, nodalconnectivity)

    posactivenodeids = active_node_ids(+1, cellsign, nodalconnectivity)
    negactivenodeids = active_node_ids(-1, cellsign, nodalconnectivity)

    totalnumnodes = total_number_of_nodes(mesh)
    cutmeshnodeids =
        cut_mesh_nodeids(posactivenodeids, negactivenodeids, totalnumnodes)
    return CutMesh(mesh, cellsign, cutmeshnodeids)
end

function nodal_connectivity(cutmesh::CutMesh, s, cellid)
    @assert s == -1 || s == +1
    ncells = cutmesh.mesh.ncells
    @assert 1 <= cellid <= ncells
    @assert cell_sign(cutmesh,cellid) == s || cell_sign(cutmesh,cellid) == 0
    row = s == +1 ? 1 : 2

    nc = nodal_connectivity(cutmesh.mesh)
    ids = nc[:, cellid]
    nodeids = cutmesh.cutmeshnodeids[row, ids]
    return nodeids
end
