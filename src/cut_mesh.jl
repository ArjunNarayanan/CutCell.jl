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

function Base.getindex(vquads::CutMeshCellQuadratures, s, cellid)
    flag = (s == -1 || s == +1) && (1 <= cellid <= vquads.ncells)
    flag || throw(BoundsError(vquads.celltoquad, [s, cellid]))
    row = s == +1 ? 1 : 2
    return vquads.quads[vquads.celltoquad[row, cellid]]
end

function uniform_cell_quadrature(vquads::CutMeshCellQuadratures)
    return vquads.quads[1]
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

function CutMeshCellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)

    cellsign = cell_sign(cutmesh)
    nodalconnectivity = nodal_connectivity(cutmesh.mesh)
    return CutMeshCellQuadratures(
        cellsign,
        levelset,
        levelsetcoeffs,
        nodalconnectivity,
        numqp,
    )
end

struct CutMeshInterfaceQuadratures
    quads::Any
    normals
    celltoquad::Any
    ncells::Any
    function CutMeshInterfaceQuadratures(quads, normals, celltoquad)
        ncells = length(celltoquad)
        @assert all(celltoquad .>= 0)
        @assert all(celltoquad .<= length(quads))
        @assert length(quads) == length(normals)
        new(quads, normals, celltoquad, ncells)
    end
end

function Base.getindex(iquads::CutMeshInterfaceQuadratures, cellid)
    1 <= cellid <= iquads.ncells ||
        throw(BoundsError(iquads.celltoquad, [cellid]))
    idx = iquads.celltoquad[cellid]
    idx > 0 || throw(BoundsError(iquads.quads,[idx]))
    return iquads.quads[idx]
end

function normals(iquads::CutMeshInterfaceQuadratures, cellid)
    1 <= cellid <= iquads.ncells ||
        throw(BoundsError(iquads.celltoquad, [cellid]))
    idx = iquads.celltoquad[cellid]
    idx > 0 || throw(BoundsError(iquads.normals,[idx]))
    return iquads.normals[idx]
end

function levelset_normal(levelset,p::V,invjac) where {V<:AbstractVector}
    g = vec(gradient(levelset,p))
    n = invjac .* g
    return n/norm(n)
end

function levelset_normal(levelset,points::M,invjac) where {M<:AbstractMatrix}
    npts = size(points)[2]
    g = hcat([gradient(levelset,points[:,i]) for i = 1:npts]...)
    normals = diagm(invjac)*g
    for i in 1:npts
        n = normals[:,i]
        normals[:,i] = n/norm(n)
    end
    return normals
end

function CutMeshInterfaceQuadratures(
    cellsign,
    levelset,
    levelsetcoeffs,
    nodalconnectivity,
    numqp,
    cellmap,
)

    numcells = length(cellsign)
    @assert size(nodalconnectivity)[2] == numcells

    box = IntervalBox(-1..1,2)
    invjac = inverse_jacobian(cellmap)
    quad1d = ImplicitDomainQuadrature.ReferenceQuadratureRule(numqp)

    quads = []
    normals = []
    celltoquad = zeros(Int,numcells)

    for cellid in 1:numcells
        if cellsign[cellid] == 0
            nodeids = nodalconnectivity[:,cellid]
            update!(levelset,levelsetcoeffs[nodeids])

            squad = quadrature(levelset,1,true,box,quad1d)
            push!(quads,squad)
            n = levelset_normal(levelset,squad.points,invjac)
            push!(normals,n)

            celltoquad[cellid] = length(quads)
        end
    end
    return CutMeshInterfaceQuadratures(quads,normals,celltoquad)
end

function face_quadrature_rules(levelset, signcondition, quad1d)
    bq = QuadratureRule(quadrature(
        [x -> levelset(extend(x, 2, -1.0))],
        [signcondition],
        -1.0,
        +1.0,
        quad1d,
    ))
    bq = extend_to_face(bq, 1)

    rq = QuadratureRule(quadrature(
        [x -> levelset(extend(x, 1, +1.0))],
        [signcondition],
        -1.0,
        +1.0,
        quad1d,
    ))
    rq = extend_to_face(rq, 2)

    tq = QuadratureRule(quadrature(
        [x -> levelset(extend(x, 2, +1.0))],
        [signcondition],
        -1.0,
        +1.0,
        quad1d,
    ))
    tq = extend_to_face(tq, 3)

    lq = QuadratureRule(quadrature(
        [x -> levelset(extend(x, 1, -1.0))],
        [signcondition],
        -1.0,
        +1.0,
        quad1d,
    ))
    lq = extend_to_face(lq, 4)

    return bq, rq, tq, lq
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
    ncells::Int
    numnodes::Int
    function CutMesh(
        mesh::Mesh,
        cellsign::Vector{Int},
        cutmeshnodeids::Matrix{Int},
    )
        ncells = number_of_cells(mesh)
        numnodes = total_number_of_nodes(mesh)
        @assert length(cellsign) == ncells
        @assert size(cutmeshnodeids) == (2, numnodes)
        totalnumnodes = maximum(cutmeshnodeids)
        new(mesh, cellsign, cutmeshnodeids, ncells, totalnumnodes)
    end
end

function total_number_of_nodes(cutmesh::CutMesh)
    return cutmesh.numnodes
end

function number_of_cells(cutmesh::CutMesh)
    return cutmesh.ncells
end

function cell_sign(cutmesh::CutMesh)
    return cutmesh.cellsign
end

function cell_sign(cutmesh::CutMesh, cellid)
    return cutmesh.cellsign[cellid]
end

function cell_map(cutmesh::CutMesh, cellid)
    return cell_map(cutmesh.mesh, cellid)
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
    @assert cell_sign(cutmesh, cellid) == s || cell_sign(cutmesh, cellid) == 0
    row = s == +1 ? 1 : 2

    nc = nodal_connectivity(cutmesh.mesh)
    ids = nc[:, cellid]
    nodeids = cutmesh.cutmeshnodeids[row, ids]
    return nodeids
end

struct CutMeshBilinearForms
    cellmatrices::Any
    celltomatrix::Any
    ncells::Any
    function CutMeshBilinearForms(cellmatrices, celltomatrix)
        nphase, ncells = size(celltomatrix)
        @assert nphase == 2
        @assert all(celltomatrix .>= 0)
        @assert all(celltomatrix .<= length(cellmatrices))
        new(cellmatrices, celltomatrix, ncells)
    end
end

function Base.getindex(cbf::CutMeshBilinearForms, s, cellid)
    flag = (s == -1 || s == +1) && (1 <= cellid <= cbf.ncells)
    flag || throw(BoundsError(cbf.celltomatrix, [s, cellid]))
    row = s == +1 ? 1 : 2
    return cbf.cellmatrices[cbf.celltomatrix[row, cellid]]
end

function Base.getindex(cbf::CutMeshBilinearForms, s)
    flag = (s == -1 || s == +1)
    flag || error("Expected s ∈ {-1,1}, got s = $s")
    idx = s == +1 ? 1 : 2
    return cbf.cellmatrices[idx]
end

function CutMeshBilinearForms(
    basis,
    cutmeshquads,
    stiffnesses,
    cellsign,
    cellmap,
)
    @assert length(stiffnesses) == 2
    ncells = length(cellsign)

    uniformquad = uniform_cell_quadrature(cutmeshquads)
    uniformbf1 = bilinear_form(basis, uniformquad, stiffnesses[1], cellmap)
    uniformbf2 = bilinear_form(basis, uniformquad, stiffnesses[2], cellmap)

    cellmatrices = [uniformbf1, uniformbf2]
    celltomatrix = zeros(Int, 2, ncells)

    for cellid = 1:ncells
        if cellsign[cellid] == +1
            celltomatrix[1, cellid] = 1
        elseif cellsign[cellid] == -1
            celltomatrix[2, cellid] = 2
        else
            pquad = cutmeshquads[+1, cellid]
            pbf = bilinear_form(basis, pquad, stiffnesses[1], cellmap)
            push!(cellmatrices, pbf)
            celltomatrix[1, cellid] = length(cellmatrices)

            nquad = cutmeshquads[-1, cellid]
            nbf = bilinear_form(basis, nquad, stiffnesses[2], cellmap)
            push!(cellmatrices, nbf)
            celltomatrix[2, cellid] = length(cellmatrices)
        end
    end
    return CutMeshBilinearForms(cellmatrices, celltomatrix)
end

function CutMeshBilinearForms(basis, cutmeshquads, stiffnesses, cutmesh)
    cellsign = cell_sign(cutmesh)
    cellmap = cell_map(cutmesh, 1)
    return CutMeshBilinearForms(
        basis,
        cutmeshquads,
        stiffnesses,
        cellsign,
        cellmap,
    )
end

function assemble_bilinear_form!(
    sysmatrix::SystemMatrix,
    cutmeshbfs::CutMeshBilinearForms,
    cutmesh::CutMesh,
    dofspernode,
)

    ncells = number_of_cells(cutmesh)
    cellsign = cell_sign(cutmesh)

    uniformvals1 = vec(cutmeshbfs[+1])
    uniformvals2 = vec(cutmeshbfs[-1])

    for cellid = 1:ncells
        s = cellsign[cellid]
        if s == +1
            nodeids = nodal_connectivity(cutmesh, +1, cellid)
            assemble_cell_bilinear_form!(
                sysmatrix,
                nodeids,
                dofspernode,
                uniformvals1,
            )
        elseif s == -1
            nodeids = nodal_connectivity(cutmesh, -1, cellid)
            assemble_cell_bilinear_form!(
                sysmatrix,
                nodeids,
                dofspernode,
                uniformvals2,
            )
        elseif s == 0
            nodeids1 = nodal_connectivity(cutmesh, +1, cellid)
            vals1 = vec(cutmeshbfs[+1, cellid])
            assemble_cell_bilinear_form!(
                sysmatrix,
                nodeids1,
                dofspernode,
                vals1,
            )

            nodeids2 = nodal_connectivity(cutmesh, -1, cellid)
            vals2 = vec(cutmeshbfs[-1, cellid])
            assemble_cell_bilinear_form!(
                sysmatrix,
                nodeids2,
                dofspernode,
                vals2,
            )
        else
            error("Expected s ∈ {+1,0,-1}, received s = $s")
        end
    end
end

struct CutMeshInterfaceConstraints
    cellmatrices
    celltomatrix
    ncells
    function CutMeshInterfaceConstraints(cellmatrices,celltomatrix)
        ncells = length(celltomatrix)
        @assert all(celltomatrix .>= 0)
        @assert all(celltomatrix .<= length(cellmatrices))
        new(cellmatrices,celltomatrix,ncells)
    end
end

function CutMeshInterfaceConstraints(basis,cutmeshinterfacequads,cellsign,)

end
