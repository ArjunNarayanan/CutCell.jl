struct SystemMatrix
    rows::Any
    cols::Any
    vals::Any
    function SystemMatrix(rows, cols, vals)
        @assert length(rows) == length(cols) == length(vals)
        new(rows, cols, vals)
    end
end

function SystemMatrix()
    rows = Int[]
    cols = Int[]
    vals = zeros(0)
    SystemMatrix(rows, cols, vals)
end

function Base.show(io::IO, sysmatrix::SystemMatrix)
    numvals = length(sysmatrix.rows)
    str = "SystemMatrix with $numvals entries"
    print(io, str)
end

function assemble!(matrix::SystemMatrix, rows, cols, vals)
    @assert length(rows) == length(cols) == length(vals)
    append!(matrix.rows, rows)
    append!(matrix.cols, cols)
    append!(matrix.vals, vals)
end

struct SystemRHS
    rows::Any
    vals::Any
    function SystemRHS(rows, vals)
        @assert length(rows) == length(vals)
        new(rows, vals)
    end
end

function SystemRHS()
    SystemRHS(Int[], zeros(0))
end

function Base.show(io::IO, sysrhs::SystemRHS)
    numvals = length(sysrhs.rows)
    str = "SystemRHS with $numvals entries"
    print(io, str)
end

function assemble!(systemrhs::SystemRHS, rows, vals)
    @assert length(rows) == length(vals)
    append!(systemrhs.rows, rows)
    append!(systemrhs.vals, vals)
end

function node_to_dof_id(nodeid, dof, dofspernode)
    return (nodeid - 1) * dofspernode + dof
end

function element_dofs(nodeids, dofspernode)
    numnodes = length(nodeids)
    extnodeids = repeat(nodeids, inner = dofspernode)
    dofs = repeat(1:dofspernode, numnodes)
    edofs = [node_to_dof_id(n, d, dofspernode) for (n, d) in zip(extnodeids, dofs)]
    return edofs
end

function element_dofs_to_operator_dofs(rowdofs, coldofs)
    nr = length(rowdofs)
    nc = length(coldofs)
    rows = repeat(rowdofs, outer = nc)
    cols = repeat(coldofs, inner = nr)
    return rows, cols
end

function assemble_couple_cell_matrix!(sysmatrix, nodeids1, nodeids2, dofspernode, vals)
    edofs1 = element_dofs(nodeids1, dofspernode)
    edofs2 = element_dofs(nodeids2, dofspernode)

    rows, cols = element_dofs_to_operator_dofs(edofs1, edofs2)
    assemble!(sysmatrix, rows, cols, vals)
end

function assemble_cell_matrix!(sysmatrix::SystemMatrix, nodeids, dofspernode, vals)

    edofs = element_dofs(nodeids, dofspernode)
    rows, cols = element_dofs_to_operator_dofs(edofs, edofs)
    assemble!(sysmatrix, rows, cols, vals)
end

function assemble_bilinear_form!(
    sysmatrix::SystemMatrix,
    cellmatrix,
    nodalconnectivity,
    dofspernode,
)

    ncells = size(nodalconnectivity)[2]
    vals = vec(cellmatrix)
    for cellid = 1:ncells
        nodeids = nodalconnectivity[:, cellid]
        assemble_cell_matrix!(sysmatrix, nodeids, dofspernode, vals)
    end
end

function assemble_bilinear_form!(sysmatrix::SystemMatrix, cellmatrix, mesh::Mesh)

    dofspernode = dimension(mesh)
    nodalconnectivity = nodal_connectivity(mesh)
    assemble_bilinear_form!(sysmatrix, cellmatrix, nodalconnectivity, dofspernode)
end

function assemble_cell_rhs!(sysrhs, nodeids, dofspernode, vals)
    rows = element_dofs(nodeids, dofspernode)
    assemble!(sysrhs, rows, vals)
end

function assemble_body_force_linear_form!(
    systemrhs,
    rhsfunc,
    basis,
    quad,
    cellmaps,
    nodalconnectivity,
)

    ncells = length(cellmaps)
    nf = number_of_basis_functions(basis)
    dim = dimension(basis)
    @assert size(nodalconnectivity) == (nf, ncells)
    for (idx, cellmap) in enumerate(cellmaps)
        rhs = linear_form(rhsfunc, basis, quad, cellmap)
        edofs = element_dofs(nodalconnectivity[:, idx], dim)
        assemble!(systemrhs, edofs, rhs)
    end
end

function assemble_body_force_linear_form!(systemrhs, rhsfunc, basis, quad, mesh::Mesh)
    cellmaps = cell_maps(mesh)
    nodalconnectivity = nodal_connectivity(mesh)
    assemble_body_force_linear_form!(
        systemrhs,
        rhsfunc,
        basis,
        quad,
        cellmaps,
        nodalconnectivity,
    )
end

function assemble_traction_force_linear_form!(
    systemrhs,
    tractionfunc,
    basis,
    facequads,
    cellmaps,
    nodalconnectivity,
    cellconnectivity,
    istractionboundary,
)

    dim = dimension(basis)
    refmidpoints = reference_face_midpoints()
    isboundarycell = is_boundary_cell(cellconnectivity)
    cellids = findall(isboundarycell)
    facedetjac = face_determinant_jacobian(cellmaps[1])
    for cellid in cellids
        cellmap = cellmaps[cellid]
        for (faceid, nbrcellid) in enumerate(cellconnectivity[:, cellid])
            if nbrcellid == 0
                if istractionboundary(cellmap(refmidpoints[faceid]))
                    rhs = linear_form(
                        tractionfunc,
                        basis,
                        facequads[faceid],
                        cellmap,
                        facedetjac[faceid],
                    )
                    edofs = element_dofs(nodalconnectivity[:, cellid], dim)
                    assemble!(systemrhs, edofs, rhs)
                end
            end
        end
    end
end

function assemble_stress_linear_form!(
    systemrhs,
    basis,
    quad,
    stiffness,
    nodaldisplacement,
    nodalconnectivity,
    jacobian,
)

    dim = dimension(basis)
    sdim = number_of_symmetric_degrees_of_freedom(dim)
    nf,ncells = size(nodalconnectivity)

    for cellid in 1:ncells
        nodeids = nodalconnectivity[:,cellid]
        elementdofs = element_dofs(nodeids,dim)
        celldisp = nodaldisplacement[elementdofs]
        rhs = stress_cell_rhs(basis,quad,stiffness,celldisp,jacobian)

        stressdofs = element_dofs(nodeids,sdim)
        assemble!(systemrhs,stressdofs,rhs)
    end
end

function make_sparse(sysmatrix::SystemMatrix, ndofs::Int)
    return dropzeros!(sparse(sysmatrix.rows, sysmatrix.cols, sysmatrix.vals, ndofs, ndofs))
end

function make_sparse(sysmatrix::SystemMatrix, mesh)
    totaldofs = number_of_degrees_of_freedom(mesh)
    return make_sparse(sysmatrix, totaldofs)
end

function make_sparse_stress_operator(sysmatrix,mesh)
    dim = dimension(mesh)
    sdim = number_of_symmetric_degrees_of_freedom(dim)
    numnodes = number_of_nodes(mesh)
    totaldofs = sdim*numnodes
    return make_sparse(sysmatrix,totaldofs)
end

function rhs(sysrhs::SystemRHS, ndofs::Int)
    return Array(sparsevec(sysrhs.rows, sysrhs.vals, ndofs))
end

function rhs(sysrhs::SystemRHS, mesh)
    totaldofs = number_of_degrees_of_freedom(mesh)
    return rhs(sysrhs, totaldofs)
end

function stress_rhs(sysrhs,mesh)
    dim = dimension(mesh)
    sdim = number_of_symmetric_degrees_of_freedom(dim)
    numnodes = number_of_nodes(mesh)
    totaldofs = sdim*numnodes
    return rhs(sysrhs,totaldofs)
end
