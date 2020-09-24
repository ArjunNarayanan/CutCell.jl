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
    numdofs = numnodes * dofspernode
    edofs = zeros(Int, numdofs)
    count = 1
    for nodeid in nodeids
        for dof = 1:dofspernode
            edofs[count] = node_to_dof_id(nodeid, dof, dofspernode)
            count += 1
        end
    end
    return edofs
end

function element_dofs_to_operator_dofs(rowdofs, coldofs)
    nr = length(rowdofs)
    nc = length(coldofs)
    rows = repeat(rowdofs, outer = nc)
    cols = repeat(coldofs, inner = nr)
    return rows, cols
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
        edofs = element_dofs(nodeids, dofspernode)
        rows, cols = element_dofs_to_operator_dofs(edofs, edofs)
        assemble!(sysmatrix, rows, cols, vals)
    end
end

function assemble_body_force_linear_form!(
    systemrhs::SystemRHS,
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

function apply_dirichlet_bc!(
    matrix::SparseMatrixCSC,
    rhs,
    index::Z,
    value,
) where {Z<:Integer}

    modifyrhs = matrix[:, index]
    for i in modifyrhs.nzind
        if i == index
            rhs[i] = modifyrhs[i] * value
        else
            rhs[i] -= modifyrhs[i] * value
            matrix[i, index] = 0.0
            matrix[index, i] = 0.0
        end
    end
end

function apply_dirichlet_bc!(
    matrix,
    rhs,
    nodeids::V,
    dofs,
    values,
) where {V<:AbstractVector}

    @assert length(nodeids) == length(dofs) == length(values)
    for (nodeid, dof, val) in zip(nodeids, dofs, values)
        index = node_to_dof_id(nodeid, dof)
        apply_dirichlet_bc!(matrix, rhs, index, val)
    end
end

function apply_dirichlet_bc!(
    matrix,
    rhs,
    nodeids::V,
    bcvals,
) where {V<:AbstractVector}

    dofspernode, numnodes = size(bcvals)
    @assert length(nodeids) == numnodes
    for (idx, nodeid) in enumerate(nodeids)
        for dof = 1:dofspernode
            index = node_to_dof_id(nodeid, dof, dofspernode)
            apply_dirichlet_bc!(matrix, rhs, index, bcvals[dof, idx])
        end
    end
end

function SparseArrays.sparse(sysmatrix::SystemMatrix, ndofs)
    return sparse(sysmatrix.rows, sysmatrix.cols, sysmatrix.vals, ndofs, ndofs)
end

function rhs(sysrhs::SystemRHS, ndofs)
    return Array(sparsevec(sysrhs.rows, sysrhs.vals, ndofs))
end
