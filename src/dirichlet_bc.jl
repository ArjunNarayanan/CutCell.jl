function apply_dirichlet_bc!(
    matrix::SparseMatrixCSC,
    rhs,
    index::Z,
    value::R,
) where {Z<:Integer,R<:Real}

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
    matrix::Matrix,
    rhs,
    index::Z,
    value::R,
) where {Z<:Integer,R<:Real}

    m, n = size(matrix)
    modifyrhs = matrix[:, index]
    for i = 1:m
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
    dofspernode,
) where {V<:AbstractVector}

    @assert length(nodeids) == length(dofs) == length(values)
    for (nodeid, dof, val) in zip(nodeids, dofs, values)
        index = node_to_dof_id(nodeid, dof,dofspernode)
        apply_dirichlet_bc!(matrix, rhs, index, val)
    end
end

function apply_dirichlet_bc!(
    matrix,
    rhs,
    nodeids::V,
    dofs::Z,
    value::R,
    dofspernode,
) where {V<:AbstractVector,Z<:Integer,R<:Real}
    numnodes = length(nodeids)
    vecdofs = repeat([dofs],numnodes)
    vecvalues = repeat([value],numnodes)
    apply_dirichlet_bc!(matrix,rhs,nodeids,vecdofs,vecvalues,dofspernode)
end

function apply_dirichlet_bc!(
    matrix,
    rhs,
    nodeids::V,
    bcvals::M,
) where {V<:AbstractVector,M<:AbstractMatrix}

    dofspernode, numnodes = size(bcvals)
    @assert length(nodeids) == numnodes
    for (idx, nodeid) in enumerate(nodeids)
        for dof = 1:dofspernode
            index = node_to_dof_id(nodeid, dof, dofspernode)
            apply_dirichlet_bc!(matrix, rhs, index, bcvals[dof, idx])
        end
    end
end

function apply_dirichlet_bc!(
    matrix,
    rhs,
    nodeid::Z,
    bcvals::V,
) where {Z<:Integer,V<:AbstractVector}

    dofs = length(bcvals)
    for dof = 1:dofs
        index = node_to_dof_id(nodeid, dof, dofs)
        apply_dirichlet_bc!(matrix, rhs, index, bcvals[dof])
    end
end
