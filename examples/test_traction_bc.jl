# using PyPlot
# using LinearAlgebra
# using SparseArrays
using CSV, DataFrames
using CartesianMesh
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCell

function displacement(alpha, x::V) where {V<:AbstractVector}
    u1 = alpha * x[2] * sin(pi * x[1])
    u2 = alpha * (x[1]^3 + cos(pi * x[2]))
    return [u1, u2]
end

function displacement(alpha, x::M) where {M<:AbstractMatrix}
    npts = size(x)[2]
    return hcat([displacement(alpha, x[:, i]) for i = 1:npts]...)
end

function body_force(lambda, mu, alpha, x)
    b1 = alpha * (lambda + 2mu) * pi^2 * x[2] * sin(pi * x[1])
    b2 =
        -alpha * (6mu * x[1] + (lambda + mu) * pi * cos(pi * x[1])) +
        alpha * (lambda + 2mu) * pi^2 * cos(pi * x[2])
    return [b1, b2]
end

function boundary_nodeids(femesh)
    bn = CutCell.bottom_boundary_node_ids(femesh)
    rn = CutCell.right_boundary_node_ids(femesh)
    tn = CutCell.top_boundary_node_ids(femesh)
    ln = CutCell.left_boundary_node_ids(femesh)
    return unique!(vcat(bn, rn, tn, ln))
end

function add_cell_error_squared!(
    err,
    interpolater,
    exactsolution,
    cellmap,
    quad,
    detjac,
)
    for (p, w) in quad
        numsol = interpolater(p)
        exsol = exactsolution(cellmap(p))
        err .+= (numsol - exsol) .^ 2 * detjac * w
    end
end

function mesh_L2_error(
    nodalsolutions,
    exactsolution,
    nodalconnectivity,
    cellmaps,
    basis,
    errorquad,
)

    ndofs = size(nodalsolutions)[1]
    detjac = CutCell.determinant_jacobian(cellmaps[1])
    err = zeros(ndofs)
    interpolater = InterpolatingPolynomial(ndofs, basis)
    for (cellid, cellmap) in enumerate(cellmaps)
        nodeids = nodalconnectivity[:, cellid]
        elementsolutions = nodalsolutions[:, nodeids]
        update!(interpolater, elementsolutions)
        add_cell_error_squared!(
            err,
            interpolater,
            exactsolution,
            cellmap,
            errorquad,
            detjac,
        )
    end
    return sqrt.(err)
end

ne = 10
mesh = CutCell.Mesh([0.,0.],[1.,1.],[ne,ne],9)
