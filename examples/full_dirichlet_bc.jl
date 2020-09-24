using PyPlot
using LinearAlgebra
using CartesianMesh
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell

x0 = [0.0, 0.0]
widths = [1.0, 1.0]
nelements = [10, 10]
polyorder = 2
numqp = 4
lambda = 1.0
mu = 2.0
alpha = 0.1
stiffness = plane_strain_voigt_hooke_matrix(1.0, 2.0)
mesh = UniformMesh(x0, widths, nelements)
basis = TensorProductBasis(2, polyorder)
quad = tensor_product_quadrature(2, numqp)
femesh = CutCell.Mesh(mesh, basis)
ndofs = CutCell.number_of_degrees_of_freedom(femesh)

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
    return vcat(bn, rn, tn, ln)
end

function cell_error(interpolater, exactsolution, cellmap, quad, detjac)
    err = 0.0
    for (p, w) in quad
        numsol = interpolater(p)
        exsol = exactsolution(cellmap(p))
        difference = numsol - exsol
        err += dot(difference, difference) * detjac * w
    end
    return err
end

function mesh_error(
    nodalsolutions,
    exactsolution,
    nodalconnectivity,
    cellmaps,
    basis,
    errorquad,
)

    ndofs = size(nodalsolutions)[1]
    detjac = CutCell.determinant_jacobian(cellmaps[1])
    err = 0.0
    interpolater = InterpolatingPolynomial(ndofs, basis)
    for (cellid, cellmap) in enumerate(cellmaps)
        nodeids = nodalconnectivity[:, cellid]
        elementsolutions = nodalsolutions[:, nodeids]
        update!(interpolater, elementsolutions)
        err +=
            cell_error(interpolater, exactsolution, cellmap, errorquad, detjac)
    end
    return err
end

function integrate_on_cell(func, cellmap, quad, detjac)
    val = 0.0
    for (p, w) in quad
        val += func(cellmap(p)) * detjac * w
    end
    return val
end

function integrate_on_mesh(func, cellmaps, quad)
    val = 0.0
    detjac = CutCell.determinant_jacobian(cellmaps[1])
    for cellmap in cellmaps
        val += integrate_on_cell(func, cellmap, quad, detjac)
    end
    return val
end

function apply_displacement_boundary_condition(
    matrix,
    rhs,
    displacement_function,
    femesh,
)

    boundarynodeids = boundary_nodeids(femesh)
    nodalcoordinates = CutCell.nodal_coordinates(femesh)
    boundarynodecoords = nodalcoordinates[:, boundarynodeids]
    disp = displacement_function(boundarynodecoords)
    CutCell.apply_dirichlet_bc!(matrix, rhs, boundarynodeids, disp)
end

function linear_system(
    basis,
    quad,
    stiffness,
    femesh,
    bodyforcefunc,
)

    cellmaps = CutCell.cell_maps(femesh)
    nodalconnectivity = CutCell.nodal_connectivity(femesh)

    sysmatrix = CutCell.SystemMatrix()
    sysrhs = CutCell.SystemRHS()

    cellmatrix = CutCell.bilinear_form(basis, quad, stiffness, cellmaps[1])
    CutCell.assemble_bilinear_form!(sysmatrix, cellmatrix, nodalconnectivity, 2)
    CutCell.assemble_body_force_linear_form!(
        sysrhs,
        bodyforcefunc,
        basis,
        quad,
        cellmaps,
        nodalconnectivity,
    )

    ndofs = CutCell.number_of_degrees_of_freedom(femesh)
    K = CutCell.sparse(sysmatrix,ndofs)
    R = CutCell.rhs(sysrhs,ndofs)

    return K,R
end

matrix,rhs = linear_system(basis,quad,stiffness,femesh,x->body_force(lambda,mu,alpha,x))

boundarynodeids = boundary_nodeids(femesh)
nodalcoordinates = CutCell.nodal_coordinates(femesh)
nodalconnectivity = CutCell.nodal_connectivity(femesh)
cellmaps = CutCell.cell_maps(femesh)
boundarynodecoords = nodalcoordinates[:, boundarynodeids]
disp = displacement(alpha, boundarynodecoords)

CutCell.apply_dirichlet_bc!(K, R, boundarynodeids, disp)
sol = K \ R
nodalsolutions = reshape(sol, 2, :)

errorquad = tensor_product_quadrature(2, numqp + 2)
err = mesh_error(
    nodalsolutions,
    x -> displacement(alpha, x),
    nodalconnectivity,
    cellmaps,
    basis,
    errorquad,
)
normalizer =
    integrate_on_mesh(x -> norm(displacement(alpha, x)), cellmaps, errorquad)
