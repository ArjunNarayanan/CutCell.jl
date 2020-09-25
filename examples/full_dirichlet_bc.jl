using PyPlot
using LinearAlgebra
using SparseArrays
using CartesianMesh
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
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

function cell_error_squared(
    interpolater,
    exactsolution,
    cellmap,
    quad,
    detjac,
    ndofs,
)
    err = zeros(ndofs)
    for (p, w) in quad
        numsol = interpolater(p)
        exsol = exactsolution(cellmap(p))
        err .+= (numsol - exsol) .^ 2 * detjac * w
    end
    return err
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
    return sqrt(val)
end

function apply_displacement_boundary_condition!(
    matrix,
    rhs,
    displacement_function,
    femesh,
)

    boundarynodeids = boundary_nodeids(femesh)
    nodalcoordinates = CutCell.nodal_coordinates(femesh)
    for (idx,nodeid) in enumerate(boundarynodeids)
        disp = displacement_function(nodalcoordinates[:,nodeid])
        CutCell.apply_dirichlet_bc!(matrix, rhs, nodeid, disp)
    end
end

function linear_system(basis, quad, stiffness, femesh, bodyforcefunc)

    cellmaps = CutCell.cell_maps(femesh)
    nodalconnectivity = CutCell.nodal_connectivity(femesh)

    sysmatrix = CutCell.SystemMatrix()
    sysrhs = CutCell.SystemRHS()

    cellmatrix = CutCell.bilinear_form_alternate(basis, quad, stiffness, cellmaps[1])
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
    K = CutCell.sparse(sysmatrix, ndofs)
    R = CutCell.rhs(sysrhs, ndofs)

    return K, R
end

function relative_error(nodalsolutions, exactsolution, basis, errorquad, femesh)

    nodalconnectivity = CutCell.nodal_connectivity(femesh)
    cellmaps = CutCell.cell_maps(femesh)

    err = mesh_error(
        nodalsolutions,
        exactsolution,
        nodalconnectivity,
        cellmaps,
        basis,
        errorquad,
    )
    normalizer =
        integrate_on_mesh(x -> norm(exactsolution(x))^2, cellmaps, errorquad)
    return err / normalizer
end

function error_for_num_elements(numelements, polyorder, numqp)
    x0 = [0.0, 0.0]
    widths = [1.0, 1.0]
    nelements = [numelements, numelements]
    mesh = UniformMesh(x0, widths, nelements)
    stiffness = plane_strain_voigt_hooke_matrix(lambda, mu)

    basis = TensorProductBasis(2, polyorder)
    quad = tensor_product_quadrature(2, numqp)
    errorquad = tensor_product_quadrature(2, numqp + 2)

    femesh = CutCell.Mesh(mesh, basis)

    matrix, rhs = linear_system(
        basis,
        quad,
        stiffness,
        femesh,
        x -> body_force(lambda, mu, alpha, x),
    )
    apply_displacement_boundary_condition!(
        matrix,
        rhs,
        x -> displacement(alpha, x),
        femesh,
    )

    sol = matrix \ rhs
    nodalsolutions = reshape(sol, 2, :)
    rerror = relative_error(
        nodalsolutions,
        x -> displacement(alpha, x),
        basis,
        errorquad,
        femesh,
    )
    return rerror
end

function shear_displacement(alpha, x::V) where {V<:AbstractVector}
    return alpha * [x[2], 0.0]
end

function shear_displacement(alpha, x::M) where {M<:AbstractMatrix}
    ux = alpha * x[2, :]
    uy = zeros(length(ux))
    return vcat(ux', uy')
end

function shear_body_force(lambda, mu, alpha, x)
    return [0.0, 0.0]
end

function quadratic_displacement(alpha,x)
    return alpha*[x[1]^2+x[2]^2, 2x[1]*x[2]]
end

function quadratic_body_force(lambda,mu,alpha,x)
    return -4alpha*[(lambda+2mu),0.0]
end

const lambda = 1.0
const mu = 2.0
const alpha = 0.01
polyorder = 2
numqp = 4

x0 = [0.0, 0.0]
widths = [1.0, 1.0]
numelements = 1
nelements = [numelements, numelements]
mesh = UniformMesh(x0, widths, nelements)
stiffness = plane_strain_voigt_hooke_matrix(lambda, mu)

basis = TensorProductBasis(2, polyorder)
quad = tensor_product_quadrature(2, numqp)
errorquad = tensor_product_quadrature(2, numqp + 2)

femesh = CutCell.Mesh(mesh, basis)
nodalcoordinates = CutCell.nodal_coordinates(femesh)
nodalconnectivity = CutCell.nodal_connectivity(femesh)
cellmaps = CutCell.cell_maps(femesh)

matrix, rhs = linear_system(
    basis,
    quad,
    stiffness,
    femesh,
    x -> quadratic_body_force(lambda, mu, alpha, x),
)
K = Array(matrix)

apply_displacement_boundary_condition!(
    matrix,
    rhs,
    x -> quadratic_displacement(alpha, x),
    femesh,
)

sol = matrix \ rhs
nodalsolutions = reshape(sol, 2, :)
exactsolution = hcat([quadratic_displacement(alpha,nodalcoordinates[:,i]) for i = 1:9]...)

err = mesh_error(
    nodalsolutions,
    x -> quadratic_displacement(alpha, x),
    nodalconnectivity,
    cellmaps,
    basis,
    errorquad,
)
println("Error = ", err)
