using Test
using CartesianMesh
using PolynomialBasis
using ImplicitDomainQuadrature
#using Revise
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

function apply_displacement_boundary_condition!(
    matrix,
    rhs,
    displacement_function,
    femesh,
)

    boundarynodeids = boundary_nodeids(femesh)
    nodalcoordinates = CutCell.nodal_coordinates(femesh)
    boundarynodecoordinates = nodalcoordinates[:, boundarynodeids]
    boundarydisplacement = displacement_function(boundarynodecoordinates)
    CutCell.apply_dirichlet_bc!(
        matrix,
        rhs,
        boundarynodeids,
        boundarydisplacement,
    )
end

function linear_system(basis, quad, stiffness, femesh, bodyforcefunc)

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

    K = CutCell.make_sparse(sysmatrix, femesh)
    R = CutCell.rhs(sysrhs, femesh)

    return K, R
end

function error_for_num_elements(
    numelements,
    polyorder,
    numqp,
    lambda,
    mu,
    alpha,
)
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

    nodalconnectivity = CutCell.nodal_connectivity(femesh)
    cellmaps = CutCell.cell_maps(femesh)

    err = mesh_L2_error(
        nodalsolutions,
        x -> displacement(alpha, x),
        nodalconnectivity,
        cellmaps,
        basis,
        errorquad,
    )
    return err
end

function required_quadrature_order(polyorder)
    ceil(Int, 0.5 * (2polyorder + 1))
end

function convergence(nelements, polyorder, lambda, mu, alpha)
    numqp = required_quadrature_order(polyorder)
    err = zeros(length(nelements), 2)
    for (idx, nelement) in enumerate(nelements)
        err[idx, :] = error_for_num_elements(
            nelement,
            polyorder,
            numqp,
            lambda,
            mu,
            alpha,
        )
    end
    dx = 1.0 ./ nelements
    u1err = err[:, 1]
    u2err = err[:, 2]

    u1rate = sum(diff(log.(u1err)) ./ diff(log.(dx))) / (length(nelements) - 1)
    u2rate = sum(diff(log.(u2err)) ./ diff(log.(dx))) / (length(nelements) - 1)

    return round(u1rate, digits = 3), round(u2rate, digits = 3)
end

lambda = 1.0
mu = 2.0
alpha = 0.01

numelements = [2, 4, 8, 16, 32, 64]
u1rate, u2rate = convergence(numelements, 1, lambda, mu, alpha)
println("Convergence of linear elements : ", u1rate, "    ", u2rate)
@test isapprox(u1rate, 2.0, atol = 0.05)
@test isapprox(u2rate, 2.0, atol = 0.05)


u1rate, u2rate = convergence(numelements, 2, lambda, mu, alpha)
println("Convergence of quadratic elements : ", u1rate, "    ", u2rate)
@test isapprox(u1rate, 3.0, atol = 0.05)
@test isapprox(u2rate, 3.0, atol = 0.05)


u1rate, u2rate = convergence(numelements, 3, lambda, mu, alpha)
println("Convergence of cubic elements : ", u1rate, "    ", u2rate)
@test isapprox(u1rate, 4.0, atol = 0.05)
@test isapprox(u2rate, 4.0, atol = 0.05)


u1rate, u2rate = convergence(numelements, 4, lambda, mu, alpha)
println("Convergence of quartic elements : ", u1rate, "    ", u2rate)
@test isapprox(u1rate, 5.0, atol = 0.05)
@test isapprox(u2rate, 5.0, atol = 0.05)
