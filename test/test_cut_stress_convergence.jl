using Test
using LinearAlgebra
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCell
include("useful_routines.jl")

function displacement(alpha, x)
    u1 = alpha * x[2] * sin(pi * x[1])
    u2 = alpha * (x[1]^3 + cos(pi * x[2]))
    return [u1, u2]
end

function stress_field(lambda, mu, alpha, x)
    s11 =
        (lambda + 2mu) * alpha * pi * x[2] * cos(pi * x[1]) -
        lambda * alpha * pi * sin(pi * x[2])
    s22 =
        -(lambda + 2mu) * alpha * pi * sin(pi * x[2]) +
        lambda * alpha * pi * x[2] * cos(pi * x[1])
    s12 = alpha * mu * (3x[1]^2 + sin(pi * x[1]))
    return [s11, s22, s12]
end

function body_force(lambda, mu, alpha, x)
    b1 = alpha * (lambda + 2mu) * pi^2 * x[2] * sin(pi * x[1])
    b2 =
        -alpha * (6mu * x[1] + (lambda + mu) * pi * cos(pi * x[1])) +
        alpha * (lambda + 2mu) * pi^2 * cos(pi * x[2])
    return [b1, b2]
end

function onboundary(x, L, W)
    return x[2] ≈ 0.0 || x[1] ≈ L || x[2] ≈ W || x[1] ≈ 0.0
end

function solve_elasticity(distancefunc, nelmts, polyorder, numqp, penaltyfactor)
    L = 1.0
    W = 1.0
    lambda, mu = 1.0, 2.0
    dx = 1.0 / nelmts
    penalty = penaltyfactor / dx * (lambda + mu)
    alpha = 0.1
    stiffness = CutCell.HookeStiffness(lambda, mu, lambda, mu)

    basis = TensorProductBasis(2, polyorder)
    mesh = CutCell.Mesh([0.0, 0.0], [L, W], [nelmts, nelmts], basis)
    levelset = InterpolatingPolynomial(1, basis)
    levelsetcoeffs = CutCell.levelset_coefficients(distancefunc, mesh)

    cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
    cellquads = CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
    interfacequads = CutCell.InterfaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
    facequads = CutCell.FaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)

    bilinearforms = CutCell.BilinearForms(basis, cellquads, stiffness, cutmesh)
    interfacecondition =
        CutCell.coherent_interface_condition(basis, interfacequads, stiffness, cutmesh, penalty)

    displacementbc = CutCell.DisplacementCondition(
        x -> displacement(alpha, x),
        basis,
        facequads,
        stiffness,
        cutmesh,
        x -> onboundary(x, L, W),
        penalty,
    )

    sysmatrix = CutCell.SystemMatrix()
    sysrhs = CutCell.SystemRHS()

    CutCell.assemble_bilinear_form!(sysmatrix, bilinearforms, cutmesh)
    CutCell.assemble_interface_condition!(sysmatrix, interfacecondition, cutmesh)
    CutCell.assemble_body_force_linear_form!(
        sysrhs,
        x -> body_force(lambda, mu, alpha, x),
        basis,
        cellquads,
        cutmesh,
    )
    CutCell.assemble_penalty_displacement_bc!(sysmatrix, sysrhs, displacementbc, cutmesh)

    matrix = CutCell.make_sparse(sysmatrix, cutmesh)
    rhs = CutCell.rhs(sysrhs, cutmesh)

    sol = matrix \ rhs

    sysmatrix = CutCell.SystemMatrix()
    sysrhs = CutCell.SystemRHS()

    CutCell.assemble_stress_mass_matrix!(sysmatrix, basis, cellquads, cutmesh)
    CutCell.assemble_stress_linear_form!(sysrhs, basis, cellquads, stiffness, sol, cutmesh)

    matrix = CutCell.make_sparse_stress_operator(sysmatrix, cutmesh)
    rhs = CutCell.stress_rhs(sysrhs, cutmesh)

    stressvec = matrix \ rhs
    stress = reshape(stressvec, 3, :)
    err = mesh_L2_error(
        stress,
        x -> stress_field(lambda, mu, alpha, x),
        basis,
        cellquads,
        cutmesh,
    )

    return err
end

function mean(v)
    return sum(v) / length(v)
end

function convergence_rate(v, dx)
    diff(log.(v)) ./ diff(log.(dx))
end

x0 = [0.5, 0.0]
normal = [1.0, 0.0]
polyorder = 1
numqp = required_quadrature_order(polyorder) + 2
penaltyfactor = 1e2
powers = [3,4,5]
nelmts = [2^p + 1 for p in powers]
dx = 1.0 ./ nelmts


err = [
    solve_elasticity(
        x -> plane_distance_function(x, normal, x0),
        ne,
        polyorder,
        numqp,
        penaltyfactor,
    ) for ne in nelmts
]
serr = [[er[i] for er in err] for i = 1:3]

rates = [convergence_rate(v, dx) for v in serr]
@test all([allapprox(rates[i],repeat([1.5],length(rates[i])),0.05) for i = 1:3])


# powers = [3,4,5]
# nelmts = [2^p + 1 for p in powers]
# dx = 1.0 ./ nelmts
# polyorder = 2
# numqp = required_quadrature_order(polyorder) + 2
# err = [
#     solve_elasticity(
#         x -> plane_distance_function(x, normal, x0),
#         ne,
#         polyorder,
#         numqp,
#         penaltyfactor,
#     ) for ne in nelmts
# ]
# serr = [[er[i] for er in err] for i = 1:3]
# rates = [convergence_rate(v, dx) for v in serr]
# @test all([allapprox(rates[i],repeat([2.0],length(rates[i])),0.05) for i = 1:3])


powers = [3,4,5]
nelmts = [2^p + 1 for p in powers]
dx = 1.0 ./ nelmts
xc = [1.0, 0.5]
radius = 0.45
polyorder = 2
numqp = required_quadrature_order(polyorder) + 2
err = [
    solve_elasticity(
        x -> circle_distance_function(x, xc, radius),
        ne,
        polyorder,
        numqp,
        penaltyfactor,
    ) for ne in nelmts
]
serr = [[er[i] for er in err] for i = 1:3]
rates = [convergence_rate(v, dx) for v in serr]
@test all([all(rates[i] .> 1.95) for i = 1:3])


# powers = [2,3,4]
# nelmts = [2^p + 1 for p in powers]
# dx = 1.0 ./ nelmts
# polyorder = 3
# numqp = required_quadrature_order(polyorder) + 2
# err = [
#     solve_elasticity(
#         x -> plane_distance_function(x, normal, x0),
#         ne,
#         polyorder,
#         numqp,
#         penaltyfactor,
#     ) for ne in nelmts
# ]
# serr = [[er[i] for er in err] for i = 1:3]
# rates = [convergence_rate(v, dx) for v in serr]
# @test all([all(rates[i] .> 3.0) for i = 1:3])

powers = [3,4,5]
nelmts = [2^p + 1 for p in powers]
dx = 1.0 ./ nelmts
xc = [1.0, 0.5]
radius = 0.45
polyorder = 3
numqp = required_quadrature_order(polyorder) + 2
err = [
    solve_elasticity(
        x -> circle_distance_function(x, xc, radius),
        ne,
        polyorder,
        numqp,
        penaltyfactor,
    ) for ne in nelmts
]
serr = [[er[i] for er in err] for i = 1:3]
rates = [convergence_rate(v, dx) for v in serr]
@test all([all(rates[i] .> 2.95) for i = 1:3])
