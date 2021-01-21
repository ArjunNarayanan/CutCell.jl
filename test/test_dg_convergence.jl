using Test
using LinearAlgebra
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("useful_routines.jl")

function displacement(alpha, x)
    u1 = alpha * x[2] * sin(pi * x[1])
    u2 = alpha * (x[1]^3 + cos(pi * x[2]))
    return [u1, u2]
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

function solve_plane_interface(x0, normal, nelmts, polyorder, numqp, theta)
    L = 1.0
    W = 1.0
    lambda, mu = 1.0, 2.0
    dx = 1.0 / nelmts
    penalty = theta / dx * (lambda + mu)
    alpha = 0.1
    stiffness = CutCell.HookeStiffness(lambda, mu, lambda, mu)

    basis = TensorProductBasis(2, polyorder)
    mesh = CutCell.DGMesh([0.0, 0.0], [L, W], [nelmts, nelmts], basis)
    levelset = InterpolatingPolynomial(1, basis)
    levelsetcoeffs = CutCell.levelset_coefficients(
        x -> plane_distance_function(x, normal, x0),
        mesh,
    )

    cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
    cellquads =
        CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
    interfacequads =
        CutCell.InterfaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
    facequads =
        CutCell.FaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)

    bilinearforms = CutCell.BilinearForms(basis, cellquads, stiffness, cutmesh)
    interfacecondition = CutCell.coherent_interface_condition(
        basis,
        interfacequads,
        stiffness,
        cutmesh,
        penalty,
    )

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
    CutCell.assemble_interelement_condition!(
        sysmatrix,
        basis,
        facequads,
        stiffness,
        cutmesh,
        penalty,
    )
    CutCell.assemble_interface_condition!(
        sysmatrix,
        interfacecondition,
        cutmesh,
    )
    CutCell.assemble_body_force_linear_form!(
        sysrhs,
        x -> body_force(lambda, mu, alpha, x),
        basis,
        cellquads,
        cutmesh,
    )
    CutCell.assemble_penalty_displacement_bc!(
        sysmatrix,
        sysrhs,
        displacementbc,
        cutmesh,
    )

    matrix = CutCell.make_sparse(sysmatrix, cutmesh)
    rhs = CutCell.rhs(sysrhs, cutmesh)

    sol = matrix \ rhs
    disp = reshape(sol, 2, :)

    err = mesh_L2_error(
        disp,
        x -> displacement(alpha, x),
        basis,
        cellquads,
        cutmesh,
    )
    return err
end

function solve_curved_interface(
    xc,
    radius,
    nelmts,
    polyorder,
    numqp,
    penaltyfactor,
)
    L = 1.0
    W = 1.0
    lambda, mu = 1.0, 2.0
    dx = 1.0 / nelmts
    penalty = penaltyfactor / dx * (lambda + mu)
    alpha = 0.1
    stiffness = CutCell.HookeStiffness(lambda, mu, lambda, mu)

    basis = TensorProductBasis(2, polyorder)
    mesh = CutCell.DGMesh([0.0, 0.0], [L, W], [nelmts, nelmts], basis)
    levelset = InterpolatingPolynomial(1, basis)
    levelsetcoeffs = CutCell.levelset_coefficients(
        x -> circle_distance_function(x, xc, radius),
        mesh,
    )

    cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
    cellquads =
        CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
    interfacequads =
        CutCell.InterfaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
    facequads =
        CutCell.FaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)

    bilinearforms = CutCell.BilinearForms(basis, cellquads, stiffness, cutmesh)
    interfacecondition = CutCell.coherent_interface_condition(
        basis,
        interfacequads,
        stiffness,
        cutmesh,
        penalty,
    )
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
    CutCell.assemble_interelement_condition!(
        sysmatrix,
        basis,
        facequads,
        stiffness,
        cutmesh,
        penalty,
    )
    CutCell.assemble_interface_condition!(
        sysmatrix,
        interfacecondition,
        cutmesh,
    )
    CutCell.assemble_body_force_linear_form!(
        sysrhs,
        x -> body_force(lambda, mu, alpha, x),
        basis,
        cellquads,
        cutmesh,
    )
    CutCell.assemble_penalty_displacement_bc!(
        sysmatrix,
        sysrhs,
        displacementbc,
        cutmesh,
    )

    matrix = CutCell.make_sparse(sysmatrix, cutmesh)
    rhs = CutCell.rhs(sysrhs, cutmesh)

    sol = matrix \ rhs
    disp = reshape(sol, 2, :)

    err = mesh_L2_error(
        disp,
        x -> displacement(alpha, x),
        basis,
        cellquads,
        cutmesh,
    )
    return err
end


# x0 = [0.5, 0.0]
# normal = [1.0, 0.0]
# powers = [3, 4, 5]
# nelmts = [2^p + 1 for p in powers]
# polyorder = 1
# numqp = required_quadrature_order(polyorder)
# theta = 1e2
#
# err = [
#     solve_plane_interface(x0, normal, ne, polyorder, numqp, theta) for
#     ne in nelmts
# ]
# u1err = [er[1] for er in err]
# u2err = [er[2] for er in err]
# dx = 1.0 ./ nelmts
#
# u1rate = diff(log.(u1err)) ./ diff(log.(dx))
# u2rate = diff(log.(u2err)) ./ diff(log.(dx))
#
# @test allapprox(u1rate, repeat([2.0], length(u1rate)), 0.1)
# @test allapprox(u2rate, repeat([2.0], length(u2rate)), 0.1)


x0 = [0.8, 0.0]
interfaceangle = 20.
normal = [cosd(interfaceangle), sind(interfaceangle)]
powers = [3, 4, 5]
nelmts = [2^p + 1 for p in powers]
polyorder = 2
numqp = required_quadrature_order(polyorder)
theta = 1e2

err = [
    solve_plane_interface(x0, normal, ne, polyorder, numqp, theta) for
    ne in nelmts
]
u1err = [er[1] for er in err]
u2err = [er[2] for er in err]
dx = 1.0 ./ nelmts

u1rate = diff(log.(u1err)) ./ diff(log.(dx))
u2rate = diff(log.(u2err)) ./ diff(log.(dx))

@test allapprox(u1rate, repeat([3.0], length(u1rate)), 0.05)
@test allapprox(u2rate, repeat([3.0], length(u2rate)), 0.05)



# polyorder = 2
# numqp = 4
# theta = 1e2
# powers = [2,3,4]
# nelmts = [2^p + 1 for p in powers]
#
# err = [solve_plane_interface(x0, normal, ne, polyorder, numqp, theta) for ne in nelmts]
# u1err = [er[1] for er in err]
# u2err = [er[2] for er in err]
# dx = 1.0 ./ nelmts
#
# u1rate = diff(log.(u1err)) ./ diff(log.(dx))
# u2rate = diff(log.(u2err)) ./ diff(log.(dx))
#
# @test allapprox(u1rate,repeat([3.0],length(u1rate)),0.05)
# @test allapprox(u2rate,repeat([3.0],length(u2rate)),0.05)
#
#
#
#
#
# xc = [1.0, 0.5]
# radius = 0.45
# polyorder = 2
# penaltyfactor = 1e2
# powers = [3,4,5]
# nelmts = [2^p + 1 for p in powers]
# numqp = required_quadrature_order(polyorder) + 2
# nelmts = [2^p + 1 for p in powers]
#
# err = [solve_curved_interface(xc, radius, ne, polyorder, numqp, penaltyfactor) for ne in nelmts]
# u1err = [er[1] for er in err]
# u2err = [er[2] for er in err]
# dx = 1.0 ./ nelmts
#
# u1rate = diff(log.(u1err)) ./ diff(log.(dx))
# u2rate = diff(log.(u2err)) ./ diff(log.(dx))
#
# @test allapprox(u1rate,repeat([3.0],length(u1rate)),0.05)
# @test allapprox(u2rate,repeat([3.0],length(u2rate)),0.05)
#
#
#
#
#
#
# xc = [0.5, 0.5]
# radius = 0.25
# polyorder = 2
# penaltyfactor = 1e2
# powers = [3,4,5]
# nelmts = [2^p + 1 for p in powers]
# numqp = required_quadrature_order(polyorder) + 2
# nelmts = [2^p + 1 for p in powers]
#
# err = [solve_curved_interface(xc, radius, ne, polyorder, numqp, penaltyfactor) for ne in nelmts]
# u1err = [er[1] for er in err]
# u2err = [er[2] for er in err]
# dx = 1.0 ./ nelmts
#
# u1rate = diff(log.(u1err)) ./ diff(log.(dx))
# u2rate = diff(log.(u2err)) ./ diff(log.(dx))
#
# @test allapprox(u1rate,repeat([3.0],length(u1rate)),0.05)
# @test allapprox(u2rate,repeat([3.0],length(u2rate)),0.05)
