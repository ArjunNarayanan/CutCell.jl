using Test
using LinearAlgebra
using CartesianMesh
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

function onleftboundary(x, L, W)
    return x[1] ≈ 0.0
end

function onbottomboundary(x, L, W)
    return x[2] ≈ 0.0
end

function onrightboundary(x, L, W)
    return x[1] ≈ L
end

function ontopboundary(x, L, W)
    return x[2] ≈ W
end

function onboundary(x, L, W)
    return onbottomboundary(x, L, W) ||
           onrightboundary(x, L, W) ||
           ontopboundary(x, L, W) ||
           onleftboundary(x, L, W)
end

function solve_plane_interface(x0,normal,nelmts,polyorder,numqp,penaltyfactor)
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
    levelsetcoeffs =
        CutCell.levelset_coefficients(x -> plane_distance_function(x, normal, x0), mesh)

    cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
    cellquads = CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
    interfacequads = CutCell.InterfaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
    facequads = CutCell.FaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)

    bilinearforms = CutCell.BilinearForms(basis, cellquads, stiffness, cutmesh)
    interfacecondition =
        CutCell.InterfaceCondition(basis, interfacequads, stiffness, cutmesh, penalty)


    bottomdisplacementbc = CutCell.DisplacementCondition(
        x -> displacement(alpha, x),
        basis,
        facequads,
        stiffness,
        cutmesh,
        x -> onbottomboundary(x, L, W),
        penalty,
    )
    rightdisplacementbc = CutCell.DisplacementComponentCondition(
        x -> displacement(alpha, x)[1],
        basis,
        facequads,
        stiffness,
        cutmesh,
        x -> onrightboundary(x, L, W),
        [1.0, 0.0],
        penalty,
    )
    topdisplacementbc = CutCell.DisplacementCondition(
        x -> displacement(alpha, x),
        basis,
        facequads,
        stiffness,
        cutmesh,
        x -> ontopboundary(x, L, W),
        penalty,
    )
    leftdisplacementbc = CutCell.DisplacementCondition(
        x -> displacement(alpha, x),
        basis,
        facequads,
        stiffness,
        cutmesh,
        x -> onleftboundary(x, L, W),
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
    CutCell.assemble_penalty_displacement_bc!(sysmatrix, sysrhs, bottomdisplacementbc, cutmesh)
    CutCell.assemble_penalty_displacement_bc!(sysmatrix, sysrhs, rightdisplacementbc, cutmesh)
    CutCell.assemble_penalty_displacement_bc!(sysmatrix, sysrhs, topdisplacementbc, cutmesh)
    CutCell.assemble_penalty_displacement_bc!(sysmatrix, sysrhs, leftdisplacementbc, cutmesh)
    CutCell.assemble_traction_force_component_linear_form!(
        sysrhs,
        x -> stress_field(lambda, mu, alpha, x)[3],
        basis,
        facequads,
        cutmesh,
        x -> onrightboundary(x, L, W),
        [0.0, 1.0],
    )


    matrix = CutCell.make_sparse(sysmatrix, cutmesh)
    rhs = CutCell.rhs(sysrhs, cutmesh)

    sol = matrix \ rhs
    disp = reshape(sol, 2, :)

    err = mesh_L2_error(disp, x -> displacement(alpha, x), basis, cellquads, cutmesh)

    return err
end

x0 = [0.5, 0.0]
normal = [1.0, 0.0]
penaltyfactor = 1e2
polyorder = 2
numqp = required_quadrature_order(polyorder) + 2
powers = 1:7
nelmts = [2^p+1 for p in powers]

err = [solve_plane_interface(x0,normal,ne,polyorder,numqp,penaltyfactor) for ne in nelmts]
