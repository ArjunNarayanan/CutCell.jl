using Printf
using CSV, DataFrames
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("../../test/useful_routines.jl")

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

function solve_elasticity(x0, normal, nelmts, polyorder, numqp, penaltyfactor; eta = 1.0)
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
    CutCell.assemble_interface_condition!(sysmatrix, interfacecondition, cutmesh, eta = eta)
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
    disp = reshape(sol, 2, :)

    err = mesh_L2_error(disp, x -> displacement(alpha, x), basis, cellquads, cutmesh)
    normalizer =
        integral_norm_on_cut_mesh(x -> displacement(alpha, x), cellquads, cutmesh, 2)
    return err ./ normalizer
end

function normal_from_angle(phi)
    return [cosd(phi), sind(phi)]
end

function write_convergence_data_to_file(x0, theta, nelmts, polyorder, numqp, penaltyfactor)
    normal = normal_from_angle(theta)
    errors =
        [solve_elasticity(x0, normal, ne, polyorder, numqp, penaltyfactor) for ne in nelmts]

    dx = 1.0 ./ nelmts
    u1err = [er[1] for er in errors]
    u1rate = [
        (log(u1err[i]) - log(u1err[i-1])) / (log(dx[i]) - log(dx[i-1]))
        for i = 2:length(nelmts)
    ]
    pushfirst!(u1rate, 0.0)

    u2err = [er[2] for er in errors]
    u2rate = [
        (log(u2err[i]) - log(u2err[i-1])) / (log(dx[i]) - log(dx[i-1]))
        for i = 2:length(nelmts)
    ]
    pushfirst!(u2rate, 0.0)

    df = DataFrame(
        [dx, u1err, u1rate, u2err, u2rate],
        ["Element Size", "U1 Error", "U1 Rate", "U2 Error", "U2 Rate"],
    )

    penaltystr = @sprintf "%1.1f" penaltyfactor

    foldername = "examples/cut-convergence/theta-" *
    string(round(Int, theta)) *
    "-penalty-" * penaltystr

    if !ispath(foldername)
        mkpath(foldername)
    end

    filename = foldername *
        "/polyorder_" *
        string(polyorder) *
        "_.csv"
    CSV.write(filename, df)
end



x0 = [0.5, 0.0]
theta = 0.0
normal = normal_from_angle(theta)
powers = 1:7
nelmts = [2^p + 1 for p in powers]
polyorder = 1
numqp = required_quadrature_order(polyorder)+2
penaltyfactor = 1e2
eta = 1.0

# errors = [solve_elasticity(x0,normal,ne,polyorder,numqp,penaltyfactor,eta=eta) for ne in nelmts]

# dx = 1.0 ./ nelmts
# u1err = [er[1] for er in errors]

# rate = diff(log.(u1err)) ./ diff(log.(dx))

write_convergence_data_to_file(x0, theta, nelmts, polyorder, numqp, penaltyfactor)
