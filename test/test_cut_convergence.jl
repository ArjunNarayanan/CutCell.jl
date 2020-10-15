using Test
using LinearAlgebra
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("useful_routines.jl")

# function quadratic_displacement(x)
#     u1 = x[1]^2 + 2 * x[1] * x[2]
#     u2 = x[2]^2 + 3x[1]
#     return [u1, u2]
# end
#
# function quadratic_body_force(lambda, mu)
#     b1 = -2 * (lambda + 2mu)
#     b2 = -(4lambda + 6mu)
#     return [b1, b2]
# end

function nonpoly_displacement(alpha, x)
    u1 = alpha * x[2] * sin(pi * x[1])
    u2 = alpha * (x[1]^3 + cos(pi * x[2]))
    return [u1, u2]
end

function nonpoly_body_force(lambda, mu, alpha, x)
    b1 = alpha * (lambda + 2mu) * pi^2 * x[2] * sin(pi * x[1])
    b2 =
        -alpha * (6mu * x[1] + (lambda + mu) * pi * cos(pi * x[1])) +
        alpha * (lambda + 2mu) * pi^2 * cos(pi * x[2])
    return [b1, b2]
end

function onboundary(x, L, W)
    return x[2] ≈ 0.0 || x[1] ≈ L || x[2] ≈ W || x[1] ≈ 0.0
end

function solve_elasticity(x0, normal, nelmts, polyorder, numqp, theta)
    L = 1.0
    W = 1.0
    lambda, mu = 1.0, 2.0
    dx = 1.0/nelmts
    penalty = theta/dx*(lambda+mu)
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
        x -> nonpoly_displacement(alpha,x),
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
        x -> nonpoly_body_force(lambda, mu, alpha, x),
        basis,
        cellquads,
        cutmesh,
    )
    CutCell.assemble_penalty_displacement_bc!(sysmatrix,sysrhs,displacementbc,cutmesh)

    matrix = CutCell.make_sparse(sysmatrix, cutmesh)
    rhs = CutCell.rhs(sysrhs, cutmesh)

    sol = matrix \ rhs
    disp = reshape(sol, 2, :)

    err = mesh_L2_error(disp, x -> nonpoly_displacement(alpha,x), basis, cellquads, cutmesh)
    return err
end

function normal_from_angle(phi)
    return [cosd(phi),sind(phi)]
end

x0 = [0.7,0.0]
phi = 10.
normal = normal_from_angle(phi)
polyorder = 2
numqp = 3
theta = 0.1
err = solve_elasticity(x0,normal,5,polyorder,numqp,theta)




x0 = [0.5,0.0]
normal = [1.,0.]
powers = 1:7
nelmts = [2^p+1 for p in powers]
polyorder = 1
numqp = 5
theta = 0.1

err = [solve_elasticity(x0,normal,ne,polyorder,numqp,theta) for ne in nelmts]
u1err = [er[1] for er in err]
u2err = [er[2] for er in err]
dx = 1.0 ./ nelmts

using PyPlot
function plot_convergence(dx,er)
    fig,ax = PyPlot.subplots()
    ax.loglog(dx,er,"-o")
    ax.grid()
    fig
end

plot_convergence(dx,u1err)
plot_convergence(dx,u2err)

u1rate = sum(diff(log.(u1err)) ./ diff(log.(dx)))/(length(powers)-1)
u2rate = sum(diff(log.(u2err)) ./ diff(log.(dx)))/(length(powers)-1)
