using Test
using LinearAlgebra
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCell
include("useful_routines.jl")

function exact_linear_solution(x, e11, e22)
    return [x[1] * e11, x[2] * e22]
end

function quadratic_displacement(x)
    u1 = x[1]^2 + 2 * x[1] * x[2]
    u2 = x[2]^2 + 3x[1]
    return [u1, u2]
end

function quadratic_body_force(lambda, mu)
    b1 = -2 * (lambda + 2mu)
    b2 = -(4lambda + 6mu)
    return [b1, b2]
end

polyorder = 2
numqp = 3
nelmts = 11
L = 2.0
W = 1.0

lambda, mu = 1.0, 2.0
penalty = 1e1

dx = 0.1
e11 = dx / L
e22 = -lambda / (lambda + 2mu) * e11
dy = e22

stiffness = CutCell.HookeStiffness(lambda, mu, lambda, mu)

basis = TensorProductBasis(2, polyorder)
levelset = InterpolatingPolynomial(1, basis)
nf = CutCell.number_of_basis_functions(basis)
mesh = CutCell.Mesh([0.0, 0.0], [L, W], [nelmts, nelmts], nf)

normal = [1.0, 0.0]
x0 = [1.0, 0.0]
levelsetcoeffs =
    CutCell.levelset_coefficients(x -> plane_distance_function(x, normal, x0), mesh)

cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
cellquads = CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
interfacequads = CutCell.InterfaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
facequads = CutCell.FaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)

bilinearforms = CutCell.BilinearForms(basis, cellquads, stiffness, cutmesh)
interfacecondition =
    CutCell.InterfaceCondition(basis, interfacequads, stiffness, cutmesh, penalty)

onbottomboundary(x) = x[2] ≈ 0.0 ? true : false
onrightboundary(x) = x[1] ≈ L ? true : false
ontopboundary(x) = x[2] ≈ W ? true : false
onleftboundary(x) = x[1] ≈ 0.0 ? true : false
onboundary(x) =
    onbottomboundary(x) || onrightboundary(x) || ontopboundary(x) || onleftboundary(x)

displacementbc = CutCell.DisplacementCondition(
    x -> quadratic_displacement(x),
    basis,
    facequads,
    stiffness,
    cutmesh,
    onboundary,
    penalty,
)

sysmatrix = CutCell.SystemMatrix()
sysrhs = CutCell.SystemRHS()

CutCell.assemble_bilinear_form!(sysmatrix, bilinearforms, cutmesh)
CutCell.assemble_interface_condition!(sysmatrix, interfacecondition, cutmesh)
CutCell.assemble_body_force_linear_form!(
    sysrhs,
    x -> quadratic_body_force(lambda, mu),
    basis,
    cellquads,
    cutmesh,
)
CutCell.assemble_penalty_displacement_bc!(sysmatrix, sysrhs, displacementbc, cutmesh)

matrix = CutCell.make_sparse(sysmatrix, cutmesh)
rhs = CutCell.rhs(sysrhs, cutmesh)

sol = matrix \ rhs
disp = reshape(sol, 2, :)

err = mesh_L2_error(disp, x -> quadratic_displacement(x), basis, cellquads, cutmesh)

@test isapprox(norm(err), 0.0, atol = 1e-13)
