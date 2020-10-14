using Test
using LinearAlgebra
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("useful_routines.jl")

function exact_linear_solution(x,e11,e22)
    return [x[1]*e11,x[2]*e22]
end

polyorder = 1
numqp = 3
nelmts = 5
L = 2.0
W = 1.0

lambda, mu = 1.0, 2.0
dx = 0.1
penalty = 1e0
e11 = dx/L
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


sysmatrix = CutCell.SystemMatrix()
sysrhs = CutCell.SystemRHS()

CutCell.assemble_bilinear_form!(sysmatrix, bilinearforms, cutmesh)
CutCell.assemble_interface_condition!(sysmatrix, interfacecondition, cutmesh)

onleftboundary(x) = x[1] == 0.0 ? true : false
CutCell.assemble_penalty_displacement_component_bc!(
    sysmatrix,
    sysrhs,
    basis,
    facequads,
    stiffness,
    cutmesh,
    onleftboundary,
    [1.0, 0.0],
    0.0,
    penalty,
)
onbottomboundary(x) = x[2] == 0.0 ? true : false
CutCell.assemble_penalty_displacement_component_bc!(
    sysmatrix,
    sysrhs,
    basis,
    facequads,
    stiffness,
    cutmesh,
    onbottomboundary,
    [0.0, 1.0],
    0.0,
    penalty,
)
onrightboundary(x) = x[1] ≈ L ? true : false
CutCell.assemble_penalty_displacement_component_bc!(
    sysmatrix,
    sysrhs,
    basis,
    facequads,
    stiffness,
    cutmesh,
    onrightboundary,
    [1.0, 0.0],
    dx,
    penalty,
)

matrix = CutCell.make_sparse(sysmatrix,cutmesh)
rhs = CutCell.rhs(sysrhs,cutmesh)

sol = matrix\rhs
disp = reshape(sol,2,:)

err = mesh_L2_error(disp,x->exact_linear_solution(x,e11,e22),basis,cellquads,cutmesh)
