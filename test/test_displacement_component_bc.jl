using Test
using LinearAlgebra
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCell
include("useful_routines.jl")


L = 1.0
W = 1.0
lambda, mu = 1.0, 2.0
penaltyfactor = 1e2
nelmts = 1
dx = L / nelmts
dy = W / nelmts
penalty = penaltyfactor / dx * (lambda + mu)

applydisplacement = 0.01

stiffness = CutCell.HookeStiffness(lambda, mu, lambda, mu)
polyorder = 1
numqp = 2
basis = TensorProductBasis(2, polyorder)
mesh = CutCell.Mesh([0.0, 0.0], [L, W], [nelmts, nelmts], basis)
levelset = InterpolatingPolynomial(1, basis)
x0 = [0.5, 0.0]
normal = [1.0, 0.0]
levelsetcoeffs =
    CutCell.levelset_coefficients(x -> plane_distance_function(x, normal, x0), mesh)

cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
cellquads = CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
interfacequads = CutCell.InterfaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
facequads = CutCell.FaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)

bilinearforms = CutCell.BilinearForms(basis, cellquads, stiffness, cutmesh)
interfacecondition =
    CutCell.coherent_interface_condition(basis, interfacequads, stiffness, cutmesh, penalty)


onbottomboundary(x) = x[2] ≈ 0.0 ? true : false
onrightboundary(x) = x[1] ≈ L ? true : false
ontopboundary(x) = x[2] ≈ W ? true : false
onleftboundary(x) = x[1] ≈ 0.0 ? true : false

leftbc = CutCell.DisplacementComponentCondition(
    x -> 0.0,
    basis,
    facequads,
    stiffness,
    cutmesh,
    onleftboundary,
    [1.0, 0.0],
    penalty,
)
bottombc = CutCell.DisplacementComponentCondition(
    x -> 0.0,
    basis,
    facequads,
    stiffness,
    cutmesh,
    onbottomboundary,
    [0.0, 1.0],
    penalty,
)
rightbc = CutCell.DisplacementComponentCondition(
    x -> applydisplacement,
    basis,
    facequads,
    stiffness,
    cutmesh,
    onrightboundary,
    [1.0, 0.0],
    penalty,
)

sysmatrix = CutCell.SystemMatrix()
sysrhs = CutCell.SystemRHS()

CutCell.assemble_bilinear_form!(sysmatrix, bilinearforms, cutmesh)
CutCell.assemble_interface_condition!(sysmatrix,interfacecondition,cutmesh)
CutCell.assemble_penalty_displacement_bc!(sysmatrix,sysrhs,leftbc,cutmesh)
CutCell.assemble_penalty_displacement_bc!(sysmatrix,sysrhs,bottombc,cutmesh)
CutCell.assemble_penalty_displacement_bc!(sysmatrix,sysrhs,rightbc,cutmesh)

matrix = CutCell.make_sparse(sysmatrix,cutmesh)
rhs = CutCell.rhs(sysrhs,cutmesh)

sol = matrix\rhs

displacement = reshape(sol,2,:)

e11 = applydisplacement/L
e22 = -lambda/(lambda + 2mu)*e11
u2 = e22*W
u1 = applydisplacement

testdisplacement = [0.0  0.0  u1  u1  0.0  0.0  u1  u1
                    0.0  u2   0.0 u2  0.0  u2   0.0 u2]
@test allapprox(testdisplacement,displacement,1e3eps())


L = 1.0
W = 1.0
lambda, mu = 1.0, 2.0
penaltyfactor = 1e2
nelmts = 1
dx = L / nelmts
dy = W / nelmts
penalty = penaltyfactor / dx * (lambda + mu)

applydisplacement = 0.01

stiffness = CutCell.HookeStiffness(lambda, mu, lambda, mu)
polyorder = 1
numqp = 2
basis = TensorProductBasis(2, polyorder)
mesh = CutCell.Mesh([0.0, 0.0], [L, W], [nelmts, nelmts], basis)
levelset = InterpolatingPolynomial(1, basis)
x0 = [0.5, 0.0]
normal = [1.0, 0.0]
levelsetcoeffs =
    CutCell.levelset_coefficients(x -> plane_distance_function(x, normal, x0), mesh)

cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
cellquads = CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
interfacequads = CutCell.InterfaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
facequads = CutCell.FaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)

bilinearforms = CutCell.BilinearForms(basis, cellquads, stiffness, cutmesh)
interfacecondition =
    CutCell.coherent_interface_condition(basis, interfacequads, stiffness, cutmesh, penalty)


onbottomboundary(x) = x[2] ≈ 0.0 ? true : false
onrightboundary(x) = x[1] ≈ L ? true : false
ontopboundary(x) = x[2] ≈ W ? true : false
onleftboundary(x) = x[1] ≈ 0.0 ? true : false

leftbc = CutCell.DisplacementComponentCondition(
    x -> 0.0,
    basis,
    facequads,
    stiffness,
    cutmesh,
    onleftboundary,
    [1.0, 0.0],
    penalty,
)
bottombc = CutCell.DisplacementComponentCondition(
    x -> 0.0,
    basis,
    facequads,
    stiffness,
    cutmesh,
    onbottomboundary,
    [0.0, 1.0],
    penalty,
)

sysmatrix = CutCell.SystemMatrix()
sysrhs = CutCell.SystemRHS()

CutCell.assemble_bilinear_form!(sysmatrix, bilinearforms, cutmesh)
CutCell.assemble_interface_condition!(sysmatrix, interfacecondition, cutmesh)
CutCell.assemble_penalty_displacement_bc!(sysmatrix, sysrhs, leftbc, cutmesh)
CutCell.assemble_penalty_displacement_bc!(sysmatrix, sysrhs, bottombc, cutmesh)

CutCell.assemble_traction_force_component_linear_form!(
    sysrhs,
    x -> 1.0,
    basis,
    facequads,
    cutmesh,
    onrightboundary,
    [1.0, 0.0],
)

matrix = CutCell.make_sparse(sysmatrix,cutmesh)
rhs = CutCell.rhs(sysrhs,cutmesh)

sol = matrix\rhs

disp = reshape(sol,2,:)

m = (lambda+2mu)
e11 = 1/(m - lambda^2/m)
e22 = - lambda/m*e11
u1 = e11
u2 = e22

testdisplacement = [0.0  0.0  u1  u1  0.0  0.0  u1  u1
                    0.0  u2   0.0 u2  0.0  u2   0.0 u2]
@test allapprox(testdisplacement,disp,1e3eps())
