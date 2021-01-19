using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("useful_routines.jl")

x0 = [0.0, 0.0]
widths = [4.0, 1.0]
nelements = [2, 2]

interfacepoint = [-1.0, 0.0]
interfacenormal = [1.0, 0.0]

lambda, mu = 1.0, 2.0
stiffness = CutCell.HookeStiffness(lambda, mu, lambda, mu)
penalty = 1.0
dx = 0.1
e11 = dx / 4.0
e22 = -lambda / (lambda + 2mu) * e11
dy = e22

polyorder = 2
numqp = required_quadrature_order(polyorder)

basis = TensorProductBasis(2, polyorder)
mesh = CutCell.DGMesh(x0, widths, nelements, basis)

levelset = InterpolatingPolynomial(1, basis)
levelsetcoeffs = CutCell.levelset_coefficients(
    x -> plane_distance_function(x, interfacenormal, interfacepoint),
    mesh,
)

cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)

cellquads = CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
facequads = CutCell.FaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)

bilinearforms = CutCell.BilinearForms(basis, cellquads, stiffness, cutmesh)

leftbc = CutCell.DisplacementComponentCondition(
    x -> 0.0,
    basis,
    facequads,
    stiffness,
    cutmesh,
    x -> x[1] ≈ 0.0,
    [1.0, 0.0],
    penalty,
)
bottombc = CutCell.DisplacementComponentCondition(
    x -> 0.0,
    basis,
    facequads,
    stiffness,
    cutmesh,
    x -> x[2] ≈ 0.0,
    [0.0, 1.0],
    penalty,
)
rightbc = CutCell.DisplacementComponentCondition(
    x -> dx,
    basis,
    facequads,
    stiffness,
    cutmesh,
    x -> x[1] ≈ widths[1],
    [1.0, 0.0],
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
CutCell.assemble_penalty_displacement_bc!(sysmatrix,sysrhs,leftbc,cutmesh)
CutCell.assemble_penalty_displacement_bc!(sysmatrix,sysrhs,bottombc,cutmesh)
CutCell.assemble_penalty_displacement_bc!(sysmatrix,sysrhs,rightbc,cutmesh)

op = CutCell.make_sparse(sysmatrix, cutmesh)
rhs = CutCell.rhs(sysrhs, cutmesh)

sol = op \ rhs
disp = reshape(sol, 2, :)

nodalcoordinates = CutCell.nodal_coordinates(cutmesh)
testdisp = copy(nodalcoordinates)
testdisp[1, :] .*= e11
testdisp[2, :] .*= e22

@test allapprox(disp, testdisp, 1e2eps())
