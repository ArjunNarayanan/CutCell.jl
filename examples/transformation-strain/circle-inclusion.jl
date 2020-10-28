using LinearAlgebra
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("../../test/useful_routines.jl")


lambda1, mu1 = 1.0, 2.0
lambda2, mu2 = 2.0, 4.0
theta0 = -0.067
stiffness = CutCell.HookeStiffness(lambda1, mu1, lambda2, mu2)
transfstress = CutCell.plane_strain_transformation_stress(lambda1, mu1, theta0)

L = 1.0
penaltyfactor = 1e2
nelmts = 4
dx = L / nelmts
penalty = penaltyfactor / dx * (lambda1 + mu1)

polyorder = 2
numqp = required_quadrature_order(polyorder) + 2

center = [0.5, 0.5]
radius = 0.25
basis = TensorProductBasis(2, polyorder)
mesh = CutCell.Mesh([0.0, 0.0], [L, L], [nelmts, nelmts], basis)
levelset = InterpolatingPolynomial(1, basis)
levelsetcoeffs =
    CutCell.levelset_coefficients(x -> -circle_distance_function(x, center, radius), mesh)

cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
cellquads = CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
interfacequads = CutCell.InterfaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
facequads = CutCell.FaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)

bilinearforms = CutCell.BilinearForms(basis, cellquads, stiffness, cutmesh)
interfacecondition =
    CutCell.InterfaceCondition(basis, interfacequads, stiffness, cutmesh, penalty)


sysmatrix = CutCell.SystemMatrix()
sysrhs = CutCell.SystemRHS()

CutCell.assemble_bilinear_form!(sysmatrix,bilinearforms,cutmesh)
CutCell.assemble_interface_condition!(sysmatrix, interfacecondition, cutmesh)
CutCell.assemble_bulk_transformation_linear_form!(
    sysrhs,
    transfstress,
    basis,
    cellquads,
    cutmesh,
)

matrix = CutCell.make_sparse(sysmatrix, cutmesh)
rhs = CutCell.rhs(sysrhs, cutmesh)

topleftnodeid = CutCell.nodes_per_mesh_side(mesh)[2]
CutCell.apply_dirichlet_bc!(matrix,rhs,[1,topleftnodeid],1,0.0,2)
CutCell.apply_dirichlet_bc!(matrix,rhs,[1],2,0.0,2)

sol = matrix\rhs
disp = reshape(sol,2,:)
