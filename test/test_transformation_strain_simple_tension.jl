using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("useful_routines.jl")



L = 1.0
W = 1.0
lambda1, mu1 = 1.0, 2.0
K1 = (lambda1+2mu1/3)
lambda2, mu2 = 2.0, 4.0
theta0 = -0.067
e22 = K1*theta0/(lambda1+2mu1)
penaltyfactor = 1e2
nelmts = 1
dx = 1.0 / nelmts
penalty = penaltyfactor / dx * (lambda1 + mu1)
stiffness = CutCell.HookeStiffness(lambda1, mu1, lambda2, mu2)
transfstress = CutCell.plane_strain_transformation_stress(lambda1, mu1, theta0)


polyorder = 1
numqp = required_quadrature_order(polyorder) + 2

basis = TensorProductBasis(2, polyorder)
mesh = CutCell.Mesh([0.0, 0.0], [L, W], [nelmts, nelmts], basis)
levelset = InterpolatingPolynomial(1, basis)
levelsetcoeffs = ones(CutCell.number_of_nodes(mesh))

cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
cellquads = CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
interfacequads = CutCell.InterfaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
facequads = CutCell.FaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)

bilinearforms = CutCell.BilinearForms(basis, cellquads, stiffness, cutmesh)

sysmatrix = CutCell.SystemMatrix()
sysrhs = CutCell.SystemRHS()

CutCell.assemble_bilinear_form!(sysmatrix,bilinearforms,cutmesh)
CutCell.assemble_bulk_transformation_linear_form!(
    sysrhs,
    transfstress,
    basis,
    cellquads,
    cutmesh,
)

matrix = CutCell.make_sparse(sysmatrix, cutmesh)
rhs = CutCell.rhs(sysrhs, cutmesh)

CutCell.apply_dirichlet_bc!(matrix,rhs,[1,2],1,0.0,2)
CutCell.apply_dirichlet_bc!(matrix,rhs,[1],2,0.,2)
CutCell.apply_dirichlet_bc!(matrix,rhs,[3,4],1,0.0,2)

sol = matrix\rhs
disp = reshape(sol,2,:)

testdisp = [0.0  0.0  0.0  0.0
            0.0  e22  0.0  e22]
@test allapprox(disp,testdisp,1e2eps())
