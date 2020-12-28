using Test
using LinearAlgebra
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("useful_routines.jl")



polyorder = 1
numqp = required_quadrature_order(polyorder)
basis = TensorProductBasis(2, polyorder)
levelset = InterpolatingPolynomial(1, basis)
levelsetcoeffs = [-1.0, -1.0, 1.0, 1.0]
update!(levelset, levelsetcoeffs)

xL, xR = [-1.0, -1.0], [1.0, 1.0]
pquad = area_quadrature(levelset, +1, xL, xR, numqp)
nquad = area_quadrature(levelset, -1, xL, xR, numqp)
squad = surface_quadrature(levelset, xL, xR, numqp)

lambda, mu = 1.0, 2.0
penalty = 1.0
dx = 0.1
e11 = dx / 2.0
e22 = -lambda / (lambda + 2mu) * e11
dy = e22
penalty = 1.0
stiffness = CutCell.plane_strain_voigt_hooke_matrix(lambda, mu)
cellmap = CutCell.CellMap([0.0, 0.0], [2.0, 1.0])

pbf = CutCell.bilinear_form(basis, pquad, stiffness, cellmap)
nbf = CutCell.bilinear_form(basis, nquad, stiffness, cellmap)

normals = repeat([1.0, 0.0], 1, numqp)
top =
    CutCell.incoherent_traction_operator(basis, squad, squad, normals, stiffness, cellmap)

facescale = CutCell.scale_area(cellmap, normals)
components = CutCell.tangents(normals)
mm =
    penalty *
    CutCell.interface_component_mass_matrix(basis, squad, squad, components, facescale)

eta = -1.0
BF = [
    pbf zeros(8, 8)
    zeros(8, 8) nbf
]
T1 = -0.5 * [
    -top -top
    +top +top
]
T2 = eta * T1'

MM = [
    +mm -mm
    -mm +mm
]

K = BF + T1 + T2 + MM
@test norm(K - K') > 1.0

rhs = zeros(16)

CutCell.apply_dirichlet_bc!(K, rhs, [5, 6], 1, 0.0, 2)
CutCell.apply_dirichlet_bc!(K, rhs, [5], 2, 0.0, 2)
CutCell.apply_dirichlet_bc!(K, rhs, [3, 4], 1, dx, 2)

sol = K \ rhs
disp = reshape(sol, 2, :)

testdisp = [
    0.0 0.0 dx dx 0.0 0.0 dx dx
    0.0 dy 0.0 dy 0.0 dy 0.0 dy
]

@test allapprox(disp, testdisp, 1e2 * eps())



###############################################################################
# Use different material properties

L = 1.0
W = 1.0
lambda1, mu1 = 1.0, 2.0
lambda2, mu2 = 3.0, 4.0
penaltyfactor = 1e2
nelmts = 1
dx = L / nelmts
dy = W / nelmts
penalty = penaltyfactor / dx * (lambda1 + mu1 + lambda2 + mu2)*0.5

applydisplacement = 0.01

stiffness = CutCell.HookeStiffness(lambda1, mu1, lambda2, mu2)
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
    CutCell.InterfaceIncoherentCondition(basis,interfacequads,stiffness,cutmesh,penalty)


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


E1 = mu1*(lambda1+mu1)/(lambda1+2mu1)
E2 = mu2*(lambda2+mu2)/(lambda2+2mu2)

uI = E1/(E1+E2)*applydisplacement

u10 = uI + (applydisplacement - uI)/0.5*(-0.5)
Δx = applydisplacement

u12 = -lambda1/(lambda1+2mu1)*2*(Δx - uI)
u22 = -lambda2/(lambda2+2mu2)*2*uI

testdisplacement = [u10   u10   Δx   Δx   0.0   0.0   2uI   2uI
                    0.0   u12   0.0  u12  0.0   u22   0.0   u22]

@test allapprox(displacement,testdisplacement,1e2eps())
