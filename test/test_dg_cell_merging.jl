using Test
using LinearAlgebra
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCell
include("useful_routines.jl")


polyorder = 1
numqp = 2

basis = TensorProductBasis(2, polyorder)
levelset = InterpolatingPolynomial(1, basis)
mesh = CutCell.DGMesh([0.0, 0.0], [2.0, 1.0], [2, 1], basis)

normal = [1.0, 0.0]
x0 = [1.1, 0.0]
levelsetcoeffs =
    CutCell.levelset_coefficients(x -> plane_distance_function(x, normal, x0), mesh)

cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
cellquads = CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
interfacequads = CutCell.InterfaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
facequads = CutCell.FaceQuadratures(levelset,levelsetcoeffs,cutmesh,numqp)

mergecutmesh = CutCell.MergeCutMesh(cutmesh)
mergemapper = CutCell.MergeMapper()
@test CutCell.number_of_nodes(mergecutmesh) == 12

CutCell.merge_cells_in_mesh!(mergecutmesh,cellquads,interfacequads,facequads,mergemapper)
mergedmesh = CutCell.MergedMesh(mergecutmesh)
@test CutCell.number_of_nodes(mergedmesh) == 8

lambda, mu = (1.0, 2.0)
penalty = 1.0
dx = 0.1
e11 = dx / 2.0
e22 = -lambda / (lambda + 2mu) * e11
dy = e22
s11 = (lambda + 2mu) * e11 + lambda * e22

stiffness = CutCell.HookeStiffness(lambda, mu, lambda, mu)

bilinearforms = CutCell.BilinearForms(basis, cellquads, stiffness, mergedmesh)
interfacecondition =
    CutCell.coherent_interface_condition(basis, interfacequads, stiffness, mergedmesh, penalty)

sysmatrix = CutCell.SystemMatrix()
sysrhs = CutCell.SystemRHS()

CutCell.assemble_bilinear_form!(sysmatrix, bilinearforms, mergedmesh)
CutCell.assemble_interelement_condition!(sysmatrix,basis,facequads,stiffness,mergedmesh,penalty)
CutCell.assemble_interface_condition!(sysmatrix, interfacecondition, mergedmesh)

matrix = CutCell.make_sparse(sysmatrix, mergedmesh)
K = Array(matrix)
# rhs = CutCell.rhs(sysrhs, mergedmesh)
#
# CutCell.apply_dirichlet_bc!(matrix, rhs, [5,6], 1, 0.0, 2)
# CutCell.apply_dirichlet_bc!(matrix, rhs, [5], 2, 0.0, 2)
# CutCell.apply_dirichlet_bc!(matrix, rhs, [3,4], 1, dx, 2)
#
# sol = matrix \ rhs
# disp = reshape(sol, 2, :)
#
# testdisp = [dx/2  dx/2  dx  dx  0.  0.  dx/2  dx/2
#             0.    dy    0.  dy  0.  dy  0.    dy]
# @test allapprox(disp,testdisp,1e2eps())
