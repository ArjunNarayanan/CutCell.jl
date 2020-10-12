using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("useful_routines.jl")

polyorder = 1
numqp = 2

basis = TensorProductBasis(2, polyorder)
levelset = InterpolatingPolynomial(1, basis)
nf = CutCell.number_of_basis_functions(basis)
mesh = CutCell.Mesh([0.0, 0.0], [3.0, 1.0], [3, 1], nf)
nodalcoordinates = CutCell.nodal_coordinates(mesh)

normal = [1.0, 0.0]
x0 = [0.5, 0.0]
levelsetcoeffs = plane_distance_function(nodalcoordinates, normal, x0)

cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
cellquads = CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)

lambda, mu = (1.0, 2.0)
stiffness = CutCell.HookeStiffness(lambda,mu,lambda,mu)

bilinearforms =
    CutCell.BilinearForms(basis, cellquads, stiffness, cutmesh)

@test length(bilinearforms.cellmatrices) == 4
testcelltomatrix = [
    3 1 1
    4 0 0
]
@test allequal(testcelltomatrix, bilinearforms.celltomatrix)
