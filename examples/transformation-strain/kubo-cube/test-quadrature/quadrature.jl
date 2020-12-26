using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("../../../../test/useful_routines.jl")




polyorder = 2
numqp = required_quadrature_order(polyorder)
nelmts = 1
width = 1.0
x0 = [0.0, 0.0]
corner = [0.5, 0.5]

basis = TensorProductBasis(2, polyorder)
mesh = CutCell.Mesh(x0, [width, width], [nelmts, nelmts], basis)
levelset = InterpolatingPolynomial(1, basis)
levelsetcoeffs =
    CutCell.levelset_coefficients(x -> -corner_distance_function(x, corner), mesh)

cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
cellquads = CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
interfacequads = CutCell.InterfaceQuadratures(levelset,levelsetcoeffs,cutmesh,numqp)
