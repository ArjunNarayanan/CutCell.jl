using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("../../test/useful_routines.jl")

polyorder = 1

distancefunction(x) = circle_distance_function(x,xc,radius)

L, W = 1.0, 1.0
basis = TensorProductBasis(2, polyorder)
numqp = required_quadrature_order(polyorder) + 2
quad = tensor_product_quadrature(2, numqp)
levelset = InterpolatingPolynomial(1, basis)

mesh = CutCell.Mesh([0.0, 0.0], [L, W], [nelmts, nelmts], basis)
levelsetcoeffs = CutCell.levelset_coefficients(distancefunction, mesh)
cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)

refseedpoints, spatialseedpoints, seedcellids =
    CutCell.seed_zero_levelset(2, levelset, levelsetcoeffs, cutmesh)

signeddistance = CutCell.reinitialize_levelset(
    refseedpoints,
    spatialseedpoints,
    seedcellids,
    levelset,
    levelsetcoeffs,
    cutmesh,
    1e-8,
    boundingradius = 4.5,
)
