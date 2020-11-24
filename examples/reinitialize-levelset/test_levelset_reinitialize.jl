using Test
using LinearAlgebra
using PyPlot
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("../../test/useful_routines.jl")


L, W = 1.0, 1.0
xc = [0.5, 0.5]
rad = 0.25
polyorder = 2
nelmts = 33
basis = TensorProductBasis(2, polyorder)
numqp = required_quadrature_order(polyorder) + 2
quad = tensor_product_quadrature(2, numqp)
levelset = InterpolatingPolynomial(1, basis)

mesh = CutCell.Mesh([0.0, 0.0], [L, W], [nelmts, nelmts], basis)
levelsetcoeffs =
    CutCell.levelset_coefficients(x -> circle_distance_function(x, xc, rad), mesh)
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

err = uniform_mesh_L2_error(
    signeddistance',
    x -> circle_distance_function(x, xc, rad)[1],
    basis,
    quad,
    cutmesh.mesh,
)
cellmaps = CutCell.cell_maps(cutmesh.mesh)
den = integral_norm_on_uniform_mesh(x->circle_distance_function(x,xc,rad),quad,cellmaps,1)

println("Error = ", err./den)
