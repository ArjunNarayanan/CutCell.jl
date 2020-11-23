using Test
using LinearAlgebra
using PyPlot
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("useful_routines.jl")


function radial_vector(r, theta)
    return r * [cosd(theta), sind(theta)]
end

xL = [0.0, 0.0]
xR = [2.0, 1.0]
cellmap = CutCell.CellMap(xL, xR)

xc = [1, 0.5]
rad = 0.5
poly = InterpolatingPolynomial(1, 2, 3)
points = cellmap(poly.basis.points)
coeffs = circle_distance_function(points, xc, rad)
update!(poly, coeffs)

# xk,iter = CutCell.saye_newton_iterate(x0,xq,poly,1e-10,1.5sqrt(2))
# xk,iter = CutCell.saye_newton_iterate_with_cellmap(x0,xq,poly,cellmap,1e-5,1.5sqrt(2))

# sx0 = cellmap(x0)
# sxk = cellmap(xk)

L,W = 1.,1.
xc = [1.5,0.5]
rad = 1.0
polyorder = 2
nelmts = 2
basis = TensorProductBasis(2,polyorder)
levelset = InterpolatingPolynomial(1,basis)

mesh = CutCell.Mesh([0.,0.],[L,W],[nelmts,nelmts],basis)
levelsetcoeffs = CutCell.levelset_coefficients(x->circle_distance_function(x,xc,rad),mesh)
cutmesh = CutCell.CutMesh(levelset,levelsetcoeffs,mesh)

update!(levelset,levelsetcoeffs[CutCell.nodal_connectivity(cutmesh.mesh,1)])

refpoints = CutCell.reference_seed_points(2)
seedpoints,seedcellids = CutCell.seed_zero_levelset(2,levelset,levelsetcoeffs,cutmesh)

nodalcoordinates = CutCell.nodal_coordinates(cutmesh)
fig, ax = PyPlot.subplots()
ax.tricontour(nodalcoordinates[1, :], nodalcoordinates[2, :], levelsetcoeffs, levels = [0.0])
ax.scatter(seedpoints[1,:],seedpoints[2,:])
ax.set_aspect("equal")
fig
