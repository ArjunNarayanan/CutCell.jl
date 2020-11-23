using Test
using LinearAlgebra
using PyPlot
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("useful_routines.jl")


function radial_vector(r,theta)
    return r*[cosd(theta),sind(theta)]
end

xL = [0.,0.]
xR = [2.,1.]
cellmap = CutCell.CellMap(xL,xR)
refcellmap = CutCell.CellMap([-1.,-1.],[1.,1.])

# cellmap = refcellmap

xc = [1,0.5]
rad = 0.5
poly = InterpolatingPolynomial(1,2,3)
points = cellmap(poly.basis.points)
coeffs = circle_distance_function(points,xc,rad)
update!(poly,coeffs)


xq = [1.0,0.5]
x0 = CutCell.inverse(cellmap,xc+radial_vector(rad,45))

# xk,iter = CutCell.saye_newton_iterate(x0,xq,poly,1e-10,1.5sqrt(2))
xk,iter = CutCell.saye_newton_iterate_with_cellmap(x0,xq,poly,cellmap,1e-5,1.5sqrt(2))

sx0 = cellmap(x0)
sxk = cellmap(xk)

xrange = -1:1e-1:1

referencepoints = vcat(repeat(xrange,inner=length(xrange))',repeat(xrange,outer=length(xrange))')
spatialpoints = cellmap(referencepoints)
vals = vec(mapslices(poly,referencepoints,dims=1))

fig,ax = PyPlot.subplots()
ax.tricontour(spatialpoints[1,:],spatialpoints[2,:],vals,levels=[0.0])
ax.scatter([sx0[1]],[sx0[2]],label="guess")
ax.scatter([xq[1]],[xq[2]],label="query")
ax.scatter([sxk[1]],[sxk[2]],label="soln")
ax.legend()
ax.set_aspect("equal")
fig
