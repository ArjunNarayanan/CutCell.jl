using Test
using LinearAlgebra
using Plots
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("useful_routines.jl")


function radial_vector(r,theta)
    return r*[cosd(theta),sind(theta)]
end


xc = [0.0,0.0]
rad = 0.5
poly = InterpolatingPolynomial(1,2,3)
coeffs = circle_distance_function(poly.basis.points,xc,rad)
update!(poly,coeffs)


xq = [0.,0.]
x0 = [1.,1.]

xk,iter = CutCell.saye_newton_iterate(x0,xq,poly,1e-5,1.5sqrt(2))

xrange = -1:1e-2:1
contour(xrange,xrange,(x,y)->poly(x,y),levels=[0.],aspect_ratio=:equal)
xlims!(-1.2,1.2)
ylims!(-1.2,1.2)
scatter!([x0[1]],[x0[2]])
scatter!([xk[1]],[xk[2]])
scatter!([xq[1]],[xq[2]])

# vp = poly(x0)
# gp = vec(gradient(poly,x0))
# h = hessian(poly,x0)
# hp = [h[1]  h[2]
#       h[2]  h[3]]
#
# l0 = gp'*(xq-x0)/(gp'*gp)
#
# gf = vcat(x0-xq+l0*gp,vp)
# hf = [I+l0*hp  gp
#       gp'      0.0]
#
# δ = hf\gf
#
# x1 = x0 - δ[1:2]
# l1 = l0 - δ[3]
#
# vp = poly(x1)
# gp = vec(gradient(poly,x1))
# h = hessian(poly,x1)
# hp = [h[1]  h[2]
#       h[2]  h[3]]
#
# gf = vcat(x1-xq+l1*gp,vp)
# hf = [I+l1*hp  gp
#       gp'      0.0]
#
# δ = hf\gf
#
# x2 = x1 - δ[1:2]
# l2 = l1 - δ[3]



# xk = CutCell.newton_iterate(cubic_func,cubic_grad,[0.5,1.5],1e-10)
