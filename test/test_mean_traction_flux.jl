using Test
using LinearAlgebra
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCell
include("useful_routines.jl")



polyorder = 1
numqp = required_quadrature_order(polyorder)
basis = TensorProductBasis(2,polyorder)
levelset = InterpolatingPolynomial(1,basis)
levelsetcoeffs = [-1.,-1.,1.,1.]
update!(levelset,levelsetcoeffs)

xL,xR = [-1.,-1.],[1.,1.]
pquad = area_quadrature(levelset,+1,xL,xR,numqp)
nquad = area_quadrature(levelset,-1,xL,xR,numqp)
squad = surface_quadrature(levelset,xL,xR,numqp)

lambda,mu = 1.,2.
penalty = 1.0
dx = 0.1
e11 = dx / 2.0
e22 = -lambda / (lambda + 2mu) * e11
dy = e22
penalty = 1.
stiffness = CutCell.plane_strain_voigt_hooke_matrix(lambda,mu)
cellmap = CutCell.CellMap([0.,0.],[2.,1.])

pbf = CutCell.bilinear_form(basis,pquad,stiffness,cellmap)
nbf = CutCell.bilinear_form(basis,nquad,stiffness,cellmap)

normals = repeat([1.,0.],1,numqp)
top = CutCell.coherent_traction_operator(basis,squad,normals,stiffness,cellmap)

facescale = CutCell.scale_area(cellmap,normals)
mm = penalty*CutCell.mass_matrix(basis,squad,facescale,2)

eta = -1.0
BF = [pbf         zeros(8,8)
      zeros(8,8)  nbf       ]
T1 = -0.5*[-top  -top
           +top  +top]
T2 = eta*T1'

MM = [+mm  -mm
      -mm  +mm]

K = BF+T1+T2+MM
@test norm(K-K') > 1.0

rhs = zeros(16)

CutCell.apply_dirichlet_bc!(K,rhs,[5,6],1,0.,2)
CutCell.apply_dirichlet_bc!(K,rhs,[5],2,0.,2)
CutCell.apply_dirichlet_bc!(K,rhs,[3,4],1,dx,2)

sol = K\rhs
disp = reshape(sol,2,:)

testdisp = [0.0 0.0 dx dx 0.0 0.0 dx dx
            0.0 dy  0. dy 0.  dy  0. dy]

@test allapprox(disp,testdisp,1e2*eps())




eta = 1.0
BF = [pbf         zeros(8,8)
      zeros(8,8)  nbf       ]
T1 = -0.5*[-top  -top
           +top  +top]
T2 = eta*T1'

MM = [+mm  -mm
      -mm  +mm]

K = BF+T1+T2+MM
@test isapprox(norm(K-K'),0.0,atol=1e1eps())

rhs = zeros(16)

CutCell.apply_dirichlet_bc!(K,rhs,[5,6],1,0.,2)
CutCell.apply_dirichlet_bc!(K,rhs,[5],2,0.,2)
CutCell.apply_dirichlet_bc!(K,rhs,[3,4],1,dx,2)

sol = K\rhs
disp = reshape(sol,2,:)

testdisp = [0.0 0.0 dx dx 0.0 0.0 dx dx
            0.0 dy  0. dy 0.  dy  0. dy]

@test allapprox(disp,testdisp,1e2*eps())


eta = 0.0
BF = [pbf         zeros(8,8)
      zeros(8,8)  nbf       ]
T1 = -0.5*[-top  -top
           +top  +top]
T2 = eta*T1'

MM = [+mm  -mm
      -mm  +mm]

K = BF+T1+T2+MM
@test norm(K-K') > 1.0

rhs = zeros(16)

CutCell.apply_dirichlet_bc!(K,rhs,[5,6],1,0.,2)
CutCell.apply_dirichlet_bc!(K,rhs,[5],2,0.,2)
CutCell.apply_dirichlet_bc!(K,rhs,[3,4],1,dx,2)

sol = K\rhs
disp = reshape(sol,2,:)

testdisp = [0.0 0.0 dx dx 0.0 0.0 dx dx
            0.0 dy  0. dy 0.  dy  0. dy]

@test allapprox(disp,testdisp,1e2*eps())
