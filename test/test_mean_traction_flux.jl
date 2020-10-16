using Test
using IntervalArithmetic
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("useful_routines.jl")



polyorder = 1
numqp = required_quadrature_order(polyorder)
basis = TensorProductBasis(2,polyorder)
levelset = InterpolatingPolynomial(1,basis)
levelsetcoeffs = [-1.,-1.,1.,1.]
update!(levelset,levelsetcoeffs)

box = IntervalBox(-1..1,2)
quad1d = ReferenceQuadratureRule(numqp)
pquad = area_quadrature(levelset,+1,box,quad1d)
nquad = area_quadrature(levelset,-1,box,quad1d)
squad = surface_quadrature(levelset,box,quad1d)

lambda,mu = 1.,2.
penalty = 1.
stiffness = CutCell.plane_strain_voigt_hooke_matrix(lambda,mu)
cellmap = CutCell.CellMap([0.,0.],[2.,1.])

pbf = CutCell.bilinear_form(basis,pquad,stiffness,cellmap)
nbf = CutCell.bilinear_form(basis,nquad,stiffness,cellmap)

normals = repeat([1.,0.],1,numqp)
top = CutCell.coherent_traction_operator(basis,squad,normals,stiffness,cellmap)


facescale = CutCell.scale_area(cellmap,normals)
mm = penalty*CutCell.mass_matrix(basis,squad,facescale,2)


eta = +1.0

BF = [nbf         zeros(8,8)
      zeros(8,8)  pbf       ]
T1 = 0.5*[-top'    top'
          -top'    top']
T2 = eta*0.5*[-top  -top
              +top  +top]
MM = [+mm  -mm
      -mm  +mm]

K = BF+T1+T2+MM
rhs = zeros(16)

CutCell.apply_dirichlet_bc!(K,rhs,[1,2],1,0.,2)
CutCell.apply_dirichlet_bc!(K,rhs,[1],2,0.,2)
CutCell.apply_dirichlet_bc!(K,rhs,[7,8],1,0.1,2)

sol = K\rhs
disp = reshape(sol,2,:)
