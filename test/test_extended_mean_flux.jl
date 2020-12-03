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
levelsetcoeffs = [-0.1,-0.1,0.9,0.9]
update!(levelset,levelsetcoeffs)


xL,xR = [-1.,-1.],[1.,1.]
tpq = tensor_product_quadrature(2,numqp)
quad1d = ReferenceQuadratureRule(numqp)
pquad = area_quadrature(levelset,+1,xL,xR,quad1d)
nquad = area_quadrature(levelset,-1,xL,xR,quad1d)
nquad = QuadratureRule(nquad.points .+ [2.0,0.0],nquad.weights)
psquad = surface_quadrature(levelset,xL,xR,quad1d)
nsquad = QuadratureRule(psquad.points .+ [2.,0.],psquad.weights)

lambda,mu = 1.,2.
penalty = 1.0
dx = 0.1
e11 = dx / 2.0
e22 = -lambda / (lambda + 2mu) * e11
dy = e22
penalty = 1.
stiffness = CutCell.plane_strain_voigt_hooke_matrix(lambda,mu)
cellmap = CutCell.CellMap([0.,0.],[1.,1.])

tpbf = CutCell.bilinear_form(basis,tpq,stiffness,cellmap)
nbf = CutCell.bilinear_form(basis,nquad,stiffness,cellmap)
pbf = CutCell.bilinear_form(basis,pquad,stiffness,cellmap)

O = zeros(8,8)
BF = [pbf   O
      O     (tpbf+nbf)]

normals = repeat([1.,0.],1,numqp)
pptop = CutCell.coherent_traction_operator(basis,psquad,psquad,normals,stiffness,cellmap)
pntop = CutCell.coherent_traction_operator(basis,psquad,nsquad,normals,stiffness,cellmap)
nptop = CutCell.coherent_traction_operator(basis,nsquad,psquad,normals,stiffness,cellmap)
nntop = CutCell.coherent_traction_operator(basis,nsquad,nsquad,normals,stiffness,cellmap)

facescale = CutCell.scale_area(cellmap,normals)
ppmass = penalty*CutCell.interface_mass_matrix(basis,psquad,psquad,facescale)
pnmass = penalty*CutCell.interface_mass_matrix(basis,psquad,nsquad,facescale)
npmass = penalty*CutCell.interface_mass_matrix(basis,nsquad,psquad,facescale)
nnmass = penalty*CutCell.interface_mass_matrix(basis,nsquad,nsquad,facescale)

eta = 1.
T1 = -0.5*[-pptop  -pntop
            nptop   nntop]
T2 = -0.5*eta*[-pptop'  nptop'
               -pntop'  nntop']
M = [ppmass  -pnmass
     -npmass  nnmass]

K = BF + T1 + T2 + M
R = zeros(16)

CutCell.apply_dirichlet_bc!(K,R,[5,6],1,0.,2)
CutCell.apply_dirichlet_bc!(K,R,[5],2,0.,2)
CutCell.apply_dirichlet_bc!(K,R,[3,4],1,dx,2)

sol = K\R
disp = reshape(sol,2,:)

testdisp = [dx/2  dx/2  dx  dx  0.  0.  dx/2  dx/2
            0.    dy    0.  dy  0.  dy  0.    dy]
@test allapprox(disp,testdisp,1e2eps())
