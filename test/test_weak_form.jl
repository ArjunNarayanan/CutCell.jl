using Test
using PolynomialBasis
using ImplicitDomainQuadrature
#using Revise
using CutCell
include("useful_routines.jl")

vectosymmconverter = CutCell.vector_to_symmetric_matrix_converter()
E1 = [
      1.0 0.0
      0.0 0.0
      0.0 1.0
]

E2 = [
      0.0 0.0
      0.0 1.0
      1.0 0.0
]

@test allapprox(vectosymmconverter[1], E1)
@test allapprox(vectosymmconverter[2], E2)

matrix = CutCell.make_row_matrix(E1,[1,2,3])
expE1 = hcat(E1,2*E1,3*E1)
@test allapprox(matrix,expE1)

matrix = CutCell.interpolation_matrix([1.,2.,3.],2)
testm = [1. 0. 2. 0. 3. 0.
         0. 1. 0. 2. 0. 3.]
@test allapprox(matrix,testm)

basis = TensorProductBasis(2,1)
cellmap = CutCell.CellMap([0.0,0.0],[3.,1.])
grad = CutCell.transform_gradient(gradient(basis,0.,0.),CutCell.jacobian(cellmap))
@test allapprox(grad[1,:],[-1/6,-1/2])
@test allapprox(grad[2,:],[-1/6,1/2])
@test allapprox(grad[3,:],[1/6,-1/2])
@test allapprox(grad[4,:],[1/6,1/2])

l,m = 1.,2.
stiffness = plane_strain_voigt_hooke_matrix(1.,2.)
cellmap = CutCell.CellMap([0.,0.],[1.,1.])
quad = tensor_product_quadrature(2,2)
bf = CutCell.bilinear_form(basis,quad,stiffness,CutCell.jacobian(cellmap))
@test size(bf) == (8,8)

rhsfunc(x) = [1.,1.]
rhs = CutCell.linear_form(rhsfunc,basis,quad,cellmap)
@test allapprox(rhs,0.25*ones(8))


basis = TensorProductBasis(2,2)
quad = tensor_product_quadrature(2,4)
cellmap = CutCell.CellMap([-1,-1.],[1.,1.])
M = CutCell.mass_matrix(basis,quad,cellmap,2)
rhsfunc(x) = [x[1]^2+x[2]^2,2x[1]*x[2]]
R = CutCell.linear_form(rhsfunc,basis,quad,cellmap)

sol = reshape(M\R,2,:)
coords = hcat([cellmap(basis.points[:,i]) for i = 1:9]...)
exactsol = hcat([rhsfunc(coords[:,i]) for i = 1:9]...)
@test allapprox(sol,exactsol,1e-10)


K = CutCell.bilinear_form(basis,quad,stiffness,CutCell.jacobian(cellmap))
bodyforce(x) = -4*(l+2m)*[1.,0.]
R = CutCell.linear_form(bodyforce,basis,quad,cellmap)

boundarynodeids = [1,4,7,8,9,6,3,2]
bcvals = exactsol[:,boundarynodeids]
CutCell.apply_dirichlet_bc!(K,R,boundarynodeids,bcvals)

sol = reshape(K\R,2,:)
@test allapprox(sol,exactsol,1e-15)

K = CutCell.bilinear_form(basis,quad,stiffness,cellmap)
bodyforce(x) = -4*(l+2m)*[1.,0.]
R = CutCell.linear_form(bodyforce,basis,quad,cellmap)

boundarynodeids = [1,4,7,8,9,6,3,2]
bcvals = exactsol[:,boundarynodeids]
CutCell.apply_dirichlet_bc!(K,R,boundarynodeids,bcvals)

sol = reshape(K\R,2,:)
@test allapprox(sol,exactsol,1e-15)
