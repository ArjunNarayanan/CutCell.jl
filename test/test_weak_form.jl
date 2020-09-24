using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
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
grad = CutCell.transform_gradient(gradient(basis,0.,0.),cellmap)
@test allapprox(grad[1,:],[-1/6,-1/2])
@test allapprox(grad[2,:],[-1/6,1/2])
@test allapprox(grad[3,:],[1/6,-1/2])
@test allapprox(grad[4,:],[1/6,1/2])

l,m = 1.,2.
stiffness = CutCell.plane_strain_voigt_hooke_matrix(l,m)
teststiffness = [5. 1. 0.
                 1. 5. 0.
                 0. 0. 2.]
@test allapprox(teststiffness,stiffness)

cellmap = CutCell.CellMap([0.,0.],[1.,1.])
quad = tensor_product_quadrature(2,2)
bf = CutCell.bilinear_form(basis,quad,stiffness,cellmap)
@test size(bf) == (8,8)

rhsfunc(x) = [1.,1.]
rhs = CutCell.linear_form(rhsfunc,basis,quad,cellmap)
@test allapprox(rhs,0.25*ones(8))
