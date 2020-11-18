using Test
using CartesianMesh
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCell




lambda,mu = 1.,2.
stiffness = plane_strain_voigt_hooke_matrix(lambda, mu)

penalty = 1e1
x0 = [0.0, 0.0]
widths = [1.0, 1.0]
nelements = [2,2]
polyorder = 1



basis = TensorProductBasis(2, polyorder)
mesh = CutCell.DGMesh(x0,widths,nelements,basis)

quad = tensor_product_quadrature(2, numqp)
errorquad = tensor_product_quadrature(2, numqp + 2)
