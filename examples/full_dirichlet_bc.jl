using CartesianMesh
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell

x0 = [0.0,0.0]
widths = [1.,1.]
nelements = [2,2]
polyorder = 1
numqp = 2
stiffness = plane_strain_voigt_hooke_matrix(1.,2.)
mesh = UniformMesh(x0,widths,nelements)
basis = TensorProductBasis(2,polyorder)
quad = tensor_product_quadrature(2,numqp)
femesh = CutCell.Mesh(mesh,basis)

cellmatrix = CutCell.bilinear_form(basis,quad,stiffness,CutCell.cellmap(femesh,1))
sysmatrix = CutCell.SystemMatrix()
CutCell.assemble_bilinear_form!(sysmatrix,cellmatrix,femesh.nodalconnectivity,2)
