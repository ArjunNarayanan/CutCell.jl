using Test
using PolynomialBasis
using ImplicitDomainQuadrature
include("plot_utils.jl")
using Revise
using CutCell

function displacement_field(x)
    u1 = x[1]^2 + 2x[1] * x[2]
    u2 = x[2]^2 + 3x[1]
    return [u1, u2]
end

function body_force(lambda, mu)
    b1 = -2 * (lambda + 2mu)
    b2 = -(4lambda + 6mu)
    return [b1, b2]
end

lambda = 1.0
mu = 2.0
stiffness = CutCell.plane_strain_voigt_hooke_matrix(lambda, mu)


polyorder = 2
numqp = 4
basis = TensorProductBasis(2, polyorder)
quad = tensor_product_quadrature(2, numqp)
mesh = CutCell.Mesh([0.0, 0.0], [1.0, 1.0], [2, 2], basis)
nodalcoordinates = CutCell.nodal_coordinates(mesh)

cellmap = CutCell.cell_map(mesh, 1)
bf = CutCell.bilinear_form(basis, quad, stiffness, cellmap)

sysmatrix = CutCell.SystemMatrix()
sysrhs = CutCell.SystemRHS()

CutCell.assemble_bilinear_form!(sysmatrix, bf, mesh)
CutCell.assemble_body_force_linear_form!(
    sysrhs,
    x -> body_force(lambda, mu),
    basis,
    quad,
    mesh,
)

matrix = CutCell.stiffness(sysmatrix,mesh)
rhs = CutCell.rhs(sysrhs,mesh)

boundarynodeids = CutCell.boundary_node_ids(mesh)
boundarynodecoords = nodalcoordinates[:, boundarynodeids]
boundarydisplacement =
    mapslices(displacement_field, boundarynodecoords, dims = 1)
CutCell.apply_dirichlet_bc!(matrix,rhs,boundarynodeids,boundarydisplacement)

sol = matrix\rhs
disp = reshape(sol,2,:)

testdisp = mapslices(displacement_field,nodalcoordinates,dims=1)
