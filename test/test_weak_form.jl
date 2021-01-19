using Test
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
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

matrix = CutCell.make_row_matrix(E1, [1, 2, 3])
expE1 = hcat(E1, 2 * E1, 3 * E1)
@test allapprox(matrix, expE1)

matrix = CutCell.interpolation_matrix([1.0, 2.0, 3.0], 2)
testm = [
    1.0 0.0 2.0 0.0 3.0 0.0
    0.0 1.0 0.0 2.0 0.0 3.0
]
@test allapprox(matrix, testm)

basis = TensorProductBasis(2, 1)
cellmap = CutCell.CellMap([0.0, 0.0], [3.0, 1.0])
grad = CutCell.transform_gradient(gradient(basis, 0.0, 0.0), CutCell.jacobian(cellmap))
@test allapprox(grad[1, :], [-1 / 6, -1 / 2])
@test allapprox(grad[2, :], [-1 / 6, 1 / 2])
@test allapprox(grad[3, :], [1 / 6, -1 / 2])
@test allapprox(grad[4, :], [1 / 6, 1 / 2])

l, m = 1.0, 2.0
stiffness = plane_strain_voigt_hooke_matrix(1.0, 2.0)
cellmap = CutCell.CellMap([0.0, 0.0], [1.0, 1.0])
quad = tensor_product_quadrature(2, 2)
bf = CutCell.bilinear_form(basis, quad, stiffness, CutCell.jacobian(cellmap))
@test size(bf) == (8, 8)

rhsfunc(x) = [1.0, 1.0]
rhs = CutCell.linear_form(rhsfunc, basis, quad, cellmap)
@test allapprox(rhs, 0.25 * ones(8))


basis = TensorProductBasis(2, 2)
quad = tensor_product_quadrature(2, 4)
cellmap = CutCell.CellMap([-1, -1.0], [1.0, 1.0])
detjac = CutCell.determinant_jacobian(cellmap)
M = CutCell.mass_matrix(basis, quad, detjac, 2)
rhsfunc(x) = [x[1]^2 + x[2]^2, 2x[1] * x[2]]
R = CutCell.linear_form(rhsfunc, basis, quad, cellmap)

sol = reshape(M \ R, 2, :)
coords = hcat([cellmap(basis.points[:, i]) for i = 1:9]...)
exactsol = hcat([rhsfunc(coords[:, i]) for i = 1:9]...)
@test allapprox(sol, exactsol, 1e3eps())


K = CutCell.bilinear_form(basis, quad, stiffness, CutCell.jacobian(cellmap))
bodyforce(x) = -4 * (l + 2m) * [1.0, 0.0]
R = CutCell.linear_form(bodyforce, basis, quad, cellmap)

boundarynodeids = [1, 4, 7, 8, 9, 6, 3, 2]
bcvals = exactsol[:, boundarynodeids]
CutCell.apply_dirichlet_bc!(K, R, boundarynodeids, bcvals)

sol = reshape(K \ R, 2, :)
@test allapprox(sol, exactsol, 1e3eps())

K = CutCell.bilinear_form(basis, quad, stiffness, cellmap)
bodyforce(x) = -4 * (l + 2m) * [1.0, 0.0]
R = CutCell.linear_form(bodyforce, basis, quad, cellmap)

boundarynodeids = [1, 4, 7, 8, 9, 6, 3, 2]
bcvals = exactsol[:, boundarynodeids]
CutCell.apply_dirichlet_bc!(K, R, boundarynodeids, bcvals)

sol = reshape(K \ R, 2, :)
@test allapprox(sol, exactsol, 1e3eps())


polyorder = 1
numqp = 2
basis = TensorProductBasis(2, polyorder)
quad = tensor_product_quadrature(2, numqp)
lambda, mu = 1.0, 2.0
penalty = 1e0
dx = 0.1
e11 = dx / 1.0
e22 = -lambda / (lambda + 2mu) * e11
dy = e22

cellmap = CutCell.CellMap([0.0, 0.0], [1.0, 1.0])
stiffness = plane_strain_voigt_hooke_matrix(lambda, mu)
facequads = CutCell.face_quadratures(numqp)
facedetjac = CutCell.face_determinant_jacobian(cellmap)
bf = CutCell.bilinear_form(basis, quad, stiffness, cellmap)

ftop1 = CutCell.face_traction_component_operator(
    basis,
    facequads[1],
    [0.,1.],
    [0.,-1.],
    stiffness,
    facedetjac[1],
    cellmap,
)
ftop2 = CutCell.face_traction_component_operator(
    basis,
    facequads[2],
    [1.,0.],
    [1.,0.],
    stiffness,
    facedetjac[2],
    cellmap,
)
ftop4 = CutCell.face_traction_component_operator(
    basis,
    facequads[4],
    [1.,0.],
    [-1.0, 0],
    stiffness,
    facedetjac[4],
    cellmap,
)

bcop1 = penalty*CutCell.component_mass_matrix(basis,facequads[1],[0.,1.],facedetjac[1])
bcop2 = penalty*CutCell.component_mass_matrix(basis,facequads[2],[1.,0.],facedetjac[2])
bcop4 = penalty*CutCell.component_mass_matrix(basis,facequads[4],[1.,0.],facedetjac[4])
bcrhs2 = penalty*CutCell.component_linear_form(dx,basis,facequads[2],[1.,0.],facedetjac[2])

K = bf - ftop1 - ftop2 - ftop4 + bcop1 + bcop2 + bcop4
rhs = bcrhs2

sol = K\rhs
disp = reshape(sol,2,:)
testdisp = [0. 0. dx dx
            0. dy 0. dy]
@test allapprox(disp,testdisp,1e3eps())
