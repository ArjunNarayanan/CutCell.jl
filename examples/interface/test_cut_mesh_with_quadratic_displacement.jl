using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell

function circle_distance_function(coords, center, radius)
    diff2 = (coords .- center) .^ 2
    return sqrt.(mapslices(sum, diff2, dims = 1)') .- radius
end

lambda = 1.0
mu = 2.0
stiffness = CutCell.plane_strain_voigt_hooke_matrix(lambda, mu)

polyorder = 2
numqp = 4
numcutqp = 6
basis = TensorProductBasis(2, polyorder)
levelset = InterpolatingPolynomial(1, basis)
nf = CutCell.number_of_basis_functions(basis)

mesh = CutCell.Mesh([0.0, 0.0], [1.0, 1.0], [2, 2], nf)

center = [0.5, 0.5]
radius = 0.25
levelsetcoeffs = CutCell.levelset_coefficients(
    x -> circle_distance_function(x, center, radius),
    mesh,
)

cutmesh = CutCell.CutMesh(levelset,levelsetcoeffs,mesh)
cellquads = CutCell.CellQuadratures(levelset,levelsetcoeffs,cutmesh,numqp,numcutqp)
