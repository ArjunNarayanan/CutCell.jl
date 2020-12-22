using Test
using PolynomialBasis
using ImplicitDomainQuadrature
#using Revise
using CutCell
include("useful_routines.jl")

polyorder = 1
numqp = 2
basis = TensorProductBasis(2, polyorder)
levelset = InterpolatingPolynomial(1, basis)
nf = CutCell.number_of_basis_functions(basis)
mesh = CutCell.Mesh([0.0, 0.0], [3.0, 1.0], [3, 1], nf)
nodalcoordinates = CutCell.nodal_coordinates(mesh)
nodalconnectivity = CutCell.nodal_connectivity(mesh)

normal = [1.0, 0.0]
x0 = [0.5, 0.0]
tol = 1e-3
perturbation = 1e-3
levelsetcoeffs = plane_distance_function(nodalcoordinates, normal, x0)

cellsign = CutCell.cell_sign!(levelset, levelsetcoeffs, nodalconnectivity, tol, perturbation)
cutmeshquads = CutCell.CellQuadratures(
    cellsign,
    levelset,
    levelsetcoeffs,
    nodalconnectivity,
    numqp,
    numqp
)
@test length(cutmeshquads.quads) == 3
testcelltoquad = [
    2 1 1
    3 0 0
]
@test allequal(cutmeshquads.celltoquad, testcelltoquad)
