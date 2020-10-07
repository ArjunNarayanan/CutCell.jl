using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
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
levelsetcoeffs = plane_distance_function(nodalcoordinates, normal, x0)

cellsign = CutCell.cell_sign(levelset, levelsetcoeffs, nodalconnectivity)
cellmap = CutCell.cell_map(mesh,1)

interfacequads = CutCell.InterfaceQuadratures(
    cellsign,
    levelset,
    levelsetcoeffs,
    nodalconnectivity,
    numqp,
    cellmap,
)
@test length(interfacequads.quads) == 1
@test allequal(interfacequads.celltoquad, [1, 0, 0])

@test length(interfacequads.quads) == 1
@test allequal(interfacequads.celltoquad, [1, 0, 0])

testnormals = repeat(normal,inner=(1,numqp))
@test allapprox(testnormals,CutCell.interface_normals(interfacequads,1))
