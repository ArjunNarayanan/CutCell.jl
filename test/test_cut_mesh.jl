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
cellmaps = CutCell.cell_maps(mesh)

normal = [1.0, 0.0]
x0 = [0.5, 0.0]
levelsetcoeffs = plane_distance_function(nodalcoordinates, normal, x0)

cellsign = CutCell.cell_sign(levelset, levelsetcoeffs, nodalconnectivity)
@test allequal(cellsign, [0, 1, 1])

posactivenodeids = CutCell.active_node_ids(+1, cellsign, nodalconnectivity)
@test allequal(posactivenodeids, 1:8)
negactivenodeids = CutCell.active_node_ids(-1, cellsign, nodalconnectivity)
@test allequal(negactivenodeids, 1:4)

totalnumnodes = CutCell.number_of_nodes(mesh)
cutmeshnodeids =
    CutCell.cut_mesh_nodeids(posactivenodeids, negactivenodeids, totalnumnodes)
testcutmeshnodeids = [
    1 2 3 4 5 6 7 8
    9 10 11 12 0 0 0 0
]
@test allequal(cutmeshnodeids, testcutmeshnodeids)


cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
@test allequal(CutCell.nodal_connectivity(cutmesh, +1, 1), [1, 2, 3, 4])
@test allequal(CutCell.nodal_connectivity(cutmesh, -1, 1), [9, 10, 11, 12])
