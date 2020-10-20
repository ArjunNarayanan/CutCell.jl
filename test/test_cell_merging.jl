using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("useful_routines.jl")


p = [0.0, 0.0]

mergemapper = CutCell.MergeMapper()

south = CutCell.cell_map_to_south()
@test allapprox(south(p), [0.0, -2.0])
@test allapprox(mergemapper[1](p), [0.0, -2.0])

southeast = CutCell.cell_map_to_south_east()
@test allapprox(southeast(p), [2.0, -2.0])
@test allapprox(mergemapper[5](p), [2.0, -2.0])

east = CutCell.cell_map_to_east()
@test allapprox(east(p), [2.0, 0.0])
@test allapprox(mergemapper[2](p), [2.0, 0.0])

northeast = CutCell.cell_map_to_north_east()
@test allapprox(northeast(p), [2.0, 2.0])
@test allapprox(mergemapper[6](p), [2.0, 2.0])

north = CutCell.cell_map_to_north()
@test allapprox(north(p), [0.0, 2.0])
@test allapprox(mergemapper[3](p), [0.0, 2.0])

northwest = CutCell.cell_map_to_north_west()
@test allapprox(northwest(p), [-2.0, 2.0])
@test allapprox(mergemapper[7](p), [-2.0, 2.0])

west = CutCell.cell_map_to_west()
@test allapprox(west(p), [-2.0, 0.0])
@test allapprox(mergemapper[4](p), [-2.0, 0.0])

southwest = CutCell.cell_map_to_south_west()
@test allapprox(southwest(p), [-2.0, -2.0])
@test allapprox(mergemapper[8](p), [-2.0, -2.0])




x0 = [1.1, 0.0]
normal = [1.0, 0.0]
mesh = CutCell.Mesh([0.0, 0.0], [2.0, 1.0], [2, 1], 4)
basis = TensorProductBasis(2,1)
levelset = InterpolatingPolynomial(1, basis)
levelsetcoeffs =
    CutCell.levelset_coefficients(x -> plane_distance_function(x, normal, x0), mesh)
cutmesh = CutCell.CutMesh(levelset,levelsetcoeffs,mesh)
cellquads = CutCell.CellQuadratures(levelset,levelsetcoeffs,cutmesh,2)

mergemapper = CutCell.MergeMapper()

CutCell.map_and_update_cell_quadrature!(cellquads,-1,2,mergemapper,2)

mergecutmesh = CutCell.MergeCutMesh(cutmesh)
CutCell.merge_cells(mergecutmesh,-1,1,2)
