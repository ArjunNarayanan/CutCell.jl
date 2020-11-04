using Test
using CartesianMesh
#using Revise
using CutCell
include("useful_routines.jl")

cm = CutCell.CellMap([0.,2.],[3.,3.])
@test CutCell.dimension(cm) == 2
@test CutCell.jacobian(cm) == [1.5,0.5]
@test allapprox(cm([0.,0.]),[1.5,2.5])
@test allapprox(cm([1.,0.]),[3.,2.5])

mesh = UniformMesh([0.,0.],[1.,1.],[2,2])
@test allequal(CutCell.elements_per_mesh_side(mesh),[2,2])
cellmaps = CutCell.cell_maps(mesh)
cm1 = CutCell.CellMap([0.,0.],[0.5,0.5])
cm2 = CutCell.CellMap([0.,0.5],[0.5,1.])
cm3 = CutCell.CellMap([0.5,0.],[1.,0.5])
cm4 = CutCell.CellMap([0.5,0.5],[1.,1.])
@test allequal(cellmaps,[cm1,cm2,cm3,cm4])

cm = CutCell.CellMap([0.,0.],[1.,1.])
@test CutCell.determinant_jacobian(cm) ≈ 0.25

cm = CutCell.CellMap([1.,1.],[5.,8.])
@test CutCell.determinant_jacobian(cm) ≈ 7.0

cm = CutCell.CellMap([2.,1.],[4.,2.])
@test allapprox(CutCell.inverse(cm,[3,1.5]),[0.,0.])
@test allapprox(CutCell.inverse(cm,[4,2.]),[1.,1.])
@test allapprox(CutCell.inverse(cm,[2,1]),[-1.,-1.])
