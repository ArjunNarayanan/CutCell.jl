using Test
using CartesianMesh
using PolynomialBasis
using Revise
using CutCell
include("useful_routines.jl")



basis = TensorProductBasis(2,1)
mesh = UniformMesh([0.,0.],[2.,1.],[2,2])
cellmaps = CutCell.cell_maps(mesh)

nodalcoordinates = CutCell.dg_nodal_coordinates(cellmaps,basis.points)
testnodalcoordinates = [0.  0.  1.  1.  0.  0.  1.  1.  1.  1.  2.  2.  1.  1.  2.  2.
                        0.  .5  0.  .5  .5  1.  .5  1.  0.  .5  0.  .5  .5  1.  .5  1.]
@test allapprox(nodalcoordinates,testnodalcoordinates)

nodalconnectivity = CutCell.dg_nodal_connectivity(4,4)
testnodalconnectivity = [1  5  9   13
                         2  6  10  14
                         3  7  11  15
                         4  8  12  16]
@test allequal(nodalconnectivity,testnodalconnectivity)

dgmesh = CutCell.DGMesh([0.,0.],[2.,1.],[2,2],basis)
