using Test
using CartesianMesh
using Revise
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

coords = CutCell.nodal_coordinates(mesh.x0,mesh.widths,mesh.nelements,[5,5])
xrange = range(0.0,stop=1.,length=5)
ycoords = repeat(xrange,5)
xcoords = repeat(xrange,inner=5)
@test allapprox(coords,vcat(xcoords',ycoords'))

connectivity = CutCell.nodal_connectivity([5,5],3,9,mesh.nelements)
c1 = [1,2,3,6,7,8,11,12,13]
c2 = [3,4,5,8,9,10,13,14,15]
c3 = [11,12,13,16,17,18,21,22,23]
c4 = [13,14,15,18,19,20,23,24,25]
testconn = hcat(c1,c2,c3,c4)
@test allequal(connectivity,testconn)

cellconnectivity = CutCell.cell_connectivity(mesh)
testcellconn = [0  1  0  3
                3  4  0  0
                2  0  4  0
                0  0  1  2]
@test allequal(testcellconn,cellconnectivity)

mesh = UniformMesh([0.,0.],[1.,1.],[3,2])
connectivity = CutCell.nodal_connectivity([10,7],4,16,mesh.nelements)
testnodeconn = vcat(43:46,50:53,57:60,64:67)
@test allequal(connectivity[:,5],testnodeconn)

cellconnectivity = CutCell.cell_connectivity(mesh)
c1 = [0,3,2,0]
c2 = [1,4,0,0]
c3 = [0,5,4,1]
c4 = [3,6,0,2]
c5 = [0,0,6,3]
c6 = [5,0,0,4]
testconn = hcat(c1,c2,c3,c4,c5,c6)
@test allequal(cellconnectivity,testconn)


cm = CutCell.CellMap([0.,0.],[1.,1.])
@test CutCell.determinant_jacobian(cm) ≈ 0.25

cm = CutCell.CellMap([1.,1.],[5.,8.])
@test CutCell.determinant_jacobian(cm) ≈ 7.0

mesh = UniformMesh([0.,0.],[1.,1.],[3,2])
femesh = CutCell.Mesh(mesh,16)
testnodeconn = vcat(43:46,50:53,57:60,64:67)
@test allequal(testnodeconn,femesh.nodalconnectivity[:,5])
@test CutCell.nodes_per_element(femesh) == 16
@test CutCell.nodes_per_mesh_side(femesh) == [10,7]
@test CutCell.total_number_of_nodes(femesh) == 70

@test allequal(CutCell.bottom_boundary_node_ids(femesh),1:7:64)
@test allequal(CutCell.right_boundary_node_ids(femesh),64:70)
@test allequal(CutCell.top_boundary_node_ids(femesh),7:7:70)
@test allequal(CutCell.left_boundary_node_ids(femesh),1:7)

mesh = CutCell.Mesh([0.,0.],[4.,3.],[4,3],9)
isboundarycell = CutCell.is_boundary_cell(mesh)
testboundarycell = ones(Int,12)
testboundarycell[5] = testboundarycell[8] = 0
@test allequal(isboundarycell,testboundarycell)

isboundarycell = CutCell.is_boundary_cell(mesh.cellconnectivity)
@test allequal(isboundarycell,testboundarycell)

@test allapprox(CutCell.reference_bottom_face_midpoint(),[0.,-1.])
@test allapprox(CutCell.reference_right_face_midpoint(),[1.,0.])
@test allapprox(CutCell.reference_top_face_midpoint(),[0.,1.])
@test allapprox(CutCell.reference_left_face_midpoint(),[-1.,0.])

refmidpoints = CutCell.reference_face_midpoints()
cellmap = CutCell.CellMap([0.,0.],[1.,1.])
spmidpoints = cellmap.(refmidpoints)
@test length(spmidpoints) == 4
@test allapprox(spmidpoints[1],[0.5,0.])
@test allapprox(spmidpoints[2],[1.,0.5])
@test allapprox(spmidpoints[3],[0.5,1.])
@test allapprox(spmidpoints[4],[0.,0.5])
