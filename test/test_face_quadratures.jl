using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("useful_routines.jl")

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

p = [1. 2. 3.]

bp = CutCell.extend_to_face(p,1)
testbp = vcat(p,-ones(3)')
@test allapprox(bp,testbp)

rp = CutCell.extend_to_face(p,2)
testrp = vcat(ones(3)',p)
@test allapprox(rp,testrp)

tp = CutCell.extend_to_face(p,3)
testtp = vcat(p,ones(3)')
@test allapprox(tp,testtp)

lp = CutCell.extend_to_face(p,4)
testlp = vcat(-ones(3)',p)
@test allapprox(testlp,lp)


polyorder = 1
numqp = 2
basis = TensorProductBasis(2, polyorder)
levelset = InterpolatingPolynomial(1, basis)
nf = CutCell.number_of_basis_functions(basis)
mesh = CutCell.Mesh([0.0, 0.0], [2.0, 1.0], [2, 1], nf)
nodalcoordinates = CutCell.nodal_coordinates(mesh)

normal = [1.0, 0.0]
x0 = [0.5, 0.0]
levelsetcoeffs = plane_distance_function(nodalcoordinates, normal, x0)

cutmesh = CutCell.CutMesh(levelset,levelsetcoeffs,mesh)
facequads = CutCell.FaceQuadratures(levelset,levelsetcoeffs,cutmesh,numqp)
idx = [5,6,7,8,9,10,11,12,1,2,3,4,0,0,0,0]
testfacetoquad = reshape(idx,4,2,2)
@test allequal(testfacetoquad,facequads.facetoquad)
