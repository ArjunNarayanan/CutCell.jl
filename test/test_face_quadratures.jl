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

points = [1.0 2.0 3.0]
bp = vcat(points,-ones(3)')
rp = vcat(ones(3)',points)
tp = vcat(points,ones(3)')
lp = vcat(-ones(3)',points)

@test allapprox(CutCell.extend_to_face(points,1),bp)
@test allapprox(CutCell.extend_to_face(points,2),rp)
@test allapprox(CutCell.extend_to_face(points,3),tp)
@test allapprox(CutCell.extend_to_face(points,4),lp)

quad1d = tensor_product_quadrature(1,4)
points = quad1d.points
bp = vcat(points,-ones(4)')
rp = vcat(ones(4)',points)
tp = vcat(points,ones(4)')
lp = vcat(-ones(4)',points)
bq = CutCell.extend_to_face(quad1d,1)
rq = CutCell.extend_to_face(quad1d,2)
tq = CutCell.extend_to_face(quad1d,3)
lq = CutCell.extend_to_face(quad1d,4)
@test allapprox(bq.points,bp)
@test allapprox(rq.points,rp)
@test allapprox(tq.points,tp)
@test allapprox(lq.points,lp)
@test allapprox(bq.weights,quad1d.weights)
@test allapprox(rq.weights,quad1d.weights)
@test allapprox(tq.weights,quad1d.weights)
@test allapprox(lq.weights,quad1d.weights)

facequads = CutCell.face_quadratures(4)
@test allapprox(facequads[1].points,bp)
@test allapprox(facequads[2].points,rp)
@test allapprox(facequads[3].points,tp)
@test allapprox(facequads[4].points,lp)
@test allapprox(facequads[1].weights,quad1d.weights)
@test allapprox(facequads[2].weights,quad1d.weights)
@test allapprox(facequads[3].weights,quad1d.weights)
@test allapprox(facequads[4].weights,quad1d.weights)

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
idx = [5,9,6,10,7,11,8,12,1,0,2,0,3,0,4,0]
testfacetoquad = reshape(idx,2,4,2)
@test allequal(testfacetoquad,facequads.facetoquad)
