using WriteVTK
using Triangulate
using ConcaveHull
using Plots
using PolynomialBasis
using Revise
using CutCell
include("../../test/useful_routines.jl")

function my_field(x)
    return sin(x[1])*3x[2]^2
end

basis = TensorProductBasis(2, 2)
mesh = CutCell.Mesh([0.0, 0.0], [2.0, 1.0], [1, 1], basis)
levelset = InterpolatingPolynomial(1, basis)
xc = [2.0, 0.5]
radius = 0.5
levelsetcoeffs =
    CutCell.levelset_coefficients(x -> circle_distance_function(x, xc, radius), mesh)

numqp = 10
cutmesh = CutCell.CutMesh(levelset,levelsetcoeffs,mesh)
cellquads = CutCell.CellQuadratures(levelset,levelsetcoeffs,cutmesh,numqp)
interfacequads = CutCell.InterfaceQuadratures(levelset,levelsetcoeffs,cutmesh,numqp)
facequads = CutCell.FaceQuadratures(levelset,levelsetcoeffs,cutmesh,numqp)

positivepoints = cellquads[-1,1].points
negativepoints = cellquads[+1,1].points

points = [[positivepoints[1,i],positivepoints[2,i]] for i = 1:size(positivepoints)[2]]
hull = concave_hull(points)


positivetriin=Triangulate.TriangulateIO()
positivetriin.pointlist=positivepoints
(positivetriout, vorout)=triangulate("Q", positivetriin)

positiveconn = positivetriout.trianglelist
positivecells = [MeshCell(VTKCellTypes.VTK_TRIANGLE, positiveconn[:,i]) for i = 1:size(positiveconn)[2]]


vtmfile = vtk_multiblock("test")
vtkfile = vtk_grid(vtmfile,positivepoints,positivecells)
positivefield = vec(mapslices(my_field,positivepoints,dims=1))
vtkfile["field",VTKPointData()] = positivefield

outfiles = vtk_save(vtkfile)
