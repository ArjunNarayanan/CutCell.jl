using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("useful_routines.jl")

x0 = [1.,5.]
L, W = 4.0, 1.0
nelmtsx = 2
nelmtsy = 1
numghostlayers = 2
nodesperelement = 9

mesh = CutCell.Mesh(x0, [L, W], [nelmtsx, nelmtsy], nodesperelement)
paddedmesh = CutCell.BoundaryPaddedMesh(mesh,numghostlayers)

dx = 1.0
dy = 0.5

testbottomcoords = [1.  1.  2.  2.  3.  3.  4.  4.  5.  5.
                    4.  4.5 4.  4.5 4.  4.5 4.  4.5 4.  4.5]
testrightcoords = [6.  6.  6.  7.  7.  7.
                   5.  5.5 6.  5.  5.5 6.]
testtopcoords = [1.  1.  2.  2.  3.  3.  4.  4.  5.  5.
                 6.5 7.  6.5 7.  6.5 7.  6.5 7.  6.5 7.]
testleftcoords = [-1.  -1.  -1.  0.  0.  0.
                   5.   5.5  6.  5.  5.5 6.]

@test allapprox(paddedmesh.bottomghostcoords,testbottomcoords)
@test allapprox(paddedmesh.rightghostcoords,testrightcoords)
@test allapprox(paddedmesh.topghostcoords,testtopcoords)
@test allapprox(paddedmesh.leftghostcoords,testleftcoords)



x0 = [0.,0.]
L,W = 1.,1.
nelmtsx,nelmtsy = 2,2
numghostlayers = 1
polyorder = 2

xc = [0.5,0.5]
radius = 0.25

basis = TensorProductBasis(2,polyorder)
mesh = CutCell.Mesh(x0,[L,W],[nelmtsx,nelmtsy],basis)

levelset = InterpolatingPolynomial(1,basis)
levelsetcoeffs = CutCell.levelset_coefficients(x->circle_distance_function(x,xc,radius),mesh)

cutmesh = CutCell.CutMesh(levelset,levelsetcoeffs,mesh)
paddedmesh = CutCell.BoundaryPaddedMesh(cutmesh,numghostlayers)

refseedpoints,spatialseedpoints,seedcellids = CutCell.seed_zero_levelset(2,levelset,levelsetcoeffs,cutmesh)
