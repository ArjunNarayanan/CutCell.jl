using Test
using LinearAlgebra
using IntervalArithmetic
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("useful_routines.jl")

function plane_distance_function(coords, normal, x0)
    return (coords .- x0)' * normal
end

function circle_distance_function(coords, center, radius)
    difference = coords .- center
    distance = [radius - norm(difference[:, i]) for i = 1:size(difference)[2]]
    return distance
end

polyorder = 1
numqp = 2
basis = TensorProductBasis(2, polyorder)
levelset = InterpolatingPolynomial(1, basis)
nf = CutCell.number_of_basis_functions(basis)
mesh = CutCell.Mesh([0.0, 0.0], [3.0, 1.0], [3, 1], nf)
nodalcoordinates = CutCell.nodal_coordinates(mesh)
lambda, mu = (1.0, 2.0)
stiffness = CutCell.plane_strain_voigt_hooke_matrix(lambda, mu)
stiffnesses = [stiffness, stiffness]

normal = [1.0, 0.0]
x0 = [1.5, 0.0]
levelsetcoeffs = plane_distance_function(nodalcoordinates, normal, x0)

cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
cutmeshcellquads =
    CutCell.CutMeshCellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
cutmeshbf =
    CutCell.CutMeshBilinearForms(basis, cutmeshcellquads, stiffnesses, cutmesh)

sysmatrix = CutCell.SystemMatrix()
sysrhs = CutCell.SystemRHS()

CutCell.assemble_bilinear_form!(sysmatrix,cutmeshbf,cutmesh,2)
