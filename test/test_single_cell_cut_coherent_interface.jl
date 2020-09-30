using Test
using LinearAlgebra
using IntervalArithmetic
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell

function plane_distance_function(coords, normal, x0)
    return (coords .- x0)' * normal
end

function circle_distance_function(coords, center, radius)
    difference = coords .- center
    distance = [radius - norm(difference[:,i]) for i = 1:size(difference)[2]]
    return distance
end

function left_boundary_nodeids(nf)
    nfside = sqrt(nf)
    @assert isinteger(nfside)
    nfside = round(Int, nfside)
    return 1:nfside
end

function right_boundary_nodeids(nf)
    nfside = sqrt(nf)
    @assert isinteger(nfside)
    nfside = round(Int, nfside)
    start = 2nf-nfside+1
    return start:2nf
end

lambda = 1.0
mu = 2.0
stiffness = CutCell.plane_strain_voigt_hooke_matrix(lambda, mu)

polyorder = 1
numqp = 3
basis = TensorProductBasis(2, polyorder)
quad1d = ImplicitDomainQuadrature.ReferenceQuadratureRule(numqp)
levelset = InterpolatingPolynomial(1, basis)
nf = CutCell.number_of_basis_functions(basis)

cellmap = CutCell.CellMap([0.0, 0.0], [1.0, 1.0])
coords = cellmap(basis.points)

x0 = [0.5, 0.0]
normal = [1.0, 0.0]
levelsetcoeffs = plane_distance_function(coords, normal, x0)
# center = [1.5,0.5]
# radius = 1.0
# levelsetcoeffs = circle_distance_function(coords,center,radius)

update!(levelset, levelsetcoeffs)

box = IntervalBox(-1..1, 2)
pquad = quadrature(levelset, +1, false, box, quad1d)
nquad = quadrature(levelset, -1, false, box, quad1d)
squad = quadrature(levelset, +1, true, box, quad1d)

nbf = CutCell.bilinear_form(basis, nquad, stiffness, cellmap)
pbf = CutCell.bilinear_form(basis, pquad, stiffness, cellmap)

sysmatrix = CutCell.SystemMatrix()
sysrhs = CutCell.SystemRHS()

nnodeids = 1:nf
nedofs = CutCell.element_dofs(nnodeids, 2)
nr, nc = CutCell.element_dofs_to_operator_dofs(nedofs, nedofs)
CutCell.assemble!(sysmatrix, nr, nc, vec(nbf))

pnodeids = (nf+1):2nf
pedofs = CutCell.element_dofs(pnodeids, 2)
pr, pc = CutCell.element_dofs_to_operator_dofs(pedofs, pedofs)
CutCell.assemble!(sysmatrix, pr, pc, vec(pbf))

normals = repeat([1.0, 0.0], inner = (1, length(squad)))
iscale = CutCell.scale_area(cellmap, normals)

penalty = 1e3
massmatrix = penalty*CutCell.mass_matrix(basis, squad, iscale, 2)
CutCell.assemble_coherent_interface!(
    sysmatrix,
    massmatrix,
    nnodeids,
    pnodeids,
    2,
)

globalndofs = 4 * nf
matrix = CutCell.sparse(sysmatrix, globalndofs)
rhs = CutCell.rhs(sysrhs, globalndofs)

leftnodeids = left_boundary_nodeids(nf)
CutCell.apply_dirichlet_bc!(matrix, rhs, leftnodeids, 1, 0.0, 2)
CutCell.apply_dirichlet_bc!(matrix, rhs, [1], 2, 0.0, 2)

rightnodeids = right_boundary_nodeids(nf)
CutCell.apply_dirichlet_bc!(matrix,rhs,rightnodeids,1,0.01,2)

sol = matrix \ rhs
disp = reshape(sol,2,:)
