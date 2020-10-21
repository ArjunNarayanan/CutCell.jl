using Test
using LinearAlgebra
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("useful_routines.jl")

function test_linear_cut_cell_assembly()
    polyorder = 1
    numqp = 2

    basis = TensorProductBasis(2, polyorder)
    levelset = InterpolatingPolynomial(1, basis)
    nf = CutCell.number_of_basis_functions(basis)
    mesh = CutCell.Mesh([0.0, 0.0], [2.0, 1.0], [1, 1], nf)

    normal = [1.0, 0.0]
    x0 = [0.5, 0.0]
    levelsetcoeffs =
        CutCell.levelset_coefficients(x -> plane_distance_function(x, normal, x0), mesh)

    cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
    cellquads = CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
    interfacequads = CutCell.InterfaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)

    lambda, mu = (1.0, 2.0)
    penalty = 1.0
    dx = 0.1
    e11 = dx / 2.0
    e22 = -lambda / (lambda + 2mu) * e11
    dy = e22
    s11 = (lambda + 2mu) * e11 + lambda * e22

    stiffness = CutCell.HookeStiffness(lambda, mu, lambda, mu)

    bilinearforms = CutCell.BilinearForms(basis, cellquads, stiffness, cutmesh)
    interfacecondition =
        CutCell.InterfaceCondition(basis, interfacequads, stiffness, cutmesh, penalty)

    sysmatrix = CutCell.SystemMatrix()
    sysrhs = CutCell.SystemRHS()

    CutCell.assemble_bilinear_form!(sysmatrix, bilinearforms, cutmesh)
    CutCell.assemble_interface_condition!(sysmatrix, interfacecondition, cutmesh)

    matrix = CutCell.make_sparse(sysmatrix, cutmesh)
    rhs = CutCell.rhs(sysrhs, cutmesh)

    CutCell.apply_dirichlet_bc!(matrix, rhs, 9, 0.0)
    CutCell.apply_dirichlet_bc!(matrix, rhs, 10, 0.0)
    CutCell.apply_dirichlet_bc!(matrix, rhs, 11, 0.0)

    CutCell.apply_dirichlet_bc!(matrix, rhs, 5, dx)
    CutCell.apply_dirichlet_bc!(matrix, rhs, 7, dx)

    sol = matrix \ rhs
    disp = reshape(sol, 2, :)

    testdisp = [
        0.0 0.0 dx dx 0.0 0.0 dx dx
        0.0 dy 0.0 dy 0.0 dy 0.0 dy
    ]
    @test allapprox(disp, testdisp, 1e2eps())
end


function test_curved_interface_assembly()
    polyorder = 2
    numqp = 4

    basis = TensorProductBasis(2, polyorder)
    levelset = InterpolatingPolynomial(1, basis)
    nf = CutCell.number_of_basis_functions(basis)
    mesh = CutCell.Mesh([0.0, 0.0], [2.0, 1.0], [1, 1], nf)

    xc = [2., 0.5]
    rad = 1.0
    levelsetcoeffs =
        CutCell.levelset_coefficients(x -> circle_distance_function(x, xc, rad), mesh)

    cutmesh = CutCell.CutMesh(levelset,levelsetcoeffs,mesh)
    cellquads = CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
    interfacequads = CutCell.InterfaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)

    lambda, mu = (1.0, 2.0)
    penalty = 1.0
    dx = 0.1
    e11 = dx / 2.0
    e22 = -lambda / (lambda + 2mu) * e11
    dy = e22
    s11 = (lambda + 2mu) * e11 + lambda * e22

    stiffness = CutCell.HookeStiffness(lambda, mu, lambda, mu)

    bilinearforms = CutCell.BilinearForms(basis, cellquads, stiffness, cutmesh)
    interfacecondition =
        CutCell.InterfaceCondition(basis, interfacequads, stiffness, cutmesh, penalty)

    sysmatrix = CutCell.SystemMatrix()
    sysrhs = CutCell.SystemRHS()

    CutCell.assemble_bilinear_form!(sysmatrix, bilinearforms, cutmesh)
    CutCell.assemble_interface_condition!(sysmatrix, interfacecondition, cutmesh)

    matrix = CutCell.make_sparse(sysmatrix, cutmesh)
    rhs = CutCell.rhs(sysrhs, cutmesh)

    CutCell.apply_dirichlet_bc!(matrix,rhs,[10,11,12],1,0.,2)
    CutCell.apply_dirichlet_bc!(matrix,rhs,[10],2,0.,2)
    CutCell.apply_dirichlet_bc!(matrix,rhs,[7,8,9],1,dx,2)

    sol = matrix\rhs
    disp = reshape(sol,2,:)

    testdisp = [
        0.0 0.0 0.0 dx/2 dx/2 dx/2 dx dx dx
        0.0 dy/2 dy 0.0 dy/2 dy 0.0 dy/2 dy
    ]
    testdisp = repeat(testdisp, outer = (1, 2))

    @test allapprox(disp,testdisp,1e2eps())
end


polyorder = 1
numqp = 2

basis = TensorProductBasis(2, polyorder)
levelset = InterpolatingPolynomial(1, basis)
nf = CutCell.number_of_basis_functions(basis)
mesh = CutCell.Mesh([0.0, 0.0], [2.0, 1.0], [1, 1], nf)

normal = [1.0, 0.0]
x0 = [0.5, 0.0]
levelsetcoeffs =
    CutCell.levelset_coefficients(x -> plane_distance_function(x, normal, x0), mesh)

cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
cellquads = CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
interfacequads = CutCell.InterfaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)

lambda, mu = (1.0, 2.0)
penalty = 1.0
dx = 0.1
e11 = dx / 2.0
e22 = -lambda / (lambda + 2mu) * e11
dy = e22
s11 = (lambda + 2mu) * e11 + lambda * e22

stiffness = CutCell.HookeStiffness(lambda, mu, lambda, mu)

bilinearforms = CutCell.BilinearForms(basis, cellquads, stiffness, cutmesh)
interfacecondition =
    CutCell.InterfaceCondition(basis, interfacequads, stiffness, cutmesh, penalty)

sysmatrix = CutCell.SystemMatrix()
sysrhs = CutCell.SystemRHS()

# CutCell.assemble_bilinear_form!(sysmatrix, bilinearforms, cutmesh)
# CutCell.assemble_interface_condition!(sysmatrix, interfacecondition, cutmesh)
#
# matrix = CutCell.make_sparse(sysmatrix, cutmesh)
# rhs = CutCell.rhs(sysrhs, cutmesh)
#
# CutCell.apply_dirichlet_bc!(matrix, rhs, 9, 0.0)
# CutCell.apply_dirichlet_bc!(matrix, rhs, 10, 0.0)
# CutCell.apply_dirichlet_bc!(matrix, rhs, 11, 0.0)
#
# CutCell.apply_dirichlet_bc!(matrix, rhs, 5, dx)
# CutCell.apply_dirichlet_bc!(matrix, rhs, 7, dx)
#
# sol = matrix \ rhs
# disp = reshape(sol, 2, :)
#
# testdisp = [
#     0.0 0.0 dx dx 0.0 0.0 dx dx
#     0.0 dy 0.0 dy 0.0 dy 0.0 dy
# ]
# @test allapprox(disp, testdisp, 1e2eps())




# test_linear_cut_cell_assembly()
# test_curved_interface_assembly()
