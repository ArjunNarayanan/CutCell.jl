using Test
using LinearAlgebra
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("useful_routines.jl")


p = [0.0, 0.0]

mergemapper = CutCell.MergeMapper()

south = CutCell.cell_map_to_south()
@test allapprox(south(p), [0.0, -2.0])
@test allapprox(mergemapper[1](p), [0.0, -2.0])

southeast = CutCell.cell_map_to_south_east()
@test allapprox(southeast(p), [2.0, -2.0])
@test allapprox(mergemapper[5](p), [2.0, -2.0])

east = CutCell.cell_map_to_east()
@test allapprox(east(p), [2.0, 0.0])
@test allapprox(mergemapper[2](p), [2.0, 0.0])

northeast = CutCell.cell_map_to_north_east()
@test allapprox(northeast(p), [2.0, 2.0])
@test allapprox(mergemapper[6](p), [2.0, 2.0])

north = CutCell.cell_map_to_north()
@test allapprox(north(p), [0.0, 2.0])
@test allapprox(mergemapper[3](p), [0.0, 2.0])

northwest = CutCell.cell_map_to_north_west()
@test allapprox(northwest(p), [-2.0, 2.0])
@test allapprox(mergemapper[7](p), [-2.0, 2.0])

west = CutCell.cell_map_to_west()
@test allapprox(west(p), [-2.0, 0.0])
@test allapprox(mergemapper[4](p), [-2.0, 0.0])

southwest = CutCell.cell_map_to_south_west()
@test allapprox(southwest(p), [-2.0, -2.0])
@test allapprox(mergemapper[8](p), [-2.0, -2.0])



function test_linear_merge_cell_assembly()

    polyorder = 1
    numqp = 2

    basis = TensorProductBasis(2, polyorder)
    levelset = InterpolatingPolynomial(1, basis)
    nf = CutCell.number_of_basis_functions(basis)
    mesh = CutCell.Mesh([0.0, 0.0], [2.0, 1.0], [2, 1], nf)

    normal = [1.0, 0.0]
    x0 = [1.1, 0.0]
    levelsetcoeffs =
        CutCell.levelset_coefficients(x -> plane_distance_function(x, normal, x0), mesh)

    cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
    cellquads = CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
    interfacequads = CutCell.InterfaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)

    mergecutmesh = CutCell.MergeCutMesh(cutmesh)
    mergemapper = CutCell.MergeMapper()
    @test CutCell.number_of_nodes(mergecutmesh) == 10

    CutCell.merge_cells_in_mesh!(mergecutmesh,cellquads,interfacequads,mergemapper)
    mergedmesh = CutCell.MergedMesh(mergecutmesh)
    @test CutCell.number_of_nodes(mergedmesh) == 8

    lambda, mu = (1.0, 2.0)
    penalty = 1.0
    dx = 0.1
    e11 = dx / 2.0
    e22 = -lambda / (lambda + 2mu) * e11
    dy = e22
    s11 = (lambda + 2mu) * e11 + lambda * e22

    stiffness = CutCell.HookeStiffness(lambda, mu, lambda, mu)

    bilinearforms = CutCell.BilinearForms(basis, cellquads, stiffness, mergedmesh)
    interfacecondition =
        CutCell.InterfaceCondition(basis, interfacequads, stiffness, mergedmesh, penalty)

    sysmatrix = CutCell.SystemMatrix()
    sysrhs = CutCell.SystemRHS()

    CutCell.assemble_bilinear_form!(sysmatrix, bilinearforms, mergedmesh)
    CutCell.assemble_interface_condition!(sysmatrix, interfacecondition, mergedmesh)

    matrix = CutCell.make_sparse(sysmatrix, mergedmesh)
    rhs = CutCell.rhs(sysrhs, mergedmesh)

    CutCell.apply_dirichlet_bc!(matrix, rhs, 9, 0.0)
    CutCell.apply_dirichlet_bc!(matrix, rhs, 10, 0.0)
    CutCell.apply_dirichlet_bc!(matrix, rhs, 11, 0.0)

    CutCell.apply_dirichlet_bc!(matrix, rhs, 5, dx)
    CutCell.apply_dirichlet_bc!(matrix, rhs, 7, dx)

    sol = matrix \ rhs
    disp = reshape(sol, 2, :)

    testdisp = [dx/2  dx/2  dx  dx  0.  0.  dx/2  dx/2
                0.    dy    0.  dy  0.  dy  0.    dy]
    @test allapprox(disp,testdisp,1e2eps())
end


function test_curved_merge_assembly()
    polyorder = 2
    numqp = 4

    basis = TensorProductBasis(2, polyorder)
    levelset = InterpolatingPolynomial(1, basis)
    nf = CutCell.number_of_basis_functions(basis)
    mesh = CutCell.Mesh([0.0, 0.0], [2.0, 1.0], [2, 1], nf)

    xc = [2.0, 0.5]
    rad = 0.9
    levelsetcoeffs =
        CutCell.levelset_coefficients(x -> circle_distance_function(x, xc, rad), mesh)

    cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
    cellquads = CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
    interfacequads = CutCell.InterfaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)

    mergemapper = CutCell.MergeMapper()
    mergecutmesh = CutCell.MergeCutMesh(cutmesh)
    @test CutCell.number_of_nodes(mergecutmesh) == 24

    CutCell.merge_cells_in_mesh!(mergecutmesh,cellquads,interfacequads,mergemapper)
    mergedmesh = CutCell.MergedMesh(mergecutmesh)
    @test CutCell.number_of_nodes(mergedmesh) == 18

    lambda, mu = (1.0, 2.0)
    penalty = 1.0
    dx = 0.1
    e11 = dx / 2.0
    e22 = -lambda / (lambda + 2mu) * e11
    dy = e22
    s11 = (lambda + 2mu) * e11 + lambda * e22

    stiffness = CutCell.HookeStiffness(lambda, mu, lambda, mu)

    bilinearforms = CutCell.BilinearForms(basis, cellquads, stiffness, mergedmesh)
    interfacecondition =
        CutCell.InterfaceCondition(basis, interfacequads, stiffness, mergedmesh, penalty)

    sysmatrix = CutCell.SystemMatrix()
    sysrhs = CutCell.SystemRHS()

    CutCell.assemble_bilinear_form!(sysmatrix, bilinearforms, mergedmesh)
    CutCell.assemble_interface_condition!(sysmatrix, interfacecondition, mergedmesh)

    matrix = CutCell.make_sparse(sysmatrix, mergedmesh)
    rhs = CutCell.rhs(sysrhs, mergedmesh)

    CutCell.apply_dirichlet_bc!(matrix, rhs, [10, 11, 12], 1, 0.0, 2)
    CutCell.apply_dirichlet_bc!(matrix, rhs, [10], 2, 0.0, 2)
    CutCell.apply_dirichlet_bc!(matrix, rhs, [7, 8, 9], 1, dx, 2)

    sol = matrix \ rhs
    disp = reshape(sol, 2, :)

    testdisp = [dx/2  dx/2  dx/2  3dx/4  3dx/4  3dx/4  dx  dx    dx  0.  0.    0.   dx/4  dx/4  dx/4  dx/2  dx/2  dx/2
                0.    dy/2  dy    0.     dy/2   dy     0.  dy/2  dy  0.  dy/2  dy   0.    dy/2  dy    0.    dy/2  dy]
    @test allapprox(disp,testdisp,1e2eps())
end

function test_four_cell_vertical_tension()
    polyorder = 1
    numqp = 3

    basis = TensorProductBasis(2, polyorder)
    levelset = InterpolatingPolynomial(1, basis)
    nf = CutCell.number_of_basis_functions(basis)
    mesh = CutCell.Mesh([0.0, 0.0], [2.0, 2.0], [2, 2], nf)

    normal = [0.0, 1.0]
    x0 = [0.0, 1.1]
    levelsetcoeffs =
        CutCell.levelset_coefficients(x -> plane_distance_function(x, normal, x0), mesh)

    cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
    cellquads = CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
    interfacequads = CutCell.InterfaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)

    mergemapper = CutCell.MergeMapper()
    mergecutmesh = CutCell.MergeCutMesh(cutmesh)

    CutCell.merge_cells_in_mesh!(mergecutmesh,cellquads,interfacequads,mergemapper)
    mergedmesh = CutCell.MergedMesh(mergecutmesh)

    lambda, mu = (1.0, 2.0)
    penalty = 1.0
    dy = 0.1
    e22 = dy / 2.0
    e11 = -lambda / (lambda + 2mu) * e22
    dx = e11*2.0

    stiffness = CutCell.HookeStiffness(lambda, mu, lambda, mu)

    bilinearforms = CutCell.BilinearForms(basis, cellquads, stiffness, cutmesh)
    interfacecondition =
        CutCell.InterfaceCondition(basis, interfacequads, stiffness, cutmesh, penalty)

    sysmatrix = CutCell.SystemMatrix()
    sysrhs = CutCell.SystemRHS()

    CutCell.assemble_bilinear_form!(sysmatrix, bilinearforms, mergedmesh)
    CutCell.assemble_interface_condition!(sysmatrix, interfacecondition, mergedmesh)

    matrix = CutCell.make_sparse(sysmatrix, mergedmesh)
    rhs = CutCell.rhs(sysrhs, mergedmesh)

    CutCell.apply_dirichlet_bc!(matrix, rhs, [7,9,11],2,0.0,2)
    CutCell.apply_dirichlet_bc!(matrix, rhs, [7],1,0.0,2)

    CutCell.apply_dirichlet_bc!(matrix, rhs, [2,4,6],2,dy,2)

    sol = matrix \ rhs
    disp = reshape(sol, 2, :)

    testdisp = [0.   0.  dx/2  dx/2  dx    dx  0.  0.   dx/2  dx/2  dx  dx
                dy/2 dy  dy/2  dy    dy/2  dy  0.  dy/2 0.    dy/2  0.  dy/2]
    @test allapprox(disp,testdisp,1e2eps())
end


test_linear_merge_cell_assembly()
test_curved_merge_assembly()
test_four_cell_vertical_tension()
