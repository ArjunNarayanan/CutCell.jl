using Test
using LinearAlgebra
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCell
include("useful_routines.jl")

function displacement(x)
    u1 = x[1]^2 + 2x[1]*x[2]
    u2 = x[2]^2 + 3x[1]
    return [u1, u2]
end

function body_force(lambda, mu, x)
    b1 = -2*(lambda + 2mu)
    b2 = -(4lambda + 6mu)
    return [b1, b2]
end

function onboundary(x, L, W)
    return x[2] ≈ 0.0 || x[1] ≈ L || x[2] ≈ W || x[1] ≈ 0.0
end

using Plots

function plot_cell_quadrature!(fig,cellquads,mergedmesh,s,cellid)
    cellmap = CutCell.cell_map(mergedmesh,s,cellid)
    quad = cellquads[s,cellid]
    p = cellmap(quad.points)
    scatter!(fig,p[1,:],p[2,:])
end


L = 1.0
W = 1.0
lambda, mu = 1.0, 2.0
penaltyfactor = 1e2
polyorder = 2
numqp = required_quadrature_order(polyorder)+2
nelmts = 2

dx = 1.0/nelmts
penalty = penaltyfactor/dx*(lambda+mu)
stiffness = CutCell.HookeStiffness(lambda, mu, lambda, mu)

x0 = [0.1,0.0]
normal = [1.,-1.]/sqrt(2)


basis = TensorProductBasis(2, polyorder)

# mesh = CutCell.Mesh([0.0, 0.0], [L, W], [nelmts, nelmts], basis)
# levelset = InterpolatingPolynomial(1, basis)
#
# levelsetcoeffs =
#     CutCell.levelset_coefficients(x -> plane_distance_function(x,normal,x0), mesh)
#
# cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
# cellquads = CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
# interfacequads = CutCell.InterfaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
# facequads = CutCell.FaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
#
#
# mergecutmesh = CutCell.MergeCutMesh(cutmesh)
# mergemapper = CutCell.MergeMapper()
#
# CutCell.merge_cells_in_mesh!(mergecutmesh,cellquads,interfacequads,facequads,mergemapper)
# mergedmesh = CutCell.MergedMesh(mergecutmesh)
#
# bilinearforms = CutCell.BilinearForms(basis, cellquads, stiffness, mergedmesh)
# interfacecondition =
#     CutCell.InterfaceCondition(basis, interfacequads, stiffness, mergedmesh, penalty)
# displacementbc = CutCell.DisplacementCondition(
#     x -> displacement(x),
#     basis,
#     facequads,
#     stiffness,
#     mergedmesh,
#     x -> onboundary(x, L, W),
#     penalty,
# )
#
# sysmatrix = CutCell.SystemMatrix()
# sysrhs = CutCell.SystemRHS()
#
# CutCell.assemble_bilinear_form!(sysmatrix, bilinearforms, mergedmesh)
# CutCell.assemble_interface_condition!(sysmatrix, interfacecondition, mergedmesh)
# CutCell.assemble_body_force_linear_form!(
#     sysrhs,
#     x -> body_force(lambda, mu, x),
#     basis,
#     cellquads,
#     mergedmesh,
# )
# CutCell.assemble_penalty_displacement_bc!(sysmatrix,sysrhs,displacementbc,mergedmesh)
#
# matrix = CutCell.make_sparse(sysmatrix, mergedmesh)
# rhs = CutCell.rhs(sysrhs, mergedmesh)
#
# sol = matrix \ rhs
# disp = reshape(sol, 2, :)
#
# err = mesh_L2_error(disp, x -> displacement(x), basis, cellquads, mergedmesh)
#
# println("Merged Error = ", err)
