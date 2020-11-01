using LinearAlgebra
using PyPlot
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("../../test/useful_routines.jl")

function compute_at_cell_quadrature_points!(vals,interpolater,quad)
    for (p,w) in quad
        append!(vals,interpolater(p))
    end
end

function compute_field_at_quadrature_points(nodalvals,basis,cellquads,cutmesh)
    ndofs,numnodes = size(nodalvals)
    ncells = CutCell.number_of_cells(cutmesh)
    interpolater = InterpolatingPolynomial(ndofs,basis)
    quadvals = zeros(0)

    for cellid in 1:ncells
        s = CutCell.cell_sign(cutmesh,cellid)
        @assert s == -1 || s == 0 || s == +1
        if s == +1 || s == 0
            nodeids = CutCell.nodal_connectivity(cutmesh,+1,cellid)
            cellvals = nodalvals[:,nodeids]
            update!(interpolater,cellvals)
            quad = cellquads[+1,cellid]
            compute_at_cell_quadrature_points!(quadvals,interpolater,quad)
        end
        if s == -1 || s == 0
            nodeids = CutCell.nodal_connectivity(cutmesh,-1,cellid)
            cellvals = nodalvals[:,nodeids]
            update!(interpolater,cellvals)
            quad = cellquads[-1,cellid]
            compute_at_cell_quadrature_points!(quadvals,interpolater,quad)
        end
    end
    return reshape(quadvals,ndofs,:)
end

function compute_quadrature_points(cellquads,cutmesh)
    ncells = CutCell.number_of_cells(cutmesh)
    coords = zeros(2,0)
    for cellid in 1:ncells
        s = CutCell.cell_sign(cutmesh,cellid)
        cellmap = CutCell.cell_map(cutmesh,cellid)
        @assert s == -1 || s == 0 || s == +1
        if s == +1 || s == 0
            coords = hcat(coords,cellmap(cellquads[+1,cellid].points))
        end
        if s == -1 || s == 0
            coords = hcat(coords,cellmap(cellquads[-1,cellid].points))
        end
    end
    return coords
end

lambda1, mu1 = 100.0, 80.0
lambda2, mu2 = 100.0, 80.0
theta0 = -0.067
stiffness = CutCell.HookeStiffness(lambda1, mu1, lambda2, mu2)
transfstress = CutCell.plane_strain_transformation_stress(lambda1, mu1, theta0)

L = 1.0
penaltyfactor = 1e2
nelmts = 21
dx = L / nelmts
penalty = penaltyfactor / dx * (lambda1 + mu1)

polyorder = 2
numqp = required_quadrature_order(polyorder) + 2

center = [L/2, L/2]
radius = L/4
basis = TensorProductBasis(2, polyorder)
mesh = CutCell.Mesh([0.0, 0.0], [L, L], [nelmts, nelmts], basis)
levelset = InterpolatingPolynomial(1, basis)
levelsetcoeffs =
    CutCell.levelset_coefficients(x -> -circle_distance_function(x, center, radius), mesh)

cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
cellquads = CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
interfacequads = CutCell.InterfaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
facequads = CutCell.FaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)

bilinearforms = CutCell.BilinearForms(basis, cellquads, stiffness, cutmesh)
interfacecondition =
    CutCell.InterfaceCondition(basis, interfacequads, stiffness, cutmesh, penalty)


sysmatrix = CutCell.SystemMatrix()
sysrhs = CutCell.SystemRHS()

CutCell.assemble_bilinear_form!(sysmatrix, bilinearforms, cutmesh)
CutCell.assemble_interface_condition!(sysmatrix, interfacecondition, cutmesh)
CutCell.assemble_bulk_transformation_linear_form!(
    sysrhs,
    transfstress,
    basis,
    cellquads,
    cutmesh,
)
CutCell.assemble_interface_transformation_rhs!(
    sysrhs,
    transfstress,
    basis,
    interfacequads,
    cutmesh,
)

matrix = CutCell.make_sparse(sysmatrix, cutmesh)
rhs = CutCell.rhs(sysrhs, cutmesh)

topleftnodeid = CutCell.nodes_per_mesh_side(mesh)[2]
CutCell.apply_dirichlet_bc!(matrix,rhs,[1,topleftnodeid],1,0.0,2)
CutCell.apply_dirichlet_bc!(matrix,rhs,[1],2,0.0,2)

sol = matrix\rhs
disp = reshape(sol,2,:)

sysmatrix = CutCell.SystemMatrix()
sysrhs = CutCell.SystemRHS()

CutCell.assemble_stress_mass_matrix!(sysmatrix,basis,cellquads,cutmesh)
CutCell.assemble_stress_linear_form!(sysrhs,basis,cellquads,stiffness,sol,cutmesh)
CutCell.assemble_transformation_stress_linear_form!(sysrhs,transfstress,basis,cellquads,cutmesh)

matrix = CutCell.make_sparse_stress_operator(sysmatrix,cutmesh)
rhs = CutCell.stress_rhs(sysrhs,cutmesh)

stressvec = matrix\rhs
stress = reshape(stressvec,3,:)

quadvals = compute_field_at_quadrature_points(stress,basis,cellquads,cutmesh)
pressure = -(quadvals[1,:] + quadvals[2,:])
quadcoords = compute_quadrature_points(cellquads,cutmesh)

fig,ax = PyPlot.subplots()
cbar = ax.tricontourf(quadcoords[1,:],quadcoords[2,:],pressure)
ax.set_aspect("equal")
fig.colorbar(cbar)
savefig("circle-inclusion.png")
