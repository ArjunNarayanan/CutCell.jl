using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("useful_routines.jl")


function istractionboundary(x)
    if x[1] ≈ 1.0
        return true
    else
        return false
    end
end

function tractionfunc(x)
    return [x[1], 0.0]
end

lambda = 1.0
mu = 2.0
stiffness = CutCell.plane_strain_voigt_hooke_matrix(lambda, mu)
polyorder = 1
numqp = 2
basis = TensorProductBasis(2, polyorder)
nf = CutCell.number_of_basis_functions(basis)
vquad = tensor_product_quadrature(2, numqp)
facequads = CutCell.face_quadratures(numqp)

mesh = CutCell.Mesh([0.0, 0.0], [1.0, 1.0], [1, 1], nf)
cellmaps = CutCell.cell_maps(mesh)
nodalconnectivity = CutCell.nodal_connectivity(mesh)
cellconnectivity = CutCell.cell_connectivity(mesh)
cellmatrix = CutCell.bilinear_form(basis, vquad, stiffness, cellmaps[1])

sysmatrix = CutCell.SystemMatrix()
sysrhs = CutCell.SystemRHS()

CutCell.assemble_bilinear_form!(sysmatrix, cellmatrix, nodalconnectivity, 2)
CutCell.assemble_traction_force_linear_form!(
    sysrhs,
    tractionfunc,
    basis,
    facequads,
    cellmaps,
    nodalconnectivity,
    cellconnectivity,
    istractionboundary,
)

globalndofs = CutCell.number_of_nodes(mesh)*2
matrix = CutCell.sparse(sysmatrix,globalndofs)
rhs = CutCell.rhs(sysrhs,globalndofs)

bottomnodeids = CutCell.bottom_boundary_node_ids(mesh)
leftnodeids = CutCell.left_boundary_node_ids(mesh)

CutCell.apply_dirichlet_bc!(matrix,rhs,bottomnodeids,2,0.0,2)
CutCell.apply_dirichlet_bc!(matrix,rhs,leftnodeids,1,0.0,2)

sol = matrix\rhs
disp = reshape(sol,2,:)

m = (lambda+2mu)
e11 = 1/(m - lambda^2/m)
e22 = - lambda/m*e11
u1 = e11
u2 = e22

testdisp = [0. 0. u1  u1
            0. u2 0.  u2]
@test allapprox(disp,testdisp)





#########################################################
# Test 4 element quadratic basis uniform tension
lambda = 1.0
mu = 2.0
stiffness = CutCell.plane_strain_voigt_hooke_matrix(lambda, mu)
polyorder = 2
numqp = 3
basis = TensorProductBasis(2, polyorder)
nf = CutCell.number_of_basis_functions(basis)
vquad = tensor_product_quadrature(2, numqp)
facequads = CutCell.face_quadratures(numqp)

mesh = CutCell.Mesh([0.0, 0.0], [1.0, 1.0], [2, 2], nf)
cellmaps = CutCell.cell_maps(mesh)
nodalconnectivity = CutCell.nodal_connectivity(mesh)
cellconnectivity = CutCell.cell_connectivity(mesh)
cellmatrix = CutCell.bilinear_form(basis, vquad, stiffness, cellmaps[1])

sysmatrix = CutCell.SystemMatrix()
sysrhs = CutCell.SystemRHS()

CutCell.assemble_bilinear_form!(sysmatrix, cellmatrix, nodalconnectivity, 2)
CutCell.assemble_traction_force_linear_form!(
    sysrhs,
    tractionfunc,
    basis,
    facequads,
    cellmaps,
    nodalconnectivity,
    cellconnectivity,
    istractionboundary,
)

globalndofs = CutCell.number_of_nodes(mesh)*2
matrix = CutCell.sparse(sysmatrix,globalndofs)
rhs = CutCell.rhs(sysrhs,globalndofs)

bottomnodeids = CutCell.bottom_boundary_node_ids(mesh)
leftnodeids = CutCell.left_boundary_node_ids(mesh)

CutCell.apply_dirichlet_bc!(matrix,rhs,bottomnodeids,2,0.0,2)
CutCell.apply_dirichlet_bc!(matrix,rhs,leftnodeids,1,0.0,2)

sol = matrix\rhs
disp = reshape(sol,2,:)

nodalcoordinates = CutCell.nodal_coordinates(mesh)

function displacement_of_mesh(nodalcoordinates,u1,u2)
    disp = copy(nodalcoordinates)
    disp[1,:] .*= u1
    disp[2,:] .*= u2
    return disp
end

testdisp = displacement_of_mesh(nodalcoordinates,u1,u2)

@test allapprox(testdisp,disp)
