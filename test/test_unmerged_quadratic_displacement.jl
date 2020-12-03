using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("useful_routines.jl")

function displacement(x)
    u1 = x[1]^2 + 2x[1] * x[2]
    u2 = x[2]^2 + 3x[1]
    return [u1, u2]
end

function body_force(lambda, mu, x)
    b1 = -2 * (lambda + 2mu)
    b2 = -(4lambda + 6mu)
    return [b1, b2]
end

function onboundary(x, L, W)
    return x[2] ≈ 0.0 || x[1] ≈ L || x[2] ≈ W || x[1] ≈ 0.0
end

function linear_system(basis, quad, stiffness, femesh, bodyforcefunc)

    cellmaps = CutCell.cell_maps(femesh)
    nodalconnectivity = CutCell.nodal_connectivity(femesh)

    sysmatrix = CutCell.SystemMatrix()
    sysrhs = CutCell.SystemRHS()

    cellmatrix = CutCell.bilinear_form(basis, quad, stiffness, cellmaps[1])
    CutCell.assemble_bilinear_form!(sysmatrix, cellmatrix, nodalconnectivity, 2)
    CutCell.assemble_body_force_linear_form!(
        sysrhs,
        bodyforcefunc,
        basis,
        quad,
        cellmaps,
        nodalconnectivity,
    )

    K = CutCell.make_sparse(sysmatrix, femesh)
    R = CutCell.rhs(sysrhs, femesh)

    return K, R
end

function apply_displacement_boundary_condition!(matrix, rhs, displacement_function, femesh)

    boundarynodeids = CutCell.boundary_node_ids(femesh)
    nodalcoordinates = CutCell.nodal_coordinates(femesh)
    boundarynodecoordinates = nodalcoordinates[:, boundarynodeids]
    boundarydisplacement =
        mapslices(displacement_function, boundarynodecoordinates, dims = 1)
    CutCell.apply_dirichlet_bc!(matrix, rhs, boundarynodeids, boundarydisplacement)
end


L = 1.0
W = 1.0
lambda, mu = 1.0, 2.0
stiffness = CutCell.HookeStiffness(lambda,mu,lambda,mu)
polyorder = 2
numqp = required_quadrature_order(polyorder) + 2
penalty = 1e2
nelmts = 2

dx = 1.0/nelmts

# xc = [1.0,0.5]
# radius = xc[1] - 0.5 - 0.1dx
x0 = [0.1,0.0]
normal = [1.,-1.]/sqrt(2)



basis = TensorProductBasis(2, polyorder)
quad = tensor_product_quadrature(2, numqp)
mesh = CutCell.Mesh([0.0, 0.0], [L, W], [nelmts, nelmts], basis)
levelset = InterpolatingPolynomial(1, basis)


# levelsetcoeffs =
#     CutCell.levelset_coefficients(x -> circle_distance_function(x,xc,radius), mesh)
levelsetcoeffs =
    CutCell.levelset_coefficients(x -> plane_distance_function(x,normal,x0), mesh)




cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
cellquads = CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
interfacequads = CutCell.InterfaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
facequads = CutCell.FaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)


bilinearforms = CutCell.BilinearForms(basis, cellquads, stiffness, cutmesh)
interfacecondition =
    CutCell.InterfaceCondition(basis, interfacequads, stiffness, cutmesh, penalty)
displacementbc = CutCell.DisplacementCondition(
    displacement,
    basis,
    facequads,
    stiffness,
    cutmesh,
    x -> onboundary(x, L, W),
    penalty,
)

sysmatrix = CutCell.SystemMatrix()
sysrhs = CutCell.SystemRHS()

CutCell.assemble_bilinear_form!(sysmatrix, bilinearforms, cutmesh)
CutCell.assemble_interface_condition!(sysmatrix, interfacecondition, cutmesh)
CutCell.assemble_body_force_linear_form!(
    sysrhs,
    x -> body_force(lambda, mu, x),
    basis,
    cellquads,
    cutmesh,
)
CutCell.assemble_penalty_displacement_bc!(sysmatrix,sysrhs,displacementbc,cutmesh)


matrix = CutCell.make_sparse(sysmatrix, cutmesh)
rhs = CutCell.rhs(sysrhs, cutmesh)

sol = matrix \ rhs
disp = reshape(sol, 2, :)

err = mesh_L2_error(disp,displacement,basis,cellquads,cutmesh)

println("Unmerged Error = ", err)
