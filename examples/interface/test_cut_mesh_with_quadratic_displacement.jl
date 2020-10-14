using Test
using PolynomialBasis
using ImplicitDomainQuadrature
include("plot_utils.jl")
using Revise
using CutCell

function circle_distance_function(coords, center, radius)
    diff2 = (coords .- center) .^ 2
    return sqrt.(mapslices(sum, diff2, dims = 1)') .- radius
end

function displacement_field(x)
    u1 = x[1]^2 + 2x[1] * x[2]
    u2 = x[2]^2 + 3x[1]
    return [u1, u2]
end

function body_force(lambda, mu)
    b1 = -2 * (lambda + 2mu)
    b2 = -(4lambda + 6mu)
    return [b1, b2]
end

function add_cell_error_squared!(err, interpolater, exactsolution, cellmap, quad)
    detjac = CutCell.determinant_jacobian(cellmap)
    for (p, w) in quad
        numsol = interpolater(p)
        exsol = exactsolution(cellmap(p))
        err .+= (numsol - exsol) .^ 2 * detjac * w
    end
end

function mesh_L2_error(nodalsolutions, exactsolution, basis, cellquads, cutmesh)
    err = zeros(2)
    interpolater = InterpolatingPolynomial(2, basis)
    ncells = CutCell.number_of_cells(cutmesh)
    for cellid = 1:ncells
        s = CutCell.cell_sign(cutmesh, cellid)
        cellmap = CutCell.cell_map(cutmesh,cellid)
        @assert s == -1 || s == 0 || s == 1
        if s == 1 || s == 0
            nodeids = CutCell.nodal_connectivity(cutmesh, 1, cellid)
            elementsolution = nodalsolutions[:, nodeids]
            update!(interpolater, elementsolution)
            quad = cellquads[1, cellid]
            add_cell_error_squared!(err, interpolater, exactsolution, cellmap, quad)
        end
        if s == -1 || s == 0
            nodeids = CutCell.nodal_connectivity(cutmesh, -1, cellid)
            elementsolution = nodalsolutions[:, nodeids]
            update!(interpolater, elementsolution)
            quad = cellquads[-1, cellid]
            add_cell_error_squared!(err, interpolater, exactsolution, cellmap, quad)
        end
    end
    return sqrt.(err)
end

lambda = 1.0
mu = 2.0
stiffness = CutCell.HookeStiffness(lambda, mu, lambda, mu)

polyorder = 2
numqp = 6
numcutqp = 6
interfacepenalty = 10.
basis = TensorProductBasis(2, polyorder)
levelset = InterpolatingPolynomial(1, basis)
nf = CutCell.number_of_basis_functions(basis)

mesh = CutCell.Mesh([0.0, 0.0], [1.0, 1.0], [5, 5], nf)

center = [0.5, 0.5]
radius = 0.25
levelsetcoeffs =
    CutCell.levelset_coefficients(x -> circle_distance_function(x, center, radius), mesh)

cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
cellquads = CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp, numcutqp)
interfacequads = CutCell.InterfaceQuadratures(levelset, levelsetcoeffs, cutmesh, numcutqp)

bilinearforms = CutCell.BilinearForms(basis, cellquads, stiffness, cutmesh)
interfacecondition =
    CutCell.InterfaceCondition(basis, interfacequads, stiffness, cutmesh, interfacepenalty)

sysmatrix = CutCell.SystemMatrix()
sysrhs = CutCell.SystemRHS()

CutCell.assemble_bilinear_form!(sysmatrix, bilinearforms, cutmesh)
CutCell.assemble_interface_condition!(sysmatrix, interfacecondition, cutmesh)
CutCell.assemble_cut_mesh_body_force_linear_form!(
    sysrhs,
    x -> body_force(lambda, mu),
    basis,
    cellquads,
    cutmesh,
)

matrix = CutCell.make_sparse(sysmatrix, cutmesh)
rhs = CutCell.rhs(sysrhs, cutmesh)

nodalcoordinates = CutCell.nodal_coordinates(cutmesh.mesh)
boundarynodeids = unique!(sort!(CutCell.boundary_node_ids(cutmesh.mesh)))
boundarynodecoords = nodalcoordinates[:, boundarynodeids]
boundarydisplacement = mapslices(displacement_field, boundarynodecoords, dims = 1)
CutCell.apply_dirichlet_bc!(matrix, rhs, boundarynodeids, boundarydisplacement)

sol = matrix \ rhs
cutdisp = reshape(sol, 2, :)

err = mesh_L2_error(cutdisp, displacement_field, basis, cellquads, cutmesh)

println(err)

# plot_interface_quadrature_points(interfacequads,cutmesh)
# plot_cell_quadrature_points(cellquads,cutmesh,-1,(0,1),(0,1))
