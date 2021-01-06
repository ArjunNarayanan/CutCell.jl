using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("../../../test/useful_routines.jl")

function bulk_modulus(l, m)
    return l + 2m / 3
end

function lame_lambda(k, m)
    return k - 2m / 3
end

function solve_for_displacement(
    basis,
    cellquads,
    interfacequads,
    facequads,
    cutmesh,
    levelset,
    levelsetcoeffs,
    stiffness,
    theta0,
    penalty,
)

    bilinearforms = CutCell.BilinearForms(basis, cellquads, stiffness, cutmesh)
    interfacecondition = CutCell.incoherent_interface_condition(
        basis,
        interfacequads,
        stiffness,
        cutmesh,
        penalty,
    )

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
    CutCell.assemble_incoherent_interface_transformation_rhs!(
        sysrhs,
        transfstress,
        basis,
        interfacequads,
        cutmesh,
    )

    matrix = CutCell.make_sparse(sysmatrix, cutmesh)
    rhs = CutCell.rhs(sysrhs, cutmesh)

    topleftnodeid = CutCell.nodes_per_mesh_side(cutmesh.mesh)[2]
    CutCell.apply_dirichlet_bc!(matrix, rhs, [1, topleftnodeid], 1, 0.0, 2)
    CutCell.apply_dirichlet_bc!(matrix, rhs, [1], 2, 0.0, 2)

    nodaldisplacement = matrix \ rhs

    return nodaldisplacement
end

K1, K2 = 247.0, 192.0
mu1, mu2 = 126.0, 87.0
lambda1 = lame_lambda(K1, mu1)
lambda2 = lame_lambda(K2, mu2)
stiffness = CutCell.HookeStiffness(lambda1, mu1, lambda2, mu2)

V1 = 1.0 / 3.68e-6
V2 = 1.0 / 3.93e-6
theta0 = -0.067
ΔG = -6.95

width = 1.0
penaltyfactor = 1e2

polyorder = 2
numqp = required_quadrature_order(polyorder) + 2
nelmts = 11
center = [width / 2, width / 2]
inradius = width / 4
outradius = width

transfstrain = CutCell.plane_transformation_strain(theta0)
transfstress = CutCell.plane_strain_transformation_stress(lambda1, mu1, theta0)

dx = width / nelmts
meanmoduli = 0.5 * (lambda1 + lambda2 + mu1 + mu2)
penalty = penaltyfactor / dx * meanmoduli


basis = TensorProductBasis(2, polyorder)
mesh = CutCell.Mesh([0.0, 0.0], [width, width], [nelmts, nelmts], basis)
levelset = InterpolatingPolynomial(1, basis)
levelsetcoeffs =
    CutCell.levelset_coefficients(x -> -circle_distance_function(x, center, inradius), mesh)

cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
cellquads = CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
interfacequads = CutCell.InterfaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
facequads = CutCell.FaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)

nodaldisplacement = solve_for_displacement(
    basis,
    cellquads,
    interfacequads,
    facequads,
    cutmesh,
    levelset,
    levelsetcoeffs,
    stiffness,
    theta0,
    penalty,
)

refseedpoints, spatialseedpoints, seedcellids =
    CutCell.seed_zero_levelset_with_interfacequads(interfacequads, cutmesh)
nodalcoordinates = CutCell.nodal_coordinates(cutmesh)

tol = 1e-8
boundingradius = 3.0
refclosestpoints, refclosestcellids, refgradients =
    CutCell.closest_reference_points_on_zero_levelset(
        nodalcoordinates,
        refseedpoints,
        spatialseedpoints,
        seedcellids,
        levelset,
        levelsetcoeffs,
        mesh,
        tol,
        boundingradius,
    )


product_stress_at_cp = CutCell.product_stress_at_reference_points(
    refclosestpoints,
    refclosestcellids,
    basis,
    stiffness,
    transfstress,
    theta0,
    nodaldisplacement,
    cutmesh,
)



# using PyPlot
# fig, ax = PyPlot.subplots()
# ax.tricontour(nodalcoordinates[1, :], nodalcoordinates[2, :], newlevelsetcoeffs, [0.0])
# ax.set_aspect("equal")
# fig
