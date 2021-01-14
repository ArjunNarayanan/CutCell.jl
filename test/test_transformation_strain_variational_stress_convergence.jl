using Test
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCell
include("useful_routines.jl")
include("cylindrical_particle_exact_solution.jl")

function onboundary(x, L, W)
    return x[2] ≈ 0.0 || x[1] ≈ L || x[2] ≈ W || x[1] ≈ 0.0
end

function solve_and_compute_variational_stress_error(
    width,
    center,
    inradius,
    outradius,
    stiffness,
    theta0,
    nelmts,
    polyorder,
    numqp,
    penaltyfactor,
)

    lambda1, mu1 = CutCell.lame_coefficients(stiffness, +1)
    lambda2, mu2 = CutCell.lame_coefficients(stiffness, -1)
    transfstress =
        CutCell.plane_strain_transformation_stress(lambda1, mu1, theta0)

    analyticalsolution = AnalyticalSolution(
        inradius,
        outradius,
        center,
        lambda1,
        mu1,
        lambda2,
        mu2,
        theta0,
    )

    dx = width / nelmts
    meanmoduli = 0.5 * (lambda1 + lambda2 + mu1 + mu2)
    penalty = penaltyfactor / dx * meanmoduli


    basis = TensorProductBasis(2, polyorder)
    mesh = CutCell.Mesh([0.0, 0.0], [width, width], [nelmts, nelmts], basis)
    levelset = InterpolatingPolynomial(1, basis)
    levelsetcoeffs = CutCell.levelset_coefficients(
        x -> -circle_distance_function(x, center, inradius),
        mesh,
    )

    cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
    cellquads =
        CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
    interfacequads =
        CutCell.InterfaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
    facequads =
        CutCell.FaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)

    bilinearforms = CutCell.BilinearForms(basis, cellquads, stiffness, cutmesh)
    interfacecondition = CutCell.coherent_interface_condition(
        basis,
        interfacequads,
        stiffness,
        cutmesh,
        penalty,
    )

    displacementbc = CutCell.DisplacementCondition(
        analyticalsolution,
        basis,
        facequads,
        stiffness,
        cutmesh,
        x -> onboundary(x, width, width),
        penalty,
    )


    sysmatrix = CutCell.SystemMatrix()
    sysrhs = CutCell.SystemRHS()

    CutCell.assemble_bilinear_form!(sysmatrix, bilinearforms, cutmesh)
    CutCell.assemble_interface_condition!(
        sysmatrix,
        interfacecondition,
        cutmesh,
    )
    CutCell.assemble_bulk_transformation_linear_form!(
        sysrhs,
        transfstress,
        basis,
        cellquads,
        cutmesh,
    )
    CutCell.assemble_coherent_interface_transformation_rhs!(
        sysrhs,
        transfstress,
        basis,
        interfacequads,
        cutmesh,
    )
    CutCell.assemble_penalty_displacement_bc!(
        sysmatrix,
        sysrhs,
        displacementbc,
        cutmesh,
    )
    CutCell.assemble_penalty_displacement_transformation_rhs!(
        sysrhs,
        transfstress,
        basis,
        facequads,
        cutmesh,
        x -> onboundary(x, width, width),
    )


    matrix = CutCell.make_sparse(sysmatrix, cutmesh)
    rhs = CutCell.rhs(sysrhs, cutmesh)

    nodaldisplacement = matrix \ rhs

    sysmatrix = CutCell.SystemMatrix()
    sysrhs = CutCell.SystemRHS()

    CutCell.assemble_stress_mass_matrix!(sysmatrix, basis, cellquads, cutmesh)
    CutCell.assemble_stress_linear_form!(
        sysrhs,
        basis,
        cellquads,
        stiffness,
        nodaldisplacement,
        cutmesh,
    )
    CutCell.assemble_transformation_stress_linear_form!(
        sysrhs,
        transfstress,
        basis,
        cellquads,
        cutmesh,
    )

    matrix = CutCell.make_sparse_stress_operator(sysmatrix, cutmesh)
    rhs = CutCell.stress_rhs(sysrhs, cutmesh)

    stressvec = matrix \ rhs
    stress = reshape(stressvec, 3, :)

    err = mesh_L2_error(
        stress,
        x -> exact_stress(analyticalsolution, x)[1:3],
        basis,
        cellquads,
        cutmesh,
    )

    return err
end


K1, K2 = 247.0, 192.0
mu1, mu2 = 126.0, 87.0
lambda1 = lame_lambda(K1, mu1)
lambda2 = lame_lambda(K2, mu2)

stiffness = CutCell.HookeStiffness(lambda1, mu1, lambda2, mu2)

theta0 = -0.067

width = 1.0

center = [width / 2, width / 2]
inradius = width / 4
outradius = width

polyorder = 2
numqp = required_quadrature_order(polyorder) + 2
penaltyfactor = 1e2

powers = [3,4,5]
nelmts = [2^p + 1 for p in powers]

err = [solve_and_compute_variational_stress_error(
    width,
    center,
    inradius,
    outradius,
    stiffness,
    theta0,
    ne,
    polyorder,
    numqp,
    penaltyfactor,
) for ne in nelmts]

dx = 1.0 ./ nelmts
serr = [[er[i] for er in err] for i = 1:3]
rates = [convergence_rate(v,dx) for v in serr]
