using Triangulate
using WriteVTK
using CSV, DataFrames
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("../../../test/useful_routines.jl")
include("../compute-stress.jl")

function bulk_modulus(l, m)
    return l + 2m / 3
end

function onleftboundary(x, L, W)
    return x[1] ≈ 0.0
end

function onbottomboundary(x, L, W)
    return x[2] ≈ 0.0
end

function onrightboundary(x, L, W)
    return x[1] ≈ L
end

function ontopboundary(x, L, W)
    return x[2] ≈ W
end

function onboundary(x, L, W)
    return onleftboundary(x, L, W) ||
           onbottomboundary(x, L, W) ||
           onrightboundary(x, L, W) ||
           ontopboundary(x, L, W)
end

function solve_and_compute_stress(
    width,
    corner,
    stiffness,
    theta0,
    nelmts,
    polyorder,
    numqp,
    penaltyfactor,
)

    lambda1, mu1 = CutCell.lame_coefficients(stiffness, +1)
    lambda2, mu2 = CutCell.lame_coefficients(stiffness, -1)
    transfstress = CutCell.plane_strain_transformation_stress(lambda1, mu1, theta0)

    dx = width / nelmts
    meanmoduli = 0.5 * (lambda1 + lambda2 + mu1 + mu2)
    penalty = penaltyfactor / dx * meanmoduli

    basis = TensorProductBasis(2, polyorder)
    mesh = CutCell.Mesh([0.0, 0.0], [width, width], [nelmts, nelmts], basis)
    levelset = InterpolatingPolynomial(1, basis)
    levelsetcoeffs = CutCell.levelset_coefficients(
        x -> -corner_distance_function(x, corner),
        mesh,
    )

    cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
    cellquads = CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
    interfacequads = CutCell.InterfaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
    facequads = CutCell.FaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)

    bilinearforms = CutCell.BilinearForms(basis, cellquads, stiffness, cutmesh)
    interfacecondition =
        CutCell.InterfaceCondition(basis, interfacequads, stiffness, cutmesh, penalty)

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
    CutCell.assemble_penalty_displacement_bc!(sysmatrix, sysrhs, displacementbc, cutmesh)
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

    displacement = matrix \ rhs


end
