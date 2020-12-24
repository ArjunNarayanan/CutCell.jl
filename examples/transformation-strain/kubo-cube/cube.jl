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
    levelsetcoeffs =
        CutCell.levelset_coefficients(x -> -corner_distance_function(x, corner), mesh)

    cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
    cellquads = CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
    interfacequads = CutCell.InterfaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
    facequads = CutCell.FaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)

    bilinearforms = CutCell.BilinearForms(basis, cellquads, stiffness, cutmesh)
    interfacecondition =
        CutCell.InterfaceCondition(basis, interfacequads, stiffness, cutmesh, penalty)

    leftdisplacementbc = CutCell.DisplacementComponentCondition(
        x -> 0.0,
        basis,
        facequads,
        stiffness,
        cutmesh,
        x -> onleftboundary(x, width, width),
        [1.0, 0.0],
        penalty,
    )
    bottomdisplacementbc = CutCell.DisplacementComponentCondition(
        x -> 0.0,
        basis,
        facequads,
        stiffness,
        cutmesh,
        x -> onbottomboundary(x, width, width),
        [0.0, 1.0],
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
    CutCell.assemble_penalty_displacement_bc!(
        sysmatrix,
        sysrhs,
        leftdisplacementbc,
        cutmesh,
    )
    CutCell.assemble_penalty_displacement_bc!(
        sysmatrix,
        sysrhs,
        bottomdisplacementbc,
        cutmesh,
    )
    CutCell.assemble_penalty_displacement_component_transformation_rhs!(
        sysrhs,
        transfstress,
        basis,
        facequads,
        cutmesh,
        x -> onleftboundary(x, width, width),
        [1.0, 0.0],
    )
    CutCell.assemble_penalty_displacement_component_transformation_rhs!(
        sysrhs,
        transfstress,
        basis,
        facequads,
        cutmesh,
        x -> onbottomboundary(x, width, width),
        [0.0, 1.0],
    )

    matrix = CutCell.make_sparse(sysmatrix, cutmesh)
    rhs = CutCell.rhs(sysrhs, cutmesh)

    displacement = matrix \ rhs

    qpstress = compute_stress_at_quadrature_points(
        displacement,
        basis,
        stiffness,
        transfstress,
        theta0,
        cellquads,
        cutmesh,
    )
    qpcoords = compute_quadrature_points(cellquads, cutmesh)

    return qpstress, qpcoords
end


K1, K2 = 247.0, 192.0
mu1, mu2 = 126.0, 87.0
lambda1 = lame_lambda(K1, mu1)
lambda2 = lame_lambda(K2, mu2)

theta0 = -0.067
stiffness = CutCell.HookeStiffness(lambda1, mu1, lambda2, mu2)

width = 1.0
corner = [0.8, 0.8]
penaltyfactor = 1e2

nelmts = 37
polyorder = 2
numqp = required_quadrature_order(polyorder) + 2

qpstress, qpcoords = solve_and_compute_stress(
    width,
    corner,
    stiffness,
    theta0,
    nelmts,
    polyorder,
    numqp,
    penaltyfactor,
)

pressure = -(qpstress[1, :] + qpstress[2, :] + qpstress[4, :]) / 3
triin = Triangulate.TriangulateIO()
triin.pointlist = qpcoords
(triout, vorout) = triangulate("", triin)
connectivity = triout.trianglelist
cells = [
    MeshCell(VTKCellTypes.VTK_TRIANGLE, connectivity[:, i]) for i = 1:size(connectivity)[2]
]
filename = "stress-poly-"*string(polyorder)*"-nelmts-" * string(nelmts)
vtkfile = vtk_grid(
    "examples/transformation-strain/kubo-cube/" * filename,
    qpcoords[1, :],
    qpcoords[2, :],
    cells,
)
vtkfile["pressure"] = pressure
vtkfile["s11"] = qpstress[1,:]
vtkfile["s22"] = qpstress[2,:]
vtkfile["s12"] = qpstress[3,:]
vtkfile["s33"] = qpstress[4,:]
outfiles = vtk_save(vtkfile)
