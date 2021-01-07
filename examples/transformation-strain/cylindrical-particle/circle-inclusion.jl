using Triangulate
using WriteVTK
using PyPlot
using CSV, DataFrames
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("../../test/useful_routines.jl")

function compute_at_cell_quadrature_points!(vals, interpolater, quad)
    for (p, w) in quad
        append!(vals, interpolater(p))
    end
end

function compute_field_at_quadrature_points(nodalvals, basis, cellquads, cutmesh)
    ndofs, numnodes = size(nodalvals)
    ncells = CutCell.number_of_cells(cutmesh)
    interpolater = InterpolatingPolynomial(ndofs, basis)
    quadvals = zeros(0)

    for cellid = 1:ncells
        s = CutCell.cell_sign(cutmesh, cellid)
        @assert s == -1 || s == 0 || s == +1
        if s == +1 || s == 0
            nodeids = CutCell.nodal_connectivity(cutmesh, +1, cellid)
            cellvals = nodalvals[:, nodeids]
            update!(interpolater, cellvals)
            quad = cellquads[+1, cellid]
            compute_at_cell_quadrature_points!(quadvals, interpolater, quad)
        end
        if s == -1 || s == 0
            nodeids = CutCell.nodal_connectivity(cutmesh, -1, cellid)
            cellvals = nodalvals[:, nodeids]
            update!(interpolater, cellvals)
            quad = cellquads[-1, cellid]
            compute_at_cell_quadrature_points!(quadvals, interpolater, quad)
        end
    end
    return reshape(quadvals, ndofs, :)
end

function compute_quadrature_points(cellquads, cutmesh)
    ncells = CutCell.number_of_cells(cutmesh)
    coords = zeros(2, 0)
    for cellid = 1:ncells
        s = CutCell.cell_sign(cutmesh, cellid)
        cellmap = CutCell.cell_map(cutmesh, cellid)
        @assert s == -1 || s == 0 || s == +1
        if s == +1 || s == 0
            coords = hcat(coords, cellmap(cellquads[+1, cellid].points))
        end
        if s == -1 || s == 0
            coords = hcat(coords, cellmap(cellquads[-1, cellid].points))
        end
    end
    return coords
end

function bulk_modulus(l, m)
    return l + 2m / 3
end

function analytical_coefficient_matrix(inradius, outradius, ls, ms, lc, mc)
    a = zeros(3, 3)
    a[1, 1] = inradius
    a[1, 2] = -inradius
    a[1, 3] = -1.0 / inradius
    a[2, 1] = 2 * (lc + mc)
    a[2, 2] = -2 * (ls + ms)
    a[2, 3] = 2ms / inradius^2
    a[3, 2] = 2(ls + ms)
    a[3, 3] = -2ms / outradius^2
    return a
end

function analytical_coefficient_rhs(ls, ms, theta0)
    r = zeros(3)
    Ks = bulk_modulus(ls, ms)
    r[2] = -Ks * theta0
    r[3] = Ks * theta0
    return r
end

struct AnalyticalSolution
    inradius::Any
    outradius::Any
    center::Any
    A1c::Any
    A1s::Any
    A2s::Any
    ls::Any
    ms::Any
    lc::Any
    mc::Any
    theta0::Any
    function AnalyticalSolution(inradius, outradius, center, ls, ms, lc, mc, theta0)
        a = analytical_coefficient_matrix(inradius, outradius, ls, ms, lc, mc)
        r = analytical_coefficient_rhs(ls, ms, theta0)
        coeffs = a \ r
        new(
            inradius,
            outradius,
            center,
            coeffs[1],
            coeffs[2],
            coeffs[3],
            ls,
            ms,
            lc,
            mc,
            theta0,
        )
    end
end

function radial_displacement(A::AnalyticalSolution, r)
    if r <= A.inradius
        return A.A1c * r
    else
        return A.A1s * r + A.A2s / r
    end
end

function (A::AnalyticalSolution)(x)
    relpos = x - A.center
    r = sqrt(relpos' * relpos)
    ur = radial_displacement(A, r)
    if ur ≈ 0.0
        [0.0, 0.0]
    else
        costheta = (x[1] - A.center[1]) / r
        sintheta = (x[2] - A.center[2]) / r
        u1 = ur * costheta
        u2 = ur * sintheta
        return [u1, u2]
    end
end

function shell_radial_stress(ls, ms, theta0, A1, A2, r)
    return (ls + 2ms) * (A1 - A2 / r^2) + ls * (A1 + A2 / r^2) - (ls + 2ms / 3) * theta0
end

function shell_circumferential_stress(ls, ms, theta0, A1, A2, r)
    return ls * (A1 - A2 / r^2) + (ls + 2ms) * (A1 + A2 / r^2) - (ls + 2ms / 3) * theta0
end

function core_in_plane_stress(lc, mc, A1)
    return (lc + 2mc) * A1 + lc * A1
end

function stress_in_plane_trace(A::AnalyticalSolution, x)
    relpos = x - A.center
    r = sqrt(relpos' * relpos)
    if r <= A.inradius
        return 2 * core_in_plane_stress(A.lc, A.mc, A.A1c)
    else
        srr = shell_radial_stress(A.ls, A.ms, A.theta0, A.A1s, A.A2s, r)
        stt = shell_circumferential_stress(A.ls, A.ms, A.theta0, A.A1s, A.A2s, r)
        return srr + stt
    end
end

function onboundary(x, L, W)
    return x[2] ≈ 0.0 || x[1] ≈ L || x[2] ≈ W || x[1] ≈ 0.0
end

function displacement_error(
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
    transfstress = CutCell.plane_strain_transformation_stress(lambda1, mu1, theta0)

    dx = width / nelmts
    meanmoduli = 0.5 * (lambda1 + lambda2 + mu1 + mu2)
    penalty = penaltyfactor / dx * meanmoduli

    analyticalsolution =
        AnalyticalSolution(inradius, outradius, center, lambda1, mu1, lambda2, mu2, theta0)

    basis = TensorProductBasis(2, polyorder)
    mesh = CutCell.Mesh([0.0, 0.0], [width, width], [nelmts, nelmts], basis)
    levelset = InterpolatingPolynomial(1, basis)
    levelsetcoeffs = CutCell.levelset_coefficients(
        x -> -circle_distance_function(x, center, inradius),
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

    sol = matrix \ rhs
    disp = reshape(sol, 2, :)

    err = mesh_L2_error(disp, analyticalsolution, basis, cellquads, cutmesh)
    den = integral_norm_on_cut_mesh(analyticalsolution, cellquads, cutmesh, 2)

    return err ./ den
end

function stress_error(
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
    transfstress = CutCell.plane_strain_transformation_stress(lambda1, mu1, theta0)

    dx = width / nelmts
    meanmoduli = 0.5 * (lambda1 + lambda2 + mu1 + mu2)
    penalty = penaltyfactor / dx * meanmoduli

    analyticalsolution =
        AnalyticalSolution(inradius, outradius, center, lambda1, mu1, lambda2, mu2, theta0)

    basis = TensorProductBasis(2, polyorder)
    mesh = CutCell.Mesh([0.0, 0.0], [width, width], [nelmts, nelmts], basis)
    levelset = InterpolatingPolynomial(1, basis)
    levelsetcoeffs = CutCell.levelset_coefficients(
        x -> -circle_distance_function(x, center, inradius),
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

    sol = matrix \ rhs

    sysmatrix = CutCell.SystemMatrix()
    sysrhs = CutCell.SystemRHS()

    CutCell.assemble_stress_mass_matrix!(sysmatrix, basis, cellquads, cutmesh)
    CutCell.assemble_stress_linear_form!(sysrhs, basis, cellquads, stiffness, sol, cutmesh)
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

    inplanetrace = (stress[1, :] + stress[2, :])'

    err = mesh_L2_error(
        inplanetrace,
        x -> stress_in_plane_trace(analyticalsolution, x),
        basis,
        cellquads,
        cutmesh,
    )
    den = integral_norm_on_cut_mesh(
        x -> stress_in_plane_trace(analyticalsolution, x),
        cellquads,
        cutmesh,
        1,
    )
    return err[1]/den[1]
end

function mean(v)
    return sum(v) / length(v)
end

function convergence_rate(dx, err)
    return diff(log.(err)) ./ diff(log.(dx))
end

lambda1, mu1 = 100.0, 80.0
lambda2, mu2 = 80.0, 60.0
theta0 = -0.067
stiffness = CutCell.HookeStiffness(lambda1, mu1, lambda2, mu2)

width = 1.0
penaltyfactor = 1e2

polyorder = 3
numqp = required_quadrature_order(polyorder) + 2

center = [width / 2, width / 2]
inradius = width / 4
outradius = width

powers = 1:7
nelmts = [2^p + 1 for p in powers]


err = [
    stress_error(
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
    ) for ne in nelmts
]

dx = 1.0 ./ nelmts

rate = convergence_rate(dx, err)

# err = [
#     displacement_error(
#         width,
#         center,
#         inradius,
#         outradius,
#         stiffness,
#         theta0,
#         ne,
#         polyorder,
#         numqp,
#         penaltyfactor,
#     ) for ne in nelmts
# ]
#
# dx = 1.0 ./ nelmts
#
# u1err = [er[1] for er in err]
# u2err = [er[2] for er in err]
#
# u1rate = convergence_rate(dx, u1err)
# u2rate = convergence_rate(dx, u2err)
#
# df = DataFrame("Element Size" => dx, "U1 Error" => u1err, "U2 Error" => u2err)
# CSV.write("examples/transformation-strain/convergence.csv", df)

# sysmatrix = CutCell.SystemMatrix()
# sysrhs = CutCell.SystemRHS()
#
# CutCell.assemble_stress_mass_matrix!(sysmatrix, basis, cellquads, cutmesh)
# CutCell.assemble_stress_linear_form!(sysrhs, basis, cellquads, stiffness, sol, cutmesh)
# CutCell.assemble_transformation_stress_linear_form!(
#     sysrhs,
#     transfstress,
#     basis,
#     cellquads,
#     cutmesh,
# )
#
# matrix = CutCell.make_sparse_stress_operator(sysmatrix, cutmesh)
# rhs = CutCell.stress_rhs(sysrhs, cutmesh)
#
# stressvec = matrix \ rhs
# stress = reshape(stressvec, 3, :)
#
#
# quadcoords = compute_quadrature_points(cellquads, cutmesh)
# dispquadvals = compute_field_at_quadrature_points(disp, basis, cellquads, cutmesh)
# stressquadvals = compute_field_at_quadrature_points(stress, basis, cellquads, cutmesh)
# #
# triin = Triangulate.TriangulateIO()
# triin.pointlist = quadcoords
# (triout, vorout) = triangulate("", triin)
# connectivity = triout.trianglelist
# cells = [
#     MeshCell(VTKCellTypes.VTK_TRIANGLE, connectivity[:, i]) for i = 1:size(connectivity)[2]
# ]
# vtkfile = vtk_grid(
#     "examples/transformation-strain/exact-bc-R4",
#     quadcoords[1, :],
#     quadcoords[2, :],
#     cells,
# )
# vtkfile["displacement"] = (dispquadvals[1, :], dispquadvals[2, :])
# vtkfile["s11"] = stressquadvals[1, :]
# vtkfile["s22"] = stressquadvals[2, :]
# vtkfile["s12"] = stressquadvals[3, :]
# outfiles = vtk_save(vtkfile)
