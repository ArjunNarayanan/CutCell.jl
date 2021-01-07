using Triangulate
using WriteVTK
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("../../../test/useful_routines.jl")

function update_product_stress!(
    qpstress,
    basis,
    stiffness,
    transfstress,
    theta0,
    celldisp,
    points,
    jac,
    vectosymmconverter,
)
    dim = CutCell.dimension(basis)
    lambda, mu = CutCell.lame_coefficients(stiffness, +1)
    nump = size(points)[2]

    for qpidx = 1:nump
        p = points[:, qpidx]
        grad = CutCell.transform_gradient(gradient(basis, p), jac)
        NK = sum([CutCell.make_row_matrix(vectosymmconverter[k], grad[:, k]) for k = 1:dim])
        symmdispgrad = NK * celldisp

        inplanestress = (stiffness[+1] * symmdispgrad) - transfstress
        s33 = lambda * (symmdispgrad[1] + symmdispgrad[2]) - (lambda + 2mu / 3) * theta0

        numericalstress = vcat(inplanestress, s33)

        append!(qpstress, numericalstress)
    end
end

function update_parent_stress!(
    qpstress,
    basis,
    stiffness,
    celldisp,
    points,
    jac,
    vectosymmconverter,
)

    dim = CutCell.dimension(basis)
    lambda, mu = CutCell.lame_coefficients(stiffness, -1)
    nump = size(points)[2]
    for qpidx = 1:nump
        p = points[:, qpidx]
        grad = CutCell.transform_gradient(gradient(basis, p), jac)
        NK = sum([CutCell.make_row_matrix(vectosymmconverter[k], grad[:, k]) for k = 1:dim])
        symmdispgrad = NK * celldisp

        inplanestress = stiffness[-1] * symmdispgrad
        s33 = lambda * (symmdispgrad[1] + symmdispgrad[2])

        numericalstress = vcat(inplanestress, s33)

        append!(qpstress, numericalstress)
    end
end

function compute_stress_at_quadrature_points(
    nodaldisplacement,
    basis,
    stiffness,
    transfstress,
    theta0,
    cellquads,
    cutmesh,
)

    dim = CutCell.dimension(basis)
    ncells = CutCell.number_of_cells(cutmesh)
    qpstress = zeros(0)
    jac = CutCell.jacobian(cutmesh)
    vectosymmconverter = CutCell.vector_to_symmetric_matrix_converter()

    for cellid = 1:ncells
        s = CutCell.cell_sign(cutmesh, cellid)
        @assert s == -1 || s == 0 || s == +1
        if s == +1 || s == 0
            nodeids = CutCell.nodal_connectivity(cutmesh, +1, cellid)
            celldofs = CutCell.element_dofs(nodeids, dim)
            celldisp = nodaldisplacement[celldofs]
            points = cellquads[+1, cellid].points
            update_product_stress!(
                qpstress,
                basis,
                stiffness,
                transfstress,
                theta0,
                celldisp,
                points,
                jac,
                vectosymmconverter,
            )
        end
        if s == -1 || s == 0
            nodeids = CutCell.nodal_connectivity(cutmesh, -1, cellid)
            celldofs = CutCell.element_dofs(nodeids, dim)
            celldisp = nodaldisplacement[celldofs]
            points = cellquads[-1, cellid].points
            update_parent_stress!(
                qpstress,
                basis,
                stiffness,
                celldisp,
                points,
                jac,
                vectosymmconverter,
            )
        end
    end
    return reshape(qpstress, 4, :)
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

function onboundary(x, L, W)
    return x[2] ≈ 0.0 || x[1] ≈ L || x[2] ≈ W || x[1] ≈ 0.0
end

function bulk_modulus(l, m)
    return l + 2m / 3
end

function lame_lambda(k, m)
    return k - 2m / 3
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


function solve_cylindrical_bc_elasticity(
    width,
    center,
    inradius,
    outradius,
    stiffness,
    transfstress,
    theta0,
    nelmts,
    polyorder,
    numqp,
    penaltyfactor,
)

    lambda1, mu1 = CutCell.lame_coefficients(stiffness, +1)
    lambda2, mu2 = CutCell.lame_coefficients(stiffness, -1)

    analyticalsolution =
        AnalyticalSolution(inradius, outradius, center, lambda1, mu1, lambda2, mu2, theta0)

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
    cellquads = CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
    interfacequads = CutCell.InterfaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
    facequads = CutCell.FaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)

    bilinearforms = CutCell.BilinearForms(basis, cellquads, stiffness, cutmesh)
    interfacecondition =
        CutCell.incoherent_interface_condition(basis, interfacequads, stiffness, cutmesh, penalty)

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
    CutCell.assemble_incoherent_interface_transformation_rhs!(
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

    return sol, basis, cellquads, cutmesh
end

function solve_zero_traction_bc_elasticity(
    width,
    center,
    inradius,
    outradius,
    stiffness,
    transfstress,
    theta0,
    nelmts,
    polyorder,
    numqp,
    penaltyfactor,
)

    lambda1, mu1 = CutCell.lame_coefficients(stiffness, +1)
    lambda2, mu2 = CutCell.lame_coefficients(stiffness, -1)

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
    cellquads = CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
    interfacequads = CutCell.InterfaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
    facequads = CutCell.FaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)

    bilinearforms = CutCell.BilinearForms(basis, cellquads, stiffness, cutmesh)
    interfacecondition =
        CutCell.incoherent_interface_condition(basis, interfacequads, stiffness, cutmesh, penalty)

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

    topleftnodeid = CutCell.nodes_per_mesh_side(mesh)[2]
    CutCell.apply_dirichlet_bc!(matrix, rhs, [1, topleftnodeid], 1, 0.0, 2)
    CutCell.apply_dirichlet_bc!(matrix, rhs, [1], 2, 0.0, 2)

    sol = matrix \ rhs

    return sol, basis, cellquads, cutmesh
end





K1, K2 = 247.0, 192.0
mu1, mu2 = 126.0, 87.0
lambda1 = lame_lambda(K1, mu1)
lambda2 = lame_lambda(K2, mu2)

theta0 = -0.067
stiffness = CutCell.HookeStiffness(lambda1, mu1, lambda2, mu2)

width = 1.0
penaltyfactor = 1e2

nelmts = 11
polyorder = 2
numqp = required_quadrature_order(polyorder) + 4
center = [width / 2, width / 2]
inradius = width / 4
outradius = width

transfstress = CutCell.plane_strain_transformation_stress(lambda1, mu1, theta0)

# sol, basis, cellquads, cutmesh = solve_cylindrical_bc_elasticity(
#     width,
#     center,
#     inradius,
#     outradius,
#     stiffness,
#     transfstress,
#     theta0,
#     nelmts,
#     polyorder,
#     numqp,
#     penaltyfactor,
# )

sol, basis, cellquads, cutmesh = solve_zero_traction_bc_elasticity(
    width,
    center,
    inradius,
    outradius,
    stiffness,
    transfstress,
    theta0,
    nelmts,
    polyorder,
    numqp,
    penaltyfactor,
)

qpstress = compute_stress_at_quadrature_points(
    sol,
    basis,
    stiffness,
    transfstress,
    theta0,
    cellquads,
    cutmesh,
)

numqp = size(qpstress)[2]
pressure = -[(qpstress[1, i] + qpstress[2, i] + qpstress[4, i])/3 for i = 1:numqp]

quadcoords = compute_quadrature_points(cellquads, cutmesh)

triin = Triangulate.TriangulateIO()
triin.pointlist = quadcoords
(triout, vorout) = triangulate("", triin)
connectivity = triout.trianglelist
cells = [
    MeshCell(VTKCellTypes.VTK_TRIANGLE, connectivity[:, i]) for i = 1:size(connectivity)[2]
]
vtkfile = vtk_grid(
    "examples/transformation-strain/cylindrical-particle/zero-traction-pressure",
    quadcoords[1, :],
    quadcoords[2, :],
    cells,
)
vtkfile["pressure"] = pressure
vtkfile["s11"] = qpstress[1,:]
vtkfile["s22"] = qpstress[2,:]
vtkfile["s12"] = qpstress[3,:]
vtkfile["s33"] = qpstress[4,:]
outfiles = vtk_save(vtkfile)
