using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("../../test/useful_routines.jl")

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

function shell_radial_stress(ls, ms, theta0, A1, A2, r)
    return (ls + 2ms) * (A1 - A2 / r^2) + ls * (A1 + A2 / r^2) - (ls + 2ms / 3) * theta0
end

function shell_circumferential_stress(ls, ms, theta0, A1, A2, r)
    return ls * (A1 - A2 / r^2) + (ls + 2ms) * (A1 + A2 / r^2) - (ls + 2ms / 3) * theta0
end

function shell_out_of_plane_stress(ls, ms, A1, theta0)
    return 2 * ls * A1 - (ls + 2ms / 3) * theta0
end

function core_in_plane_stress(lc, mc, A1)
    return (lc + 2mc) * A1 + lc * A1
end

function core_out_of_plane_stress(lc, A1)
    return 2 * lc * A1
end

function rotation_matrix(x, r)
    costheta = x[1] / r
    sintheta = x[2] / r
    Q = [
        costheta -sintheta
        sintheta costheta
    ]
    return Q
end

function shell_stress(A::AnalyticalSolution, x)
    relpos = x - A.center
    r = sqrt(relpos' * relpos)
    Q = rotation_matrix(x, r)

    srr = shell_radial_stress(A.ls, A.ms, A.theta0, A.A1s, A.A2s, r)
    stt = shell_circumferential_stress(A.ls, A.ms, A.theta0, A.A1s, A.A2s, r)

    cylstress = [
        srr 0.0
        0.0 stt
    ]

    cartstress = Q * cylstress * Q'
    s11 = cartstress[1, 1]
    s22 = cartstress[2, 2]
    s12 = cartstress[1, 2]
    s33 = shell_out_of_plane_stress(A.ls, A.ms, A.A1s, A.theta0)

    return [s11, s22, s12, s33]
end

function core_stress(A::AnalyticalSolution)
    s11 = core_in_plane_stress(A.lc, A.mc, A.A1c)
    s33 = core_out_of_plane_stress(A.lc, A.A1c)
    return [s11, s11, 0.0, s33]
end

function onboundary(x, L, W)
    return x[2] ≈ 0.0 || x[1] ≈ L || x[2] ≈ W || x[1] ≈ 0.0
end

function update_core_stress_error(
    err,
    basis,
    stiffness,
    celldisp,
    quad,
    jac,
    detjac,
    vectosymmconverter,
    analyticalsolution,
)
    dim = CutCell.dimension(basis)
    lambda, mu = CutCell.lame_coefficients(stiffness, -1)
    for (p, w) in quad
        grad = CutCell.transform_gradient(gradient(basis, p), jac)
        NK = sum([CutCell.make_row_matrix(vectosymmconverter[k], grad[:, k]) for k = 1:dim])
        symmdispgrad = NK * celldisp

        inplanestress = stiffness[-1] * symmdispgrad
        s33 = lambda * (symmdispgrad[1] + symmdispgrad[2])

        numericalstress = vcat(inplanestress, s33)
        exactstress = core_stress(analyticalsolution)

        err .+= (numericalstress - exactstress) .^ 2 * detjac * w
    end
end

function compute_core_stress_error(
    displacement,
    basis,
    interfacequads,
    stiffness,
    cutmesh,
    analyticalsolution,
)
    err = zeros(4)
    cellsign = CutCell.cell_sign(cutmesh)
    cellids = findall(cellsign .== 0)

    dim = CutCell.dimension(basis)
    jac = CutCell.jacobian(cutmesh)
    detjac = CutCell.determinant_jacobian(cutmesh)
    vectosymmconverter = CutCell.vector_to_symmetric_matrix_converter()

    for cellid in cellids
        nodeids = CutCell.nodal_connectivity(cutmesh, -1, cellid)
        celldofs = CutCell.element_dofs(nodeids, dim)
        celldisp = displacement[celldofs]
        quad = interfacequads[-1, cellid]

        update_core_stress_error(
            err,
            basis,
            stiffness,
            celldisp,
            quad,
            jac,
            detjac,
            vectosymmconverter,
            analyticalsolution,
        )
    end

    den = abs(core_stress(analyticalsolution)[1] * 2pi * analyticalsolution.inradius)
    return sqrt.(err) / den
end

function compute_shell_stress_error(
    displacement,
    basis,
    interfacequads,
    stiffness,
    transfstrain,
    cutmesh,
    analyticalsolution,
)
    err = zeros(4)
    cellsign = CutCell.cell_sign(cutmesh)
    cellids = findall(cellsign .== 0)

    dim = CutCell.dimension(basis)
    jac = CutCell.jacobian(cutmesh)
    detjac = CutCell.determinant_jacobian(cutmesh)
    vectosymmconverter = CutCell.vector_to_symmetric_matrix_converter()

    for cellid in cellids
        nodeids = CutCell.nodal_connectivity(cutmesh, +1, cellid)
        celldofs = CutCell.element_dofs(nodeids, dim)
        celldisp = displacement[celldofs]
        quad = interfacequads[+1, cellid]


    end
end

function core_stress_error(
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

    corestresserr = compute_core_stress_error(
        sol,
        basis,
        interfacequads,
        stiffness,
        cutmesh,
        analyticalsolution,
    )

    return corestresserr
end

function convergence_rate(dx, err)
    return diff(log.(err)) ./ diff(log.(dx))
end


K1, K2 = 247.0, 192.0
mu1, mu2 = 126.0, 87.0
lambda1 = lame_lambda(K1, mu1)
lambda2 = lame_lambda(K2, mu2)


V1 = 1.0 / 3.68e-6
V2 = 1.0 / 3.93e-6
theta0 = -0.067
stiffness = CutCell.HookeStiffness(lambda1, mu1, lambda2, mu2)

width = 1.0
penaltyfactor = 1e2

polyorder = 3
numqp = required_quadrature_order(polyorder) + 4
center = [width / 2, width / 2]
inradius = width / 4
outradius = width

powers = 1:7
nelmts = [2^p + 1 for p in powers]


err = [
    core_stress_error(
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

s11err = [er[1] for er in err]
s11rate = convergence_rate(dx, s11err)







# lambda1, mu1 = CutCell.lame_coefficients(stiffness, +1)
# lambda2, mu2 = CutCell.lame_coefficients(stiffness, -1)
# transfstress = CutCell.plane_strain_transformation_stress(lambda1, mu1, theta0)
#
# analyticalsolution =
#     AnalyticalSolution(inradius, outradius, center, lambda1, mu1, lambda2, mu2, theta0)
#
# dx = width / nelmts
# meanmoduli = 0.5 * (lambda1 + lambda2 + mu1 + mu2)
# penalty = penaltyfactor / dx * meanmoduli
#
#
# basis = TensorProductBasis(2, polyorder)
# mesh = CutCell.Mesh([0.0, 0.0], [width, width], [nelmts, nelmts], basis)
# levelset = InterpolatingPolynomial(1, basis)
# levelsetcoeffs =
#     CutCell.levelset_coefficients(x -> -circle_distance_function(x, center, inradius), mesh)
#
# cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
# cellquads = CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
# interfacequads = CutCell.InterfaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
# facequads = CutCell.FaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
#
# bilinearforms = CutCell.BilinearForms(basis, cellquads, stiffness, cutmesh)
# interfacecondition =
#     CutCell.InterfaceCondition(basis, interfacequads, stiffness, cutmesh, penalty)
#
# displacementbc = CutCell.DisplacementCondition(
#     analyticalsolution,
#     basis,
#     facequads,
#     stiffness,
#     cutmesh,
#     x -> onboundary(x, width, width),
#     penalty,
# )
#
#
# sysmatrix = CutCell.SystemMatrix()
# sysrhs = CutCell.SystemRHS()
#
# CutCell.assemble_bilinear_form!(sysmatrix, bilinearforms, cutmesh)
# CutCell.assemble_interface_condition!(sysmatrix, interfacecondition, cutmesh)
# CutCell.assemble_bulk_transformation_linear_form!(
#     sysrhs,
#     transfstress,
#     basis,
#     cellquads,
#     cutmesh,
# )
# CutCell.assemble_interface_transformation_rhs!(
#     sysrhs,
#     transfstress,
#     basis,
#     interfacequads,
#     cutmesh,
# )
# CutCell.assemble_penalty_displacement_bc!(sysmatrix, sysrhs, displacementbc, cutmesh)
# CutCell.assemble_penalty_displacement_transformation_rhs!(
#     sysrhs,
#     transfstress,
#     basis,
#     facequads,
#     cutmesh,
#     x -> onboundary(x, width, width),
# )
#
#
# matrix = CutCell.make_sparse(sysmatrix, cutmesh)
# rhs = CutCell.rhs(sysrhs, cutmesh)
#
# sol = matrix \ rhs
#
# corestresserr = compute_core_stress_error(
#     sol,
#     basis,
#     interfacequads,
#     stiffness,
#     cutmesh,
#     analyticalsolution,
# )


# quadpoints = interface_quadrature_points(interfacequads, cutmesh)
# relquadpoints = quadpoints .- center
# angularposition = angular_position(relquadpoints)
# sortidx = sortperm(angularposition)
# angularposition = angularposition[idx]
#
#
# parentsymmdispgrad =
#     symmetric_displacement_gradient(sol, basis, interfacequads, cutmesh, -1)[:, sortidx]
# productsymmdispgrad =
#     symmetric_displacement_gradient(sol, basis, interfacequads, cutmesh, +1)[:, sortidx]
# productelasticsymmdispgrad = productsymmdispgrad .- transfstrain
#
#
# parentstress = parent_stress(parentsymmdispgrad, stiffness)
# productstress = product_stress(productelasticsymmdispgrad, stiffness, theta0)
#
# parentstrainenergy = parent_strain_energy(parentsymmdispgrad, parentstress)
# productstrainenergy =
#     product_strain_energy(productelasticsymmdispgrad, productstress, theta0)
#
# parentpressure = pressure(parentstress)
# productpressure = pressure(productstress)
#
# using PyPlot
# fig, ax = PyPlot.subplots()
# ax.plot(angularposition, productpressure)
# # ax.set_ylim(0.15,0.2)
# ax.grid()
# fig
