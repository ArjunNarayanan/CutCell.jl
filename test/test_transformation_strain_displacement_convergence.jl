using Test
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCell
include("useful_routines.jl")

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
        CutCell.coherent_interface_condition(basis, interfacequads, stiffness, cutmesh, penalty)

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
    CutCell.assemble_coherent_interface_transformation_rhs!(
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

function interface_displacement_error(
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
        CutCell.coherent_interface_condition(basis, interfacequads, stiffness, cutmesh, penalty)

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
    CutCell.assemble_coherent_interface_transformation_rhs!(
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

    err1 = interface_L2_error(disp, analyticalsolution, +1, basis, interfacequads, cutmesh)
    den1 = integral_norm_on_interface(analyticalsolution,interfacequads,+1,cutmesh,2)

    err2 = interface_L2_error(disp, analyticalsolution, -1, basis, interfacequads, cutmesh)
    den2 = integral_norm_on_interface(analyticalsolution,interfacequads,-1,cutmesh,2)

    return err1 ./ den1, err2 ./ den2
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

polyorder = 2
numqp = required_quadrature_order(polyorder) + 2

center = [width/2, width/2]
inradius = width / 4
outradius = width

powers = [3,4,5]
nelmts = [2^p + 1 for p in powers]

err = [
    displacement_error(
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
u1err = [er[1] for er in err]
u2err = [er[2] for er in err]

u1rate = convergence_rate(dx,u1err)
u2rate = convergence_rate(dx,u2err)

@test allapprox(u1rate,repeat([3.0],length(u1rate)),0.1)
@test allapprox(u2rate,repeat([3.0],length(u2rate)),0.1)





center = [width, width]
inradius = width / 2
outradius = 2width
powers = [3,4,5]
nelmts = [2^p + 1 for p in powers]

err = [
    displacement_error(
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
u1err = [er[1] for er in err]
u2err = [er[2] for er in err]

u1rate = convergence_rate(dx,u1err)
u2rate = convergence_rate(dx,u2err)

@test allapprox(u1rate,repeat([3.0],length(u1rate)),0.1)
@test allapprox(u2rate,repeat([3.0],length(u2rate)),0.1)



center = [width/2, width/2]
inradius = width / 4
outradius = width

powers = [2,3,4,5]
nelmts = [2^p + 1 for p in powers]

err = [
    interface_displacement_error(
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

perr = [er[1] for er in err]
nerr = [er[2] for er in err]

dx = 1.0 ./ nelmts
pu1err = [er[1] for er in perr]
pu2err = [er[2] for er in perr]

pu1rate = convergence_rate(dx,pu1err)
pu2rate = convergence_rate(dx,pu2err)

nu1err = [er[1] for er in nerr]
nu2err = [er[2] for er in nerr]

nu1rate = convergence_rate(dx,nu1err)
nu2rate = convergence_rate(dx,nu2err)

#
# @test allapprox(u1rate,repeat([3.0],length(u1rate)),0.1)
# @test allapprox(u2rate,repeat([3.0],length(u2rate)),0.1)
