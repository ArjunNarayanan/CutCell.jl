using LinearAlgebra
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("../../../test/useful_routines.jl")
include("../compute-stress.jl")


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

function interface_quadrature_points(interfacequads, cutmesh)
    cellsign = CutCell.cell_sign(cutmesh)
    cellids = findall(cellsign .== 0)
    points = zeros(0)

    for cellid in cellids
        cellmap = CutCell.cell_map(cutmesh, cellid)
        qp = cellmap(interfacequads[1, cellid].points)
        append!(points, qp)
    end
    return reshape(points, 2, :)
end

function angular_position(points)
    cpoints = points[1, :] + im * points[2, :]
    return rad2deg.(angle.(cpoints))
end

function pressure(stress)
    return -(stress[1, :] + stress[2, :] + stress[4, :]) / 3
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

function strain_energy(stiffnessmatrix,stressvals)
    compliancematrix = inv(stiffnessmatrix)
    symmdispgrad = compliancematrix * stressvals[1:3,:]
    
end

K1, K2 = 247.0, 192.0
mu1, mu2 = 126.0, 87.0
lambda1 = lame_lambda(K1, mu1)
lambda2 = lame_lambda(K2, mu2)

V1 = 1.0 / 3.68e-6
V2 = 1.0 / 3.93e-6
theta0 = -0.067
initial_pressure = 13.5
stiffness = CutCell.HookeStiffness(lambda1, mu1, lambda2, mu2)

width = 1.0
penaltyfactor = 1e2

polyorder = 2
numqp = required_quadrature_order(polyorder) + 4
nelmts = 11
center = [width / 2, width / 2]
inradius = width / 4
outradius = width

analyticalsolution =
    AnalyticalSolution(inradius, outradius, center, lambda1, mu1, lambda2, mu2, theta0)


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

bilinearforms = CutCell.BilinearForms(basis, cellquads, stiffness, cutmesh)
interfacecondition = CutCell.incoherent_interface_condition(
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

nodaldisplacement = matrix \ rhs


quadpoints = interface_quadrature_points(interfacequads, cutmesh)
relquadpoints = quadpoints .- center
angularposition = angular_position(relquadpoints)
sortidx = sortperm(angularposition)
angularposition = angularposition[sortidx]



parentstress = parent_stress_at_interface_quadrature_points(
    nodaldisplacement,
    basis,
    stiffness,
    interfacequads,
    cutmesh,
)
parentstress = parentstress[:,sortidx]

productstress = product_stress_at_interface_quadrature_points(
    nodaldisplacement,
    basis,
    stiffness,
    transfstress,
    theta0,
    interfacequads,
    cutmesh,
)
productstress = productstress[:,sortidx]

parentpressure = pressure(parentstress)
productpressure = pressure(productstress)


# parentstrainenergy = parent_strain_energy(parentsymmdispgrad,parentstress)
# productstrainenergy = product_strain_energy(productsymmdispgrad,productstress,theta0)

using PyPlot
fig,ax = PyPlot.subplots()
ax.plot(angularposition,parentpressure)
# ax.set_ylim(0.15,0.2)
ax.grid()
fig
