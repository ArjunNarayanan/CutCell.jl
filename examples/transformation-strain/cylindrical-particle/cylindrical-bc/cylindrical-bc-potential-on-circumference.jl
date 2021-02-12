using PyPlot
using LinearAlgebra
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("../../../../test/useful_routines.jl")

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
    function AnalyticalSolution(
        inradius,
        outradius,
        center,
        ls,
        ms,
        lc,
        mc,
        theta0,
    )
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


function shell_radial_stress(ls, ms, theta0, A1, A2, r)
    return (ls + 2ms) * (A1 - A2 / r^2) + ls * (A1 + A2 / r^2) -
           (ls + 2ms / 3) * theta0
end

function shell_circumferential_stress(ls, ms, theta0, A1, A2, r)
    return ls * (A1 - A2 / r^2) + (ls + 2ms) * (A1 + A2 / r^2) -
           (ls + 2ms / 3) * theta0
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
    Q = rotation_matrix(relpos, r)

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

function exact_stress(A::AnalyticalSolution, x)
    relpos = x - A.center
    r = sqrt(relpos' * relpos)
    if r < A.inradius
        return core_stress(A)
    else
        return shell_stress(A, x)
    end
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
    analyticalsolution,
)

    lambda1, mu1 = CutCell.lame_coefficients(stiffness, +1)
    transfstress =
        CutCell.plane_strain_transformation_stress(lambda1, mu1, theta0)
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

    return nodaldisplacement
end

function spatial_closest_points(refclosestpoints, refclosestcellids, mesh)
    dim, npts = size(refclosestpoints)
    spclosestpoints = zeros(dim, npts)
    for i = 1:npts
        cellmap = CutCell.cell_map(mesh, refclosestcellids[i])
        spclosestpoints[:, i] .= cellmap(refclosestpoints[:, i])
    end
    return spclosestpoints
end

function angular_position(points)
    cpoints = points[1, :] + im * points[2, :]
    return rad2deg.(angle.(cpoints))
end


K1, K2 = 247.0, 192.0    # Pa
mu1, mu2 = 126.0, 87.0   # Pa
lambda1 = lame_lambda(K1, mu1)
lambda2 = lame_lambda(K2, mu2)
stiffness = CutCell.HookeStiffness(lambda1, mu1, lambda2, mu2)

rho1 = 1.0   # Kg/m^3
rho2 = 1.0   # Kg/m^3

V1 = inv(rho1)
V2 = inv(rho2)
theta0 = -0.067
ΔG = -1022.0    # J/mol
molarmass = 0.147   # Kg/mol
diffG0 = ΔG / molarmass # J/Kg

width = 1.0
penaltyfactor = 1e4

polyorder = 3
numqp = required_quadrature_order(polyorder) + 2
nelmts = 17
center = [width / 2, width / 2]
inradius = width / 3
outradius = 2.0

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

transfstress = CutCell.plane_strain_transformation_stress(lambda1, mu1, theta0)

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
interfacequads =
    CutCell.InterfaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
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
    analyticalsolution,
)


refseedpoints, spatialseedpoints, seedcellids =
    CutCell.seed_zero_levelset_with_interfacequads(interfacequads, cutmesh)
normals = CutCell.collect_interface_normals(interfacequads, cutmesh)

spatialpoints = spatialseedpoints[1, :, :]
referencepoints = refseedpoints
referencecellids = seedcellids

relspatialpoints = spatialpoints .- center
angularposition = angular_position(relspatialpoints)
sortidx = sortperm(angularposition)
angularposition = angularposition[sortidx]

referencepoints = referencepoints[:, :, sortidx]
referencecellids = referencecellids[:, sortidx]
spatialpoints = spatialpoints[:, sortidx]
normals = normals[:, sortidx]


productdisplacement = CutCell.displacement_at_reference_points(
    referencepoints[1, :, :],
    referencecellids[1, :],
    +1,
    basis,
    nodaldisplacement,
    cutmesh,
)
parentdisplacement = CutCell.displacement_at_reference_points(
    referencepoints[2, :, :],
    referencecellids[2, :],
    -1,
    basis,
    nodaldisplacement,
    cutmesh,
)
exactdisplacement = mapslices(analyticalsolution, spatialpoints, dims = 1)


# fig, ax = PyPlot.subplots(2, 1)
# ax[1].plot(angularposition, productdisplacement[1, :], label = "product u1")
# ax[1].plot(angularposition, parentdisplacement[1, :], label = "parent u1")
# ax[1].plot(angularposition, exactdisplacement[1, :], "--", label = "exact u1")
# ax[1].grid()
# ax[1].legend()
# ax[2].plot(angularposition, productdisplacement[2, :], label = "product u2")
# ax[2].plot(angularposition, parentdisplacement[2, :], label = "parent u2")
# ax[2].plot(angularposition, exactdisplacement[2, :], "--", label = "exact u2")
# ax[2].grid()
# ax[2].legend()
# fig.tight_layout()
# fig




productstress = CutCell.product_stress_at_reference_points(
    referencepoints[1, :, :],
    referencecellids[1, :],
    basis,
    stiffness,
    transfstress,
    theta0,
    nodaldisplacement,
    cutmesh,
)
parentstress = CutCell.parent_stress_at_reference_points(
    referencepoints[2, :, :],
    referencecellids[2, :],
    basis,
    stiffness,
    nodaldisplacement,
    cutmesh,
)

exactproductstress =
    mapslices(x -> shell_stress(analyticalsolution, x), spatialpoints, dims = 1)
exactparentstress =
    mapslices(x -> core_stress(analyticalsolution), spatialpoints, dims = 1)


producttraction = CutCell.traction_force_at_points(productstress, normals)
parenttraction = CutCell.traction_force_at_points(parentstress, normals)

exactproducttraction = CutCell.traction_force_at_points(exactproductstress, normals)
exactparenttraction = CutCell.traction_force_at_points(exactparentstress, normals)

# fig, ax = PyPlot.subplots(2, 1)
# ax[1].plot(angularposition, producttraction[1, :], label = "product")
# ax[1].plot(angularposition, parenttraction[1, :], label = "parent")
# ax[1].plot(angularposition, exactproducttraction[1, :], "--", label = "exact product")
# ax[1].plot(angularposition, exactparenttraction[1, :], "--", label = "exact parent")
# ax[1].set_title("t1")
# ax[1].legend()
# ax[1].grid()
# ax[2].plot(angularposition, producttraction[2, :], label = "product")
# ax[2].plot(angularposition, parenttraction[2, :], label = "parent")
# ax[2].plot(angularposition, exactproducttraction[2, :], "--", label = "exact product")
# ax[2].plot(angularposition, exactparenttraction[2, :], "--", label = "exact parent")
# ax[2].set_title("t2")
# ax[2].legend()
# ax[2].grid()
# fig.tight_layout()
# fig





productpressure = CutCell.pressure_at_points(productstress)
parentpressure = CutCell.pressure_at_points(parentstress)


# fig,ax = PyPlot.subplots()
# ax.plot(angularposition,productpressure)
# ax.plot(angularposition,parentpressure)
# fig


productdevstress =
    CutCell.deviatoric_stress_at_points(productstress, productpressure)
parentdevstress =
    CutCell.deviatoric_stress_at_points(parentstress, parentpressure)

productdevnorm = CutCell.stress_inner_product_over_points(productdevstress)
parentdevnorm = CutCell.stress_inner_product_over_points(parentdevstress)

productnormaldevstress =
    CutCell.normal_stress_component_over_points(productdevstress, normals)
parentnormaldevstress =
    CutCell.normal_stress_component_over_points(parentdevstress, normals)

productspecificvolume = V1 * (1.0 .- productpressure / K1)
parentspecificvolume = V2 * (1.0 .- parentpressure / K2)


productp1 = productspecificvolume .* productpressure
productp2 = V1 / (2K1) * (productpressure .^ 2)
productp3 = -(productspecificvolume .* productnormaldevstress)
productp4 = V1 / (4mu1) * productdevnorm
productpotential = productp1 + productp2 + productp3 + productp4

parentp1 = parentspecificvolume .* parentpressure
parentp2 = V2 / (2K2) * (parentpressure .^ 2)
parentp3 = -(parentspecificvolume .* parentnormaldevstress)
parentp4 = V2 / (4mu2) * parentdevnorm
parentpotential = parentp1 + parentp2 + parentp3 + parentp4

potentialdifference = productpotential - parentpotential



folderpath = "examples/transformation-strain/cylindrical-particle/cylindrical-bc/"

# using PyPlot
# fig,ax = PyPlot.subplots()
# ax.plot(angularposition,productpressure,label="product")
# # ax.plot(angularposition,parentpressure,label="parent")
# ax.grid()
# ax.set_title("Pressure along interface")
# ax.set_ylabel("Pressure (Pa)")
# ax.set_xlabel("Angular position (deg)")
# ax.legend()
# fig.tight_layout()
# fig
# fig.savefig(folderpath*"pressure.png")


# using PyPlot
# fig,ax = PyPlot.subplots()
# ax.plot(angularposition,parentp1,label="term1")
# ax.plot(angularposition,parentp2,label="term2")
# ax.plot(angularposition,parentp3,label="term3")
# ax.plot(angularposition,parentp4,label="term4")
# ax.plot(angularposition,parentpotential,label="potential")
# ax.set_xlabel("Angular position (deg)")
# ax.set_ylabel("Energy density (J/Kg)")
# ax.set_title("Potential on parent side of interface")
# # ax.set_ylim(-1e6,2e6)
# ax.grid()
# ax.legend()
# fig.tight_layout()
# fig
# fig.savefig(folderpath*"parent-potential.png")
#
#
#
# fig,ax = PyPlot.subplots()
# ax.plot(angularposition,productp1,label="term1")
# ax.plot(angularposition,productp2,label="term2")
# ax.plot(angularposition,productp3,label="term3")
# ax.plot(angularposition,productp4,label="term4")
# ax.plot(angularposition,productpotential,label="potential")
# ax.set_xlabel("Angular position (deg)")
# ax.set_ylabel("Energy density (J/Kg)")
# # ax.set_ylim(-1e6,2e6)
# ax.set_title("Potential on product side of interface")
# ax.grid()
# ax.legend()
# fig.tight_layout()
# fig
# fig.savefig(folderpath*"product-potential.png")
#
#
#
fig, ax = PyPlot.subplots()
ax.plot(angularposition, potentialdifference)
ax.grid()
ax.set_xlabel("Angular position (deg)")
ax.set_ylabel("Potential difference (J/Kg)")
ax.set_title("Potential difference along interface circumference")
fig.tight_layout()
# ax.set_ylim(0.2,0.3)
fig
# fig.savefig(folderpath*"potential-difference.png")
