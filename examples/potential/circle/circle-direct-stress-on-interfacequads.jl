using PyPlot
using LinearAlgebra
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

function displacement(alpha, x)
    u1 = alpha * x[2] * sin(pi * x[1])
    u2 = alpha * (x[1]^3 + cos(pi * x[2]))
    return [u1, u2]
end

function body_force(lambda, mu, alpha, x)
    b1 = alpha * (lambda + 2mu) * pi^2 * x[2] * sin(pi * x[1])
    b2 =
        -alpha * (6mu * x[1] + (lambda + mu) * pi * cos(pi * x[1])) +
        alpha * (lambda + 2mu) * pi^2 * cos(pi * x[2])
    return [b1, b2]
end

function stress_field(lambda, mu, alpha, x)
    s11 =
        (lambda + 2mu) * alpha * pi * x[2] * cos(pi * x[1]) -
        lambda * alpha * pi * sin(pi * x[2])
    s22 =
        -(lambda + 2mu) * alpha * pi * sin(pi * x[2]) +
        lambda * alpha * pi * x[2] * cos(pi * x[1])
    s12 = alpha * mu * (3x[1]^2 + sin(pi * x[1]))
    return [s11, s22, s12]
end

function onboundary(x, L, W)
    return x[2] ≈ 0.0 || x[1] ≈ L || x[2] ≈ W || x[1] ≈ 0.0
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
    penalty,
    alpha,
)

    L, W = CutCell.widths(cutmesh)
    lambda, mu = CutCell.lame_coefficients(stiffness, +1)

    bilinearforms = CutCell.BilinearForms(basis, cellquads, stiffness, cutmesh)
    interfacecondition = CutCell.coherent_interface_condition(
        basis,
        interfacequads,
        stiffness,
        cutmesh,
        penalty,
    )
    displacementbc = CutCell.DisplacementCondition(
        x -> displacement(alpha, x),
        basis,
        facequads,
        stiffness,
        cutmesh,
        x -> onboundary(x, L, W),
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
    CutCell.assemble_body_force_linear_form!(
        sysrhs,
        x -> body_force(lambda, mu, alpha, x),
        basis,
        cellquads,
        cutmesh,
    )
    CutCell.assemble_penalty_displacement_bc!(
        sysmatrix,
        sysrhs,
        displacementbc,
        cutmesh,
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

function traction_force(stressvector, normal)
    return [
        stressvector[1] * normal[1] + stressvector[3] * normal[2],
        stressvector[3] * normal[1] + stressvector[2] * normal[2],
    ]
end

function traction_force_at_points(stresses, normals)
    npts = size(stresses)[2]
    @assert size(normals)[2] == npts

    return hcat(
        [traction_force(stresses[:, i], normals[:, i]) for i = 1:npts]...,
    )
end

K1, K2 = 247.0, 192.0    # Pa
mu1, mu2 = 126.0, 87.0   # Pa
lambda1 = lame_lambda(K1, mu1)
lambda2 = lame_lambda(K2, mu2)
stiffness = CutCell.HookeStiffness(lambda1, mu1, lambda2, mu2)

theta0 = 0.0
transfstress = CutCell.plane_strain_transformation_stress(lambda1, mu1, theta0)

width = 1.0
displacementscale = 0.1
penaltyfactor = 1e2

polyorder = 2
numqp = required_quadrature_order(polyorder) + 2
nelmts = 17
center = [width / 2, width / 2]
inradius = width / 3

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
    penalty,
    displacementscale,
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
spatialpoints = spatialpoints[:, sortidx]
referencecellids = referencecellids[:, sortidx]
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
exactdisplacement = hcat(
    [
        displacement(displacementscale, spatialpoints[:, i]) for
        i = 1:size(spatialpoints)[2]
    ]...,
)





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
#




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
exactstress = mapslices(
    x -> stress_field(lambda1, mu1, displacementscale, x),
    spatialpoints,
    dims = 1,
)


producttraction = traction_force_at_points(productstress, normals)
parenttraction = traction_force_at_points(parentstress, normals)
exacttraction = traction_force_at_points(exactstress, normals)

fig, ax = PyPlot.subplots(2, 1)
ax[1].plot(angularposition, producttraction[1, :], label = "product")
ax[1].plot(angularposition, parenttraction[1, :], label = "parent")
ax[1].plot(angularposition, exacttraction[1, :], "--", label = "exact")
ax[1].set_title("t1")
ax[1].legend()
ax[1].grid()
ax[2].plot(angularposition, producttraction[2, :], label = "product")
ax[2].plot(angularposition, parenttraction[2, :], label = "parent")
ax[2].plot(angularposition, exacttraction[2, :], "--", label = "exact")
ax[2].set_title("t2")
ax[2].legend()
ax[2].grid()
fig.tight_layout()
fig

# productpressure = CutCell.pressure_at_points(productstress)
# parentpressure = CutCell.pressure_at_points(parentstress)
#
#
# fig,ax = PyPlot.subplots()
# ax.plot(angularposition,productpressure,label="product")
# ax.plot(angularposition,parentpressure,label="parent")
# ax.legend()
# ax.grid()
# ax.set_title("Pressure on interface")
# fig


# productdevstress = CutCell.deviatoric_stress_at_points(productstress,productpressure)
# parentdevstress = CutCell.deviatoric_stress_at_points(parentstress,parentpressure)
#
# productdevnorm = CutCell.stress_inner_product_over_points(productdevstress)
# parentdevnorm = CutCell.stress_inner_product_over_points(parentdevstress)
#
# productnormaldevstress = CutCell.normal_stress_component_over_points(productdevstress,normals)
# parentnormaldevstress = CutCell.normal_stress_component_over_points(parentdevstress,normals)
#
# productspecificvolume = V1*(1.0 .- productpressure/K1)
# parentspecificvolume = V2*(1.0 .- parentpressure/K2)
#
#
# productp1 = productspecificvolume .* productpressure
# productp2 = V1/(2K1)*(productpressure.^2)
# productp3 = -(productspecificvolume .* productnormaldevstress)
# productp4 = V1/(4mu1)*productdevnorm
# productpotential = productp1+productp2+productp3+productp4
#
# parentp1 = parentspecificvolume .* parentpressure
# parentp2 = V2/(2K2)*(parentpressure.^2)
# parentp3 = -(parentspecificvolume .* parentnormaldevstress)
# parentp4 = V2/(4mu2)*parentdevnorm
# parentpotential = parentp1+parentp2+parentp3+parentp4
#
# potentialdifference = productpotential - parentpotential
