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

function variational_stress(
    nodaldisplacement,
    basis,
    cellquads,
    stiffness,
    cutmesh,
)
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

    matrix = CutCell.make_sparse_stress_operator(sysmatrix, cutmesh)
    rhs = CutCell.stress_rhs(sysrhs, cutmesh)

    stressvec = matrix \ rhs
    stress = reshape(stressvec, 3, :)

    return stress
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

function normal_stress_component(stressvector, normal)
    snn =
        normal[1] * stressvector[1] * normal[1] +
        2.0 * normal[1] * stressvector[3] * normal[2] +
        normal[2] * stressvector[2] * normal[2]
end

function normal_stress_component_over_points(stresses, normals)
    numstresscomponents, npts = size(stresses)
    @assert size(normals) == (2, npts)

    normalstresscomp = zeros(npts)
    for i = 1:npts
        normalstresscomp[i] =
            normal_stress_component(stresses[:, i], normals[:, i])
    end
    return normalstresscomp
end

function stress_inner_product_over_points(stresses)
    return stresses[1, :] .^ 2 +
           stresses[2, :] .^ 2 +
           2.0 * stresses[3, :] .^ 2 +
           stresses[4, :] .^ 2
end

function angular_position(points)
    cpoints = points[1, :] + im * points[2, :]
    return rad2deg.(angle.(cpoints))
end

K1, K2 = 247.0, 247.0    # Pa
mu1, mu2 = 126.0, 126.0   # Pa
lambda1 = lame_lambda(K1, mu1)
lambda2 = lame_lambda(K2, mu2)
stiffness = CutCell.HookeStiffness(lambda1, mu1, lambda2, mu2)

theta0 = 0.0
transfstress = CutCell.plane_strain_transformation_stress(lambda1, mu1, theta0)

width = 1.0
displacementscale = 0.01 * width
penaltyfactor = 1e2

polyorder = 2
numqp = required_quadrature_order(polyorder) + 2
nelmts = 9
interfacepoint = [0.5, 0.0]
interfacenormal = [1.0, 0.0]

dx = width / nelmts
meanmoduli = 0.5 * (lambda1 + lambda2 + mu1 + mu2)
penalty = penaltyfactor / dx * meanmoduli

basis = TensorProductBasis(2, polyorder)
mesh = CutCell.Mesh([0.0, 0.0], [width, width], [nelmts, nelmts], basis)
levelset = InterpolatingPolynomial(1, basis)
levelsetcoeffs = CutCell.levelset_coefficients(
    x -> plane_distance_function(x, interfacenormal, interfacepoint),
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


nodalstress =
    variational_stress(nodaldisplacement, basis, cellquads, stiffness, cutmesh)


refseedpoints, spatialseedpoints, seedcellids =
    CutCell.seed_zero_levelset_with_interfacequads(interfacequads, cutmesh)

# nodalcoordinates = CutCell.nodal_coordinates(cutmesh)
# tol = 1e-8
# boundingradius = 3.0
# refclosestpoints, refclosestcellids, refgradients =
#     CutCell.closest_reference_points_on_zero_levelset(
#         nodalcoordinates,
#         refseedpoints,
#         spatialseedpoints,
#         seedcellids,
#         levelset,
#         levelsetcoeffs,
#         mesh,
#         tol,
#         boundingradius,
#     )

# spclosestpoints =
#     spatial_closest_points(refclosestpoints, refclosestcellids, cutmesh)

spatialpoints = spatialseedpoints
referencepoints = refseedpoints
referencecellids = seedcellids

sortidx = sortperm(spatialpoints[2, :])
spatialpoints = spatialpoints[:, sortidx]
referencepoints = referencepoints[:, sortidx]
referencecellids = referencecellids[sortidx]
spycoords = spatialpoints[2, :]

# productdisplacement = CutCell.displacement_at_reference_points(
#     referencepoints,
#     referencecellids,
#     +1,
#     basis,
#     nodaldisplacement,
#     cutmesh,
# )
#
# parentdisplacement = CutCell.displacement_at_reference_points(
#     referencepoints,
#     referencecellids,
#     -1,
#     basis,
#     nodaldisplacement,
#     cutmesh,
# )
#
# exactdisplacement = hcat(
#     [
#         displacement(displacementscale, spatialpoints[:, i]) for
#         i = 1:size(spatialpoints)[2]
#     ]...,
# )

# using PyPlot
# fig, ax = PyPlot.subplots(2,1)
# ax[1].plot(spycoords, productdisplacement[1, :], label = "product u1")
# ax[1].plot(spycoords, parentdisplacement[1, :], label = "parent u1")
# ax[1].plot(spycoords, exactdisplacement[1, :], "--", label = "exact")
# ax[1].grid()
# ax[1].legend()
# ax[2].plot(spycoords, productdisplacement[2, :], label = "product u2")
# ax[2].plot(spycoords, parentdisplacement[2, :], label = "parent u2")
# ax[2].plot(spycoords, exactdisplacement[2, :], "--", label = "exact")
# ax[2].grid()
# ax[2].legend()
# fig.tight_layout()
# fig

productstress = CutCell.interpolate_at_reference_points(
    referencepoints,
    referencecellids,
    +1,
    basis,
    nodalstress,
    3,
    cutmesh,
)
parentstress = CutCell.interpolate_at_reference_points(
    referencepoints,
    referencecellids,
    -1,
    basis,
    nodalstress,
    3,
    cutmesh,
)

exactstress = hcat(
    [
        stress_field(lambda1, mu1, displacementscale, spatialpoints[:, i])
        for i = 1:size(spatialpoints)[2]
    ]...,
)

using PyPlot
fig, ax = PyPlot.subplots(2, 1)
ax[1].plot(spycoords, productstress[1, :], label = "product")
ax[1].plot(spycoords, parentstress[1, :], label = "parent")
ax[1].plot(spycoords, exactstress[1, :], "--", label = "exact")
ax[1].set_title("S11")
ax[1].legend()
ax[1].grid()
ax[2].plot(spycoords, productstress[3, :], label = "product")
ax[2].plot(spycoords, parentstress[3, :], label = "parent")
ax[2].plot(spycoords, exactstress[3, :], "--", label = "exact")
ax[2].set_title("S12")
ax[2].legend()
ax[2].grid()
ax[2].set_ylim(2,2.5)
fig.tight_layout()
# ax.grid()
# ax.legend()
fig

# using PyPlot
# fig,ax = PyPlot.subplots()
# ax.plot(angularposition,productstress[1,:],label="product")
# ax.plot(angularposition,parentstress[1,:],label="parent")
# ax.grid()
# ax.legend()
# fig
