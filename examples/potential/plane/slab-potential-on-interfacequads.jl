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

K1, K2 = 247.0, 192.0    # Pa
mu1, mu2 = 126.0, 87.0   # Pa
lambda1 = lame_lambda(K1, mu1)
lambda2 = lame_lambda(K2, mu2)
stiffness = CutCell.HookeStiffness(lambda1, mu1, lambda2, mu2)

rho1 = 1.0   # Kg/m^3
rho2 = 1.0   # Kg/m^3
V1 = inv(rho1)
V2 = inv(rho2)

theta0 = 0.0
diffG0 = 0.0
transfstress = CutCell.plane_strain_transformation_stress(lambda1, mu1, theta0)

width = 1.0
displacementscale = 0.01 * width
penaltyfactor = 1e2

polyorder = 1
numqp = required_quadrature_order(polyorder) + 2
nelmts = 3
interfacepoint = [0.5, 0.0]
interfacenormal = [1.0, 0.0]
# interfacecenter = [0.5, 0.5]
# interfaceradius = 0.25

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
# levelsetcoeffs = CutCell.levelset_coefficients(
#     x -> circle_distance_function(x, interfacecenter, interfaceradius),
#     mesh,
# )

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
nodalcoordinates = CutCell.nodal_coordinates(cutmesh)


tol = 1e-8
boundingradius = 3.0
refclosestpoints, refclosestcellids, refgradients =
    CutCell.closest_reference_points_on_zero_levelset(
        nodalcoordinates,
        refseedpoints,
        spatialseedpoints,
        seedcellids,
        levelset,
        levelsetcoeffs,
        mesh,
        tol,
        boundingradius,
    )
spclosestpoints = spatial_closest_points(refclosestpoints,refclosestcellids,cutmesh)


# spseedpoints = spatial_closest_points(refseedpoints, seedcellids, cutmesh)
# relativeposition = spclosestpoints .- interfacecenter
# angularposition = angular_position(relativeposition)

sortidx = sortperm(spclosestpoints[2,:])
spclosestpoints = spclosestpoints[:,sortidx]
refclosestpoints = refclosestpoints[:,sortidx]
refclosestcellids = refclosestcellids[sortidx]
spycoords = spclosestpoints[2,:]

# sortidx = sortperm(angularposition)
# angularposition = angularposition[sortidx]
# refclosestpoints = refclosestpoints[:,sortidx]
# refclosestcellids = refclosestcellids[sortidx]
#
productdisplacement = CutCell.displacement_at_reference_points(
    refclosestpoints,
    refclosestcellids,
    +1,
    basis,
    nodaldisplacement,
    cutmesh,
)

parentdisplacement = CutCell.displacement_at_reference_points(
    refclosestpoints,
    refclosestcellids,
    -1,
    basis,
    nodaldisplacement,
    cutmesh,
)


# using PyPlot
# fig, ax = PyPlot.subplots()
# ax.plot(angularposition, productdisplacement[1, :], label = "product u1")
# ax.plot(angularposition, parentdisplacement[1, :], label = "parent u1")
# ax.plot(angularposition, productdisplacement[2, :], label = "product u2")
# ax.plot(angularposition, parentdisplacement[2, :], label = "parent u2")
# ax.legend()
# ax.grid()
# fig

productstress = CutCell.product_stress_at_reference_points(
    refclosestpoints,
    refclosestcellids,
    basis,
    stiffness,
    transfstress,
    theta0,
    nodaldisplacement,
    cutmesh,
)
parentstress = CutCell.parent_stress_at_reference_points(
    refclosestpoints,
    refclosestcellids,
    basis,
    stiffness,
    nodaldisplacement,
    cutmesh,
)

using PyPlot
fig,ax = PyPlot.subplots(2,2)
ax[1,1].plot(spycoords,productstress[1,:],label="product")
ax[1,1].plot(spycoords,parentstress[1,:],label="parent")
ax[1,1].set_title("S11")
ax[2,1].plot(spycoords,productstress[2,:],label="product")
ax[2,1].plot(spycoords,parentstress[2,:],label="parent")
ax[2,1].set_title("S22")
ax[1,2].plot(spycoords,productstress[3,:],label="product")
ax[1,2].plot(spycoords,parentstress[3,:],label="parent")
ax[1,2].set_title("S12")
ax[2,2].plot(spycoords,productstress[4,:],label="product")
ax[2,2].plot(spycoords,parentstress[4,:],label="parent")
ax[2,2].set_title("S33")
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
