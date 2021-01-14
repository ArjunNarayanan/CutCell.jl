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

function displacement(x)
    u1 = x[1]^2 + 2 * x[1] * x[2]
    u2 = x[2]^2 + 3x[1]
    return [u1, u2]
end

function body_force(lambda, mu)
    b1 = -2 * (lambda + 2mu)
    b2 = -(4lambda + 6mu)
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
        x -> displacement(x),
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
        x -> body_force(lambda, mu),
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


K1, K2 = 247.0, 247.0    # Pa
mu1, mu2 = 126.0, 126.0   # Pa
lambda1 = lame_lambda(K1, mu1)
lambda2 = lame_lambda(K2, mu2)
stiffness = CutCell.HookeStiffness(lambda1, mu1, lambda2, mu2)

theta0 = 0.0
transfstress = CutCell.plane_strain_transformation_stress(lambda1,mu1,theta0)

width = 1.0
displacementscale = 0.01 * width
penaltyfactor = 1e2

polyorder = 2
numqp = required_quadrature_order(polyorder) + 2
nelmts = 3
delta = 0.01
interfacepoint = [1/3+delta, 1/3]
interfaceangle = 45
interfacenormal = [cosd(interfaceangle),sind(interfaceangle)]

dx = width / nelmts
meanmoduli = 0.5 * (lambda1 + lambda2 + mu1 + mu2)
penalty = penaltyfactor * meanmoduli

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


refseedpoints, spatialseedpoints, seedcellids =
    CutCell.seed_zero_levelset_with_interfacequads(interfacequads, cutmesh)
interfacenormals = CutCell.collect_interface_normals(interfacequads,cutmesh)

spatialpoints = spatialseedpoints
referencepoints = refseedpoints
referencecellids = seedcellids

sortidx = sortperm(spatialpoints[2, :])
spatialpoints = spatialpoints[:, sortidx]
referencepoints = referencepoints[:, sortidx]
referencecellids = referencecellids[sortidx]
spycoords = spatialpoints[2, :]
interfacenormals = interfacenormals[:,sortidx]

productdisplacement = CutCell.displacement_at_reference_points(
    referencepoints,
    referencecellids,
    +1,
    basis,
    nodaldisplacement,
    cutmesh,
)

parentdisplacement = CutCell.displacement_at_reference_points(
    referencepoints,
    referencecellids,
    -1,
    basis,
    nodaldisplacement,
    cutmesh,
)

exactdisplacement = hcat(
    [
        displacement(spatialpoints[:, i]) for
        i = 1:size(spatialpoints)[2]
    ]...,
)

# fig, ax = PyPlot.subplots(2,1)
# ax[1].plot(spycoords, productdisplacement[1, :], label = "product u1")
# ax[1].plot(spycoords, parentdisplacement[1, :], label = "parent u1")
# # ax[1].plot(spycoords, exactdisplacement[1, :], "--", label = "exact")
# ax[1].grid()
# ax[1].legend()
# ax[2].plot(spycoords, productdisplacement[2, :], label = "product u2")
# ax[2].plot(spycoords, parentdisplacement[2, :], label = "parent u2")
# # ax[2].plot(spycoords, exactdisplacement[2, :], "--", label = "exact")
# ax[2].grid()
# ax[2].legend()
# fig.tight_layout()
# fig

productstress = CutCell.product_stress_at_reference_points(
    referencepoints,
    referencecellids,
    basis,
    stiffness,
    transfstress,
    theta0,
    nodaldisplacement,
    cutmesh,
)
parentstress = CutCell.parent_stress_at_reference_points(
    referencepoints,
    referencecellids,
    basis,
    stiffness,
    nodaldisplacement,
    cutmesh,
)
#
# exactstress = hcat([stress_field(lambda1,mu1,displacementscale,spatialpoints[:,i]) for i = 1:size(spatialpoints)[2]]...)


producttraction = CutCell.traction_force_at_points(productstress,interfacenormals)
parenttraction = CutCell.traction_force_at_points(parentstress,interfacenormals)

fig, ax = PyPlot.subplots(2, 1)
ax[1].plot(spycoords, producttraction[1, :], label = "product")
ax[1].plot(spycoords, parenttraction[1, :], label = "parent")
ax[1].set_title("t1")
ax[1].legend()
ax[1].grid()
ax[2].plot(spycoords, producttraction[2, :], label = "product")
ax[2].plot(spycoords, parenttraction[2, :], label = "parent")
ax[2].set_title("t2")
ax[2].legend()
ax[2].grid()
fig.tight_layout()
fig
#
#
# using Statistics
# difftraction = parenttraction - producttraction
#
# idx = findall(difftraction[1,:] .> 3.0)
# oscillating_cells = referencecellids[idx]
#
# fig, ax = PyPlot.subplots(2, 1)
# ax[1].plot(spycoords, difftraction[1, :])
# ax[1].set_title("diff t1")
# ax[1].legend()
# ax[1].grid()
# ax[2].plot(spycoords, difftraction[2, :])
# ax[2].set_title("diff t2")
# ax[2].legend()
# ax[2].grid()
# fig.tight_layout()
# fig




# fig, ax = PyPlot.subplots(3, 1)
# ax[1].plot(spycoords, productstress[1, :], label = "product")
# ax[1].plot(spycoords, parentstress[1, :], label = "parent")
# ax[1].plot(spycoords, exactstress[1,:], "--", label = "exact")
# ax[1].set_title("S11")
# ax[1].legend()
# ax[1].grid()
# ax[2].plot(spycoords, productstress[2, :], label = "product")
# ax[2].plot(spycoords, parentstress[2, :], label = "parent")
# ax[2].plot(spycoords, exactstress[2,:], "--", label = "exact")
# ax[2].set_title("S22")
# ax[2].legend()
# ax[2].grid()
# ax[3].plot(spycoords, productstress[3, :], label = "product")
# ax[3].plot(spycoords, parentstress[3, :], label = "parent")
# ax[3].plot(spycoords, exactstress[3,:], "--", label = "exact")
# ax[3].set_title("S12")
# ax[3].legend()
# ax[3].grid()
# # ax[3].set_ylim(2,2.5)
# fig.tight_layout()
# fig

# productpressure = CutCell.pressure_at_points(productstress)
# parentpressure = CutCell.pressure_at_points(parentstress)
#
# fig,ax = PyPlot.subplots()
# ax.plot(spycoords,productpressure,label="product")
# ax.plot(spycoords,parentpressure,label="parent")
# ax.legend()
# ax.grid()
# ax.set_title("Pressure on interface")
# fig
