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
)

    bilinearforms = CutCell.BilinearForms(basis, cellquads, stiffness, cutmesh)
    interfacecondition = CutCell.coherent_interface_condition(
        basis,
        interfacequads,
        stiffness,
        cutmesh,
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

    matrix = CutCell.make_sparse(sysmatrix, cutmesh)
    rhs = CutCell.rhs(sysrhs, cutmesh)

    topleftnodeid = CutCell.nodes_per_mesh_side(cutmesh.mesh)[2]
    CutCell.apply_dirichlet_bc!(matrix, rhs, [1, topleftnodeid], 1, 0.0, 2)
    CutCell.apply_dirichlet_bc!(matrix, rhs, [1], 2, 0.0, 2)

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

function normal_stress_component_over_points(stresses,normals)
    numstresscomponents,npts = size(stresses)
    @assert size(normals) == (2,npts)

    normalstresscomp = zeros(npts)
    for i = 1:npts
        normalstresscomp[i] = normal_stress_component(stresses[:,i],normals[:,i])
    end
    return normalstresscomp
end

function stress_inner_product_over_points(stresses)
    return stresses[1,:].^2 + stresses[2,:].^2 + 2.0*stresses[3,:].^2 + stresses[4,:].^2
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
diffG0 = ΔG/molarmass # J/Kg

width = 1.0
penaltyfactor = 1e3

polyorder = 2
numqp = required_quadrature_order(polyorder) + 2
nelmts = 17
center = [width / 2, width / 2]
inradius = width / 3
outradius = width

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

# fig, ax = PyPlot.subplots(2, 1)
# ax[1].plot(angularposition, productdisplacement[1, :], label = "product u1")
# ax[1].plot(angularposition, parentdisplacement[1, :], label = "parent u1")
# ax[1].grid()
# ax[1].legend()
# ax[2].plot(angularposition, productdisplacement[2, :], label = "product u2")
# ax[2].plot(angularposition, parentdisplacement[2, :], label = "parent u2")
# ax[2].grid()
# ax[2].legend()
# fig.tight_layout()
# fig


productstress = CutCell.product_stress_at_reference_points(
    referencepoints[1,:,:],
    referencecellids[1,:],
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

producttraction = CutCell.traction_force_at_points(productstress, normals)
parenttraction = CutCell.traction_force_at_points(parentstress, normals)

# fig, ax = PyPlot.subplots(2, 1)
# ax[1].plot(angularposition, producttraction[1, :], label = "product")
# ax[1].plot(angularposition, parenttraction[1, :], label = "parent")
# ax[1].set_title("t1")
# ax[1].legend()
# ax[1].grid()
# ax[2].plot(angularposition, producttraction[2, :], label = "product")
# ax[2].plot(angularposition, parenttraction[2, :], label = "parent")
# ax[2].set_title("t2")
# ax[2].legend()
# ax[2].grid()
# fig.tight_layout()
# fig




productpressure = CutCell.pressure_at_points(productstress)
parentpressure = CutCell.pressure_at_points(parentstress)

productdevstress = CutCell.deviatoric_stress_at_points(productstress,productpressure)
parentdevstress = CutCell.deviatoric_stress_at_points(parentstress,parentpressure)

productdevnorm = CutCell.stress_inner_product_over_points(productdevstress)
parentdevnorm = CutCell.stress_inner_product_over_points(parentdevstress)

productnormaldevstress = CutCell.normal_stress_component_over_points(productdevstress,normals)
parentnormaldevstress = CutCell.normal_stress_component_over_points(parentdevstress,normals)

productspecificvolume = V1*(1.0 .- productpressure/K1)
parentspecificvolume = V2*(1.0 .- parentpressure/K2)


productp1 = productspecificvolume .* productpressure
productp2 = V1/(2K1)*(productpressure.^2)
productp3 = -(productspecificvolume .* productnormaldevstress)
productp4 = V1/(4mu1)*productdevnorm
productpotential = productp1+productp2+productp3+productp4

parentp1 = parentspecificvolume .* parentpressure
parentp2 = V2/(2K2)*(parentpressure.^2)
parentp3 = -(parentspecificvolume .* parentnormaldevstress)
parentp4 = V2/(4mu2)*parentdevnorm
parentpotential = parentp1+parentp2+parentp3+parentp4

potentialdifference = diffG0 .+ productpotential - parentpotential


# fig,ax = PyPlot.subplots()
# ax.plot(angularposition,potentialdifference)
# ax.grid()
# ax.set_xlabel("Angular position (deg)")
# ax.set_ylabel("Potential difference (J/Kg)")
# ax.set_title("Potential difference along interface circumference")
# fig.tight_layout()
# fig
# fig.savefig("potential-difference.png")
#
#
# fig,ax = PyPlot.subplots()
# ax.plot(angularposition,parentp1,label="term1")
# ax.plot(angularposition,parentp2,label="term2")
# ax.plot(angularposition,parentp3,label="term3")
# ax.plot(angularposition,parentp4,label="term4")
# ax.plot(angularposition,parentpotential,label="potential")
# ax.set_xlabel("Angular position (deg)")
# ax.set_ylabel("Energy density (J/Kg)")
# ax.set_title("Potential on parent side of interface")
# ax.set_ylim(-1e6,2e6)
# ax.grid()
# ax.legend()
# fig.tight_layout()
# fig.savefig("parent-potential.png")
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
# ax.set_ylim(-1e6,2e6)
# ax.set_title("Potential on product side of interface")
# ax.grid()
# ax.legend()
# fig.tight_layout()
# fig.savefig("product-potential.png")
