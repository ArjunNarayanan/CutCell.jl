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
    interfacecondition = CutCell.incoherent_interface_condition(
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
    CutCell.assemble_incoherent_interface_transformation_rhs!(
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

K1, K2 = 247.0e9, 192.0e9    # Pa
mu1, mu2 = 126.0e9, 87.0e9   # Pa
lambda1 = lame_lambda(K1, mu1)
lambda2 = lame_lambda(K2, mu2)
stiffness = CutCell.HookeStiffness(lambda1, mu1, lambda2, mu2)

rho1 = 3.93e3   # Kg/m^3
rho2 = 3.68e3   # Kg/m^3

V1 = inv(rho1)
V2 = inv(rho2)
theta0 = -0.067
ΔG = -1022.0    # J/mol
molarmass = 0.147   # Kg/mol
diffG0 = ΔG/molarmass # J/Kg

width = 1.0e-3
penaltyfactor = 1e2

polyorder = 3
numqp = required_quadrature_order(polyorder) + 2
nelmts = 33
center = [width / 2, width / 2]
inradius = width / 4
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

relspatialpoints = spclosestpoints .- center
angularposition = angular_position(relspatialpoints)
sortidx = sortperm(angularposition)
angularposition = angularposition[sortidx]

refclosestpoints = refclosestpoints[:,sortidx]
refclosestcellids = refclosestcellids[sortidx]
refgradients = refgradients[:,sortidx]

invjac = CutCell.inverse_jacobian(cutmesh)
normals = diagm(invjac) * refgradients
CutCell.normalize_normals!(normals)





product_stress_at_cp = CutCell.product_stress_at_reference_points(
    refclosestpoints,
    refclosestcellids,
    basis,
    stiffness,
    transfstress,
    theta0,
    nodaldisplacement,
    cutmesh,
)
parent_stress_at_cp = CutCell.parent_stress_at_reference_points(
    refclosestpoints,
    refclosestcellids,
    basis,
    stiffness,
    nodaldisplacement,
    cutmesh,
)


# productpotential = interface_potential(V1,K1,mu1,productpressure,productdevstress,normals)
# parentpotential = interface_potential(V2,K2,mu2,parentpressure,parentdevstress,normals)


productpressure = CutCell.pressure_at_points(product_stress_at_cp)
parentpressure = CutCell.pressure_at_points(parent_stress_at_cp)

productdevstress = CutCell.deviatoric_stress_at_points(product_stress_at_cp,productpressure)
parentdevstress = CutCell.deviatoric_stress_at_points(parent_stress_at_cp,parentpressure)

productdevnorm = stress_inner_product_over_points(productdevstress)
parentdevnorm = stress_inner_product_over_points(parentdevstress)

productnormaldevstress = normal_stress_component_over_points(productdevstress,normals)
parentnormaldevstress = normal_stress_component_over_points(parentdevstress,normals)

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


fig,ax = PyPlot.subplots()
ax.plot(angularposition,potentialdifference)
ax.grid()
ax.set_xlabel("Angular position (deg)")
ax.set_ylabel("Potential difference (J/Kg)")
ax.set_title("Potential difference along interface circumference")
fig.tight_layout()
fig.savefig("potential-difference.png")


fig,ax = PyPlot.subplots()
ax.plot(angularposition,parentp1,label="term1")
ax.plot(angularposition,parentp2,label="term2")
ax.plot(angularposition,parentp3,label="term3")
ax.plot(angularposition,parentp4,label="term4")
ax.plot(angularposition,parentpotential,label="potential")
ax.set_xlabel("Angular position (deg)")
ax.set_ylabel("Energy density (J/Kg)")
ax.set_title("Potential on parent side of interface")
ax.set_ylim(-1e6,2e6)
ax.grid()
ax.legend()
fig.tight_layout()
fig.savefig("parent-potential.png")



fig,ax = PyPlot.subplots()
ax.plot(angularposition,productp1,label="term1")
ax.plot(angularposition,productp2,label="term2")
ax.plot(angularposition,productp3,label="term3")
ax.plot(angularposition,productp4,label="term4")
ax.plot(angularposition,productpotential,label="potential")
ax.set_xlabel("Angular position (deg)")
ax.set_ylabel("Energy density (J/Kg)")
ax.set_ylim(-1e6,2e6)
ax.set_title("Potential on product side of interface")
ax.grid()
ax.legend()
fig.tight_layout()
fig.savefig("product-potential.png")
