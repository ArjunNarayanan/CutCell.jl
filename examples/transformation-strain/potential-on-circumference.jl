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

function update_symmdispgrad!(totalstrain, basis, celldisp, points, jac, vectosymmconverter)

    nump = size(points)[2]
    dim = CutCell.dimension(basis)
    for i = 1:nump
        grad = CutCell.transform_gradient(gradient(basis, points[:, i]), jac)
        NK = sum([CutCell.make_row_matrix(vectosymmconverter[k], grad[:, k]) for k = 1:dim])
        symmdispgrad = NK * celldisp
        append!(totalstrain, symmdispgrad)
    end
end

function symmetric_displacement_gradient(displacement, basis, interfacequads, cutmesh, s)

    cellsign = CutCell.cell_sign(cutmesh)
    cellids = findall(cellsign .== 0)

    totalstrain = zeros(0)

    dim = CutCell.dimension(basis)
    jac = CutCell.jacobian(cutmesh)
    vectosymmconverter = CutCell.vector_to_symmetric_matrix_converter()

    for cellid in cellids
        nodeids = CutCell.nodal_connectivity(cutmesh, s, cellid)
        celldofs = CutCell.element_dofs(nodeids, dim)
        celldisp = displacement[celldofs]
        points = interfacequads[s, cellid].points

        update_symmdispgrad!(totalstrain, basis, celldisp, points, jac, vectosymmconverter)
    end
    return reshape(totalstrain, 3, :)
end

function interface_quadrature_points(interfacequads, cutmesh)
    cellsign = CutCell.cell_sign(cutmesh)
    cellids = findall(cellsign .== 0)
    numcellqps = length(interfacequads.quads[1])
    numqps = numcellqps * length(cellids)
    points = zeros(2, numqps)
    counter = 1
    for cellid in cellids
        cellmap = CutCell.cell_map(cutmesh, cellid)
        qp = cellmap(interfacequads[1, cellid].points)
        points[:, counter:(counter+numcellqps-1)] .= qp
        counter += numcellqps
    end
    return points
end

function angular_position(points)
    cpoints = points[1, :] + im * points[2, :]
    return rad2deg.(angle.(cpoints))
end

function parent_stress(symmdispgrad, stiffness)
    lambda, mu = CutCell.lame_coefficients(stiffness, -1)
    inplanestress = stiffness[-1] * symmdispgrad
    s33 = lambda * (symmdispgrad[1, :] + symmdispgrad[2, :])
    return vcat(inplanestress, s33')
end

function product_stress(elasticsymmdispgrad, stiffness, theta0)
    lambda, mu = CutCell.lame_coefficients(stiffness, +1)
    inplanestress = stiffness[+1] * elasticsymmdispgrad
    s33 = lambda * (elasticsymmdispgrad[1, :] + elasticsymmdispgrad[2, :]) .- (lambda + 2mu) * theta0 / 3
    return vcat(inplanestress,s33')
end

function parent_strain_energy(symmdispgrad,stress)
    return 0.5*sum([symmdispgrad[i,:] .* stress[i,:] for i = 1:3])
end

function product_strain_energy(elasticsymmdispgrad,stress,theta0)
    s1 = sum([elasticsymmdispgrad[i,:] .* stress[i,:] for i = 1:3])
    s2 = -theta0/3*stress[4,:]
    return 0.5*(s1+s2)
end

function pressure(stress)
    return -(stress[1,:] + stress[2,:] + stress[4,:])
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
penaltyfactor = 1e4

polyorder = 3
numqp = required_quadrature_order(polyorder) + 4
nelmts = 21
center = [width / 2, width / 2]
inradius = width / 4
outradius = width


lambda1, mu1 = CutCell.lame_coefficients(stiffness, +1)
lambda2, mu2 = CutCell.lame_coefficients(stiffness, -1)
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
interfacecondition =
    CutCell.InterfaceCondition(basis, interfacequads, stiffness, cutmesh, penalty)

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


matrix = CutCell.make_sparse(sysmatrix, cutmesh)
rhs = CutCell.rhs(sysrhs, cutmesh)

topleftnodeid = CutCell.nodes_per_mesh_side(mesh)[2]
CutCell.apply_dirichlet_bc!(matrix, rhs, [1, topleftnodeid], 1, 0.0, 2)
CutCell.apply_dirichlet_bc!(matrix, rhs, [1], 2, 0.0, 2)

sol = matrix \ rhs



quadpoints = interface_quadrature_points(interfacequads,cutmesh)
relquadpoints = quadpoints .- center
angularposition = angular_position(relquadpoints)
sortidx = sortperm(angularposition)
angularposition = angularposition[idx]


parentsymmdispgrad  = symmetric_displacement_gradient(sol, basis, interfacequads, cutmesh, -1)[:,sortidx]
productsymmdispgrad = symmetric_displacement_gradient(sol, basis, interfacequads, cutmesh, +1)[:,sortidx]
productelasticsymmdispgrad = productsymmdispgrad .- transfstrain


parentstress = parent_stress(parentsymmdispgrad, stiffness)
productstress = product_stress(productelasticsymmdispgrad,stiffness,theta0)

parentstrainenergy = parent_strain_energy(parentsymmdispgrad,parentstress)
productstrainenergy = product_strain_energy(productelasticsymmdispgrad,productstress,theta0)

parentpressure = pressure(parentstress)
productpressure = pressure(productstress)

using PyPlot
fig,ax = PyPlot.subplots()
ax.plot(angularposition,productpressure)
# ax.set_ylim(0.15,0.2)
ax.grid()
fig
