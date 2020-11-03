using Triangulate
using WriteVTK
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("../../test/useful_routines.jl")

function compute_at_cell_quadrature_points!(vals, interpolater, quad)
    for (p, w) in quad
        append!(vals, interpolater(p))
    end
end

function compute_field_at_quadrature_points(nodalvals, basis, cellquads, cutmesh)
    ndofs, numnodes = size(nodalvals)
    ncells = CutCell.number_of_cells(cutmesh)
    interpolater = InterpolatingPolynomial(ndofs, basis)
    quadvals = zeros(0)

    for cellid = 1:ncells
        s = CutCell.cell_sign(cutmesh, cellid)
        @assert s == -1 || s == 0 || s == +1
        if s == +1 || s == 0
            nodeids = CutCell.nodal_connectivity(cutmesh, +1, cellid)
            cellvals = nodalvals[:, nodeids]
            update!(interpolater, cellvals)
            quad = cellquads[+1, cellid]
            compute_at_cell_quadrature_points!(quadvals, interpolater, quad)
        end
        if s == -1 || s == 0
            nodeids = CutCell.nodal_connectivity(cutmesh, -1, cellid)
            cellvals = nodalvals[:, nodeids]
            update!(interpolater, cellvals)
            quad = cellquads[-1, cellid]
            compute_at_cell_quadrature_points!(quadvals, interpolater, quad)
        end
    end
    return reshape(quadvals, ndofs, :)
end

function compute_quadrature_points(cellquads, cutmesh)
    ncells = CutCell.number_of_cells(cutmesh)
    coords = zeros(2, 0)
    for cellid = 1:ncells
        s = CutCell.cell_sign(cutmesh, cellid)
        cellmap = CutCell.cell_map(cutmesh, cellid)
        @assert s == -1 || s == 0 || s == +1
        if s == +1 || s == 0
            coords = hcat(coords, cellmap(cellquads[+1, cellid].points))
        end
        if s == -1 || s == 0
            coords = hcat(coords, cellmap(cellquads[-1, cellid].points))
        end
    end
    return coords
end

function add_cell_jump_squared!(err, interp1, quad1, interp2, quad2, scalearea)
    numqp = length(quad1)
    @assert length(quad2) == length(scalearea)
    for qpidx = 1:numqp
        p1, w1 = quad1[qpidx]
        p2, w2 = quad2[qpidx]
        @assert w1 ≈ w2

        err .+= (interp1(p1) - interp2(p2)) .^ 2 * scalearea[qpidx] * w1
    end
end

function displacement_jump_error(nodaldisp, basis, interfacequads, mesh)

    dim, nnodes = size(nodaldisp)
    err = zeros(dim)
    cellmap = CutCell.cell_map(mesh)

    pinterp = InterpolatingPolynomial(dim, basis)
    ninterp = InterpolatingPolynomial(dim, basis)

    cellsign = CutCell.cell_sign(mesh)
    cellids = findall(cellsign .== 0)

    for cellid in cellids
        pquad = interfacequads[+1, cellid]
        nquad = interfacequads[-1, cellid]
        normals = CutCell.interface_normals(interfacequads, cellid)
        scalearea = CutCell.scale_area(cellmap, normals)

        pnodeids = CutCell.nodal_connectivity(mesh, +1, cellid)
        pdisp = nodaldisp[:, pnodeids]
        update!(pinterp, pdisp)

        nnodeids = CutCell.nodal_connectivity(mesh, -1, cellid)
        ndisp = nodaldisp[:, nnodeids]
        update!(ninterp, ndisp)

        add_cell_jump_squared!(err, pinterp, pquad, ninterp, nquad, scalearea)
    end
    return sqrt.(err)
end

function add_cell_traction_jump_squared!(
    err,
    interp1,
    quad1,
    interp2,
    quad2,
    normals,
    scalearea,
    vectosymmconverter,
)

    numqp = length(quad1)
    @assert length(quad2) == length(scalearea)
    @assert size(normals)[2] == numqp
    dim = length(vectosymmconverter)
    for qpidx = 1:numqp
        p1, w1 = quad1[qpidx]
        p2, w2 = quad2[qpidx]
        @assert w1 ≈ w2

        normal = normals[:, qpidx]
        NK = sum([normal[k] * vectosymmconverter[k]' for k = 1:dim])

        err .+= (NK * (interp1(p1) - interp2(p2))) .^ 2 * scalearea[qpidx] * w1
    end
end

function add_cell_mean_traction_squared!(
    err,
    interp1,
    quad1,
    interp2,
    quad2,
    normals,
    scalearea,
    vectosymmconverter,
)

    numqp = length(quad1)
    @assert length(quad2) == length(scalearea)
    @assert size(normals)[2] == numqp
    dim = length(vectosymmconverter)
    for qpidx = 1:numqp
        p1, w1 = quad1[qpidx]
        p2, w2 = quad2[qpidx]
        @assert w1 ≈ w2

        normal = normals[:, qpidx]
        NK = sum([normal[k] * vectosymmconverter[k]' for k = 1:dim])

        err .+= (NK * (interp1(p1) + interp2(p2))) .^ 2 * scalearea[qpidx] * w1
    end
end

function traction_jump_error(nodalstress, basis, interfacequads, mesh)
    dim = CutCell.dimension(mesh)
    sdim, nnodes = size(nodalstress)
    err = zeros(dim)
    normalizer = zeros(dim)
    cellmap = CutCell.cell_map(mesh)

    pinterp = InterpolatingPolynomial(sdim, basis)
    ninterp = InterpolatingPolynomial(sdim, basis)

    cellsign = CutCell.cell_sign(mesh)
    cellids = findall(cellsign .== 0)

    vectosymmconverter = CutCell.vector_to_symmetric_matrix_converter()

    for cellid in cellids
        pquad = interfacequads[+1, cellid]
        nquad = interfacequads[-1, cellid]
        normals = CutCell.interface_normals(interfacequads, cellid)
        scalearea = CutCell.scale_area(cellmap, normals)

        pnodeids = CutCell.nodal_connectivity(mesh, +1, cellid)
        pstress = nodalstress[:, pnodeids]
        update!(pinterp, pstress)

        nnodeids = CutCell.nodal_connectivity(mesh, -1, cellid)
        nstress = nodalstress[:, nnodeids]
        update!(ninterp, nstress)

        add_cell_traction_jump_squared!(
            err,
            pinterp,
            pquad,
            ninterp,
            nquad,
            normals,
            scalearea,
            vectosymmconverter,
        )
        add_cell_mean_traction_squared!(
            normalizer,
            pinterp,
            pquad,
            ninterp,
            nquad,
            normals,
            scalearea,
            vectosymmconverter,
        )
    end
    return sqrt.(err) ./ sqrt.(normalizer)
end

function bulk_modulus(lambda, mu)
    return lambda + 2mu / 3
end

function analytical_coefficients_matrix(R, ls, ms, lc, mc)
    a = zeros(2, 2)
    a[1, 1] = 1
    a[1, 2] = -1 / R^2
    a[2, 1] = 2(lc + mc)
    a[2, 2] = 2ms / R^2
    return a
end

function analytical_coefficients_rhs(R, ls, ms, lc, mc, theta0)
    Ks = bulk_modulus(ls, ms)
    r = zeros(2)
    r[1] = Ks * theta0 / (2 * (ls + ms))
    r[2] = 0.0
    return r
end

function analytical_coefficients(R, ls, ms, lc, mc, theta0)
    a = analytical_coefficients_matrix(R, ls, ms, lc, mc)
    r = analytical_coefficients_rhs(R, ls, ms, lc, mc, theta0)
    return a \ r
end

function normal_stress_in_core(R, ls, ms, lc, mc, theta0)
    coeffs = analytical_coefficients(R, ls, ms, lc, mc, theta0)
    return 2 * (lc + mc) * coeffs[1]
end

lambda1, mu1 = 100.0, 80.0
lambda2, mu2 = 80.0, 60.0
theta0 = -0.067
stiffness = CutCell.HookeStiffness(lambda1, mu1, lambda2, mu2)
transfstress = CutCell.plane_strain_transformation_stress(lambda1, mu1, theta0)

L = 1.0
penaltyfactor = 1e2
nelmts = 23
dx = L / nelmts
penalty = penaltyfactor / dx * (lambda1 + mu1)

polyorder = 2
numqp = required_quadrature_order(polyorder) + 2

center = [L / 2, L / 2]
radius = L / 20
basis = TensorProductBasis(2, polyorder)
mesh = CutCell.Mesh([0.0, 0.0], [L, L], [nelmts, nelmts], basis)
levelset = InterpolatingPolynomial(1, basis)
levelsetcoeffs =
    CutCell.levelset_coefficients(x -> -circle_distance_function(x, center, radius), mesh)

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
disp = reshape(sol, 2, :)


sysmatrix = CutCell.SystemMatrix()
sysrhs = CutCell.SystemRHS()

CutCell.assemble_stress_mass_matrix!(sysmatrix, basis, cellquads, cutmesh)
CutCell.assemble_stress_linear_form!(sysrhs, basis, cellquads, stiffness, sol, cutmesh)
CutCell.assemble_transformation_stress_linear_form!(
    sysrhs,
    transfstress,
    basis,
    cellquads,
    cutmesh,
)

matrix = CutCell.make_sparse_stress_operator(sysmatrix, cutmesh)
rhs = CutCell.stress_rhs(sysrhs, cutmesh)

stressvec = matrix \ rhs
stress = reshape(stressvec, 3, :)

quadcoords = compute_quadrature_points(cellquads, cutmesh)
dispquadvals = compute_field_at_quadrature_points(disp, basis, cellquads, cutmesh)
stressquadvals = compute_field_at_quadrature_points(stress, basis, cellquads, cutmesh)

triin = Triangulate.TriangulateIO()
triin.pointlist = quadcoords
(triout, vorout) = triangulate("", triin)
connectivity = triout.trianglelist
cells = [
    MeshCell(VTKCellTypes.VTK_TRIANGLE, connectivity[:, i]) for i = 1:size(connectivity)[2]
]
vtkfile = vtk_grid(
    "examples/transformation-strain/circle-inclusion-R20",
    quadcoords[1, :],
    quadcoords[2, :],
    cells,
)
vtkfile["displacement"] = (quadvals[1, :], quadvals[2, :])
vtkfile["s11"] = stressquadvals[1, :]
vtkfile["s22"] = stressquadvals[2, :]
vtkfile["s12"] = stressquadvals[3, :]
outfiles = vtk_save(vtkfile)


core_s11 = normal_stress_in_core(radius,lambda1,mu1,lambda2,mu2,theta0)
# tractionjumperr = traction_jump_error(stress, basis, interfacequads, cutmesh)

# quadvals = compute_field_at_quadrature_points(stress, basis, cellquads, cutmesh)
# quadcoords = compute_quadrature_points(cellquads, cutmesh)
# pressure = -(quadvals[1, :] + quadvals[2, :])

# fig,ax = PyPlot.subplots()
# cmap = ax.tricontourf(quadcoords[1,:],quadcoords[2,:],quadvals[1,:])
# ax.set_aspect("equal")
# fig.colorbar(cmap)
# fig
