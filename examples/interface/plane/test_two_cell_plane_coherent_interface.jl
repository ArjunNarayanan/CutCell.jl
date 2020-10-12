using CSV, DataFrames
using IntervalArithmetic
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell

function plane_distance_function(coords, normal, x0)
    return (coords .- x0)' * normal
end

function circle_distance_function(coords, center, radius)
    difference = coords .- center
    distance = [radius - norm(difference[:, i]) for i = 1:size(difference)[2]]
    return distance
end

function exact_solution(lambda, mu, e11, x)
    e22 = -lambda / (lambda + 2mu) * e11
    u1 = e11 * x[1]
    u2 = e22 * x[2]
    return [u1, u2]
end

function add_cell_error_squared!(
    err,
    interpolater,
    exactsolution,
    cellmap,
    quad,
    detjac,
)

    for (p, w) in quad
        numsol = interpolater(p)
        exactsol = exactsolution(cellmap(p))
        err .+= (numsol - exactsol) .^ 2 * detjac * w
    end
end

function add_exact_norm_squared(err, exactsolution, cellmap, quad, detjac)
    for (p, w) in quad
        err .+= (exactsolution(cellmap(p))) .^ 2 * detjac * w
    end
end

function integrate_exact_norm(exactsolution, cutmesh, cellquads)
    err = zeros(2)
    cellsign = CutCell.cell_sign(cutmesh)
    detjac = CutCell.determinant_jacobian(CutCell.cell_map(cutmesh, 1))

    for (cellid, s) in enumerate(cellsign)
        cellmap = CutCell.cell_map(cutmesh, cellid)
        if s == +1 || s == 0
            quad = cellquads[+1, cellid]
            add_exact_norm_squared(err, exactsolution, cellmap, quad, detjac)
        elseif s == -1 || s == 0
            quad = cellquads[-1, cellid]
            add_exact_norm_squared(err, exactsolution, cellmap, quad, detjac)
        else
            error("Got unexpected cellsign = $s")
        end
    end
    return sqrt.(err)
end

function mesh_L2_error(nodalsolutions, exactsolution, cutmesh, basis, cellquads)

    ndofs = size(nodalsolutions)[1]
    detjac = CutCell.determinant_jacobian(CutCell.cell_map(cutmesh, 1))
    err = zeros(ndofs)
    interpolater = InterpolatingPolynomial(ndofs, basis)
    cellsign = CutCell.cell_sign(cutmesh)

    for (cellid, s) in enumerate(cellsign)
        @assert s == +1 || s == 0 || s == -1
        cellmap = CutCell.cell_map(cutmesh, cellid)
        if s == +1 || s == 0
            quad = cellquads[+1, cellid]
            nodeids = CutCell.nodal_connectivity(cutmesh, +1, cellid)
            update!(interpolater, nodalsolutions[:, nodeids])
            add_cell_error_squared!(
                err,
                interpolater,
                exactsolution,
                cellmap,
                quad,
                detjac,
            )
        end
        if s == -1 || s == 0
            quad = cellquads[-1, cellid]
            nodeids = CutCell.nodal_connectivity(cutmesh, -1, cellid)
            update!(interpolater, nodalsolutions[:, nodeids])
            add_cell_error_squared!(
                err,
                interplater,
                exactsolution,
                cellmap,
                quad,
                detjac,
            )
        end
    end
    return sqrt.(err)
end

function apply_two_cell_displacement_bc!(matrix,rhs,cutmesh,nf,dx)
    nfside = round(Int,sqrt(nf))

    leftnodeids = CutCell.nodal_connectivity(cutmesh,-1,1)[1:nfside]
    CutCell.apply_dirichlet_bc!(matrix,rhs,leftnodeids,1,0.0,2)
    CutCell.apply_dirichlet_bc!(matrix,rhs,[leftnodeids[1]],2,0.0,2)

    rightnodeids = CutCell.nodal_connectivity(cutmesh,+1,2)[(nf-nfside+1):nf]
    CutCell.apply_dirichlet_bc!(matrix,rhs,rightnodeids,1,dx,2)
end

function condition_number(normal, x0, polyorder, numqp, stiffnesses, penalty)
    basis = TensorProductBasis(2, polyorder)
    levelset = InterpolatingPolynomial(1, basis)
    nf = CutCell.number_of_basis_functions(basis)

    mesh = CutCell.Mesh([0.0, 0.0], [4.0, 1.0], [2, 1], nf)
    nodalcoordinates = CutCell.nodal_coordinates(mesh)
    levelsetcoeffs = plane_distance_function(nodalcoordinates, normal, x0)

    cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
    cellquads =
        CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
    interfacequads = CutCell.CutMeshInterfaceQuadratures(
        levelset,
        levelsetcoeffs,
        cutmesh,
        numqp,
    )

    bilinearforms =
        CutCell.CutMeshBilinearForms(basis, cellquads, stiffnesses, cutmesh)
    interfaceconstraints = CutCell.coherent_constraint_on_cells(
        basis,
        interfacequads,
        cutmesh,
        penalty,
    )

    sysmatrix = CutCell.SystemMatrix()
    sysrhs = CutCell.SystemRHS()

    CutCell.assemble_bilinear_form!(sysmatrix, bilinearforms, cutmesh, 2)
    CutCell.assemble_interface_constraints!(
        sysmatrix,
        interfaceconstraints,
        cutmesh,
        2,
    )

    totalndofs = 2 * CutCell.total_number_of_nodes(cutmesh)
    matrix = CutCell.sparse(sysmatrix, totalndofs)
    rhs = CutCell.rhs(sysrhs, totalndofs)

    CutCell.apply_dirichlet_bc!(matrix, rhs, [7, 7], [1, 2], [0.0, 0.0], 2)
    CutCell.apply_dirichlet_bc!(matrix, rhs, [8], [1], [0.0], 2)
    CutCell.apply_dirichlet_bc!(matrix, rhs, [5, 6], [1, 1], [0.1, 0.1], 2)

    return cond(Array(matrix))
end

function solve_two_cell_plane_interface(
    normal,
    x0,
    polyorder,
    numqp,
    stiffnesses,
    penalty,
    exactsolution,
)
    basis = TensorProductBasis(2, polyorder)
    levelset = InterpolatingPolynomial(1, basis)
    nf = CutCell.number_of_basis_functions(basis)

    mesh = CutCell.Mesh([0.0, 0.0], [4.0, 1.0], [2, 1], nf)
    nodalcoordinates = CutCell.nodal_coordinates(mesh)
    levelsetcoeffs = plane_distance_function(nodalcoordinates, normal, x0)

    cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
    cellquads =
        CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
    interfacequads = CutCell.CutMeshInterfaceQuadratures(
        levelset,
        levelsetcoeffs,
        cutmesh,
        numqp,
    )

    bilinearforms =
        CutCell.BilinearForms(basis, cellquads, stiffnesses, cutmesh)

    
    sysmatrix = CutCell.SystemMatrix()
    sysrhs = CutCell.SystemRHS()

    CutCell.assemble_bilinear_form!(sysmatrix, bilinearforms, cutmesh, 2)
    CutCell.assemble_interface_constraints!(
        sysmatrix,
        interfaceconstraints,
        cutmesh,
        2,
    )

    totalndofs = 2 * CutCell.total_number_of_nodes(cutmesh)
    matrix = CutCell.sparse(sysmatrix, totalndofs)
    rhs = CutCell.rhs(sysrhs, totalndofs)

    apply_two_cell_displacement_bc!(matrix,rhs,cutmesh,nf,dx)

    sol = matrix \ rhs
    nodalsolutions = reshape(sol, 2, :)

    errorcellquads = CutCell.CellQuadratures(
        levelset,
        levelsetcoeffs,
        cutmesh,
        numqp + 2,
    )

    err = mesh_L2_error(
        nodalsolutions,
        exactsolution,
        cutmesh,
        basis,
        errorcellquads,
    )

    normalizer = integrate_exact_norm(exactsolution, cutmesh, errorcellquads)

    return err ./ normalizer
end

function normal_from_angle(theta)
    return [cosd(theta), sind(theta)]
end


polyorder = 1
numqp = 4
lambda, mu = (1.0, 2.0)
stiffness = CutCell.plane_strain_voigt_hooke_matrix(lambda, mu)
stiffnesses = [stiffness, stiffness]
dx = 0.1
e11 = dx / 4.0
