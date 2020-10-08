using CSV, DataFrames
using LinearAlgebra
using PolynomialBasis
using ImplicitDomainQuadrature
include("../plot_utils.jl")
using Revise
using CutCell

function plane_distance_function(coords, normal, x0)
    return (coords .- x0)' * normal
end

function circle_distance_function(coords, center, radius)
    diff2 = (coords .- center) .^ 2
    distance = radius .- sqrt.(mapslices(sum, diff2, dims = 1)')
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
                interpolater,
                exactsolution,
                cellmap,
                quad,
                detjac,
            )
        end
    end
    return sqrt.(err)
end

function left_boundary_nodeids(cutmesh, nf)
    nfside = round(Int, sqrt(nf))
    return CutCell.nodal_connectivity(cutmesh, -1, 1)[1:nfside]
end

function right_boundary_nodeids(cutmesh, nf)
    nfside = round(Int, sqrt(nf))
    return CutCell.nodal_connectivity(cutmesh, +1, 2)[(nf-nfside+1):nf]
end

function apply_two_cell_displacement_bc!(matrix, rhs, cutmesh, nf, dx)

    leftnodeids = left_boundary_nodeids(cutmesh, nf)
    CutCell.apply_dirichlet_bc!(matrix, rhs, leftnodeids, 1, 0.0, 2)
    CutCell.apply_dirichlet_bc!(matrix, rhs, [leftnodeids[1]], 2, 0.0, 2)

    rightnodeids = right_boundary_nodeids(cutmesh, nf)
    CutCell.apply_dirichlet_bc!(matrix, rhs, rightnodeids, 1, dx, 2)
end

function solve_two_cell_curved_interface(
    center,
    radius,
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
    levelsetcoeffs = circle_distance_function(nodalcoordinates, center, radius)

    cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
    cellquads =
        CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
    interfacequads =
        CutCell.InterfaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)

    bilinearforms =
        CutCell.BilinearForms(basis, cellquads, stiffnesses, cutmesh)
    interfaceconstraints = CutCell.coherent_interface_constraint(
        basis,
        interfacequads,
        cutmesh,
        penalty,
    )

    sysmatrix = CutCell.SystemMatrix()
    sysrhs = CutCell.SystemRHS()

    CutCell.assemble_bilinear_form!(sysmatrix, bilinearforms, cutmesh)
    CutCell.assemble_interface_constraints!(
        sysmatrix,
        interfaceconstraints,
        cutmesh,
    )

    totalndofs = CutCell.number_of_degrees_of_freedom(cutmesh)
    matrix = CutCell.stiffness(sysmatrix, totalndofs)
    rhs = CutCell.rhs(sysrhs, totalndofs)

    apply_two_cell_displacement_bc!(matrix, rhs, cutmesh, nf, 0.1)

    sol = matrix \ rhs
    nodalsolutions = reshape(sol, 2, :)

    errorcellquads =
        CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp + 2)

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

function error_for_penalty_parameter(
    center,
    radius,
    polyorder,
    numqp,
    stiffnesses,
    penalties,
    exactsolution,
)

    err = zeros(length(penalties), 2)

    for (idx, penalty) in enumerate(penalties)
        err[idx, :] = solve_two_cell_curved_interface(
            center,
            radius,
            polyorder,
            numqp,
            stiffnesses,
            penalty,
            exactsolution,
        )
    end

    filename = "examples/interface/curved/error-vs-penalty.csv"
    df =
        DataFrame(penalty = penalties, ErrorU1 = err[:, 1], ErrorU2 = err[:, 2])
    CSV.write(filename, df)
end


polyorder = 2
numqp = 4
penalty = 1e4
lambda, mu = (1.0, 2.0)
stiffness = CutCell.plane_strain_voigt_hooke_matrix(lambda, mu)
stiffnesses = [stiffness, stiffness]
dx = 0.1
e11 = dx / 4.0

xc = 3.0
center = [xc, 0.5]
radius = xc - 1.0
penalties = [1,1e1,1e2,1e3,1e4,1e5]

error_for_penalty_parameter(
    center,
    radius,
    polyorder,
    numqp,
    stiffnesses,
    penalties,
    x -> exact_solution(lambda, mu, e11, x),
)

# err = solve_two_cell_curved_interface(
#     center,
#     radius,
#     polyorder,
#     numqp,
#     stiffnesses,
#     penalty,
#     x -> exact_solution(lambda, mu, e11, x),
# )

# plot_interface_quadrature_points(interfacequads,cutmesh,(0,4),(0,1))
# plot_cell_quadrature_points(cellquads,cutmesh,+1,(0,4),(0,1))
