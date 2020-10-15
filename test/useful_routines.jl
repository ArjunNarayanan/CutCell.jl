using ImplicitDomainQuadrature
import Base.==, Base.≈

function allapprox(v1, v2)
    return all(v1 .≈ v2)
end

function allapprox(v1, v2, atol)
    return length(v1) == length(v2) &&
           all([isapprox(v1[i], v2[i], atol = atol) for i = 1:length(v1)])
end

function allequal(v1, v2)
    return all(v1 .== v2)
end

function Base.isequal(c1::CutCell.CellMap, c2::CutCell.CellMap)
    return allequal(c1.yL, c2.yL) && allequal(c1.yR, c2.yR)
end

function ==(c1::CutCell.CellMap, c2::CutCell.CellMap)
    return isequal(c1, c2)
end

function ≈(q1::QuadratureRule, q2::QuadratureRule)
    flag = allapprox(q1.points, q2.points) && allapprox(q1.weights, q2.weights)
end

function plane_distance_function(coords, normal, x0)
    return (coords .- x0)' * normal
end

function plane_distance_function(coords, normal, x0)
    return (coords .- x0)' * normal
end

function circle_distance_function(coords, center, radius)
    difference = (coords .- center) .^ 2
    distance = radius .- sqrt.(mapslices(sum, difference, dims = 1)')
    return distance
end

function normal_from_angle(theta)
    return [cosd(theta), sind(theta)]
end

function add_cell_error_squared!(err, interpolater, exactsolution, cellmap, quad)
    detjac = CutCell.determinant_jacobian(cellmap)
    for (p, w) in quad
        numsol = interpolater(p)
        exsol = exactsolution(cellmap(p))
        err .+= (numsol - exsol) .^ 2 * detjac * w
    end
end

function add_cell_norm_squared!(vals, func, cellmap, quad)
    detjac = CutCell.determinant_jacobian(cellmap)
    for (p, w) in quad
        v = func(cellmap(p))
        vals .+= v .^ 2 * detjac * w
    end
end

function integral_norm_on_cut_mesh(func, cellquads, cutmesh, ndofs)
    vals = zeros(ndofs)
    ncells = CutCell.number_of_cells(cutmesh)
    for cellid = 1:ncells
        s = CutCell.cell_sign(cutmesh, cellid)
        cellmap = CutCell.cell_map(cutmesh, cellid)
        @assert s == -1 || s == 0 || s == 1
        if s == 1 || s == 0
            pquad = cellquads[1, cellid]
            add_cell_norm_squared!(vals, func, cellmap, pquad)
        end
        if s == -1 || s == 0
            nquad = cellquads[-1, cellid]
            add_cell_norm_squared!(vals, func, cellmap, nquad)
        end
    end
    return sqrt.(vals)
end

function mesh_L2_error(nodalsolutions, exactsolution, basis, cellquads, cutmesh)
    err = zeros(2)
    interpolater = InterpolatingPolynomial(2, basis)
    ncells = CutCell.number_of_cells(cutmesh)
    for cellid = 1:ncells
        s = CutCell.cell_sign(cutmesh, cellid)
        cellmap = CutCell.cell_map(cutmesh, cellid)
        @assert s == -1 || s == 0 || s == 1
        if s == 1 || s == 0
            nodeids = CutCell.nodal_connectivity(cutmesh, 1, cellid)
            elementsolution = nodalsolutions[:, nodeids]
            update!(interpolater, elementsolution)
            quad = cellquads[1, cellid]
            add_cell_error_squared!(err, interpolater, exactsolution, cellmap, quad)
        end
        if s == -1 || s == 0
            nodeids = CutCell.nodal_connectivity(cutmesh, -1, cellid)
            elementsolution = nodalsolutions[:, nodeids]
            update!(interpolater, elementsolution)
            quad = cellquads[-1, cellid]
            add_cell_error_squared!(err, interpolater, exactsolution, cellmap, quad)
        end
    end
    return sqrt.(err)
end

function required_quadrature_order(polyorder)
    ceil(Int, 0.5 * (2polyorder + 1))
end
