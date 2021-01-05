function update_product_stress!(
    qpstress,
    basis,
    stiffness,
    transfstress,
    theta0,
    celldisp,
    points,
    jac,
    vectosymmconverter,
)
    dim = dimension(basis)
    lambda, mu = lame_coefficients(stiffness, +1)
    nump = size(points)[2]

    for qpidx = 1:nump
        p = points[:, qpidx]
        grad = transform_gradient(gradient(basis, p), jac)
        NK = sum([make_row_matrix(vectosymmconverter[k], grad[:, k]) for k = 1:dim])
        symmdispgrad = NK * celldisp

        inplanestress = (stiffness[+1] * symmdispgrad) - transfstress
        s33 = lambda * (symmdispgrad[1] + symmdispgrad[2]) - (lambda + 2mu / 3) * theta0

        numericalstress = vcat(inplanestress, s33)

        append!(qpstress, numericalstress)
    end
end

function update_parent_stress!(
    qpstress,
    basis,
    stiffness,
    celldisp,
    points,
    jac,
    vectosymmconverter,
)

    dim = dimension(basis)
    lambda, mu = lame_coefficients(stiffness, -1)
    nump = size(points)[2]
    for qpidx = 1:nump
        p = points[:, qpidx]
        grad = transform_gradient(gradient(basis, p), jac)
        NK = sum([make_row_matrix(vectosymmconverter[k], grad[:, k]) for k = 1:dim])
        symmdispgrad = NK * celldisp

        inplanestress = stiffness[-1] * symmdispgrad
        s33 = lambda * (symmdispgrad[1] + symmdispgrad[2])

        numericalstress = vcat(inplanestress, s33)

        append!(qpstress, numericalstress)
    end
end

function compute_stress_at_cell_quadrature_points(
    nodaldisplacement,
    basis,
    stiffness,
    transfstress,
    theta0,
    cellquads,
    cutmesh,
)

    dim = dimension(basis)
    ncells = number_of_cells(cutmesh)
    qpstress = zeros(0)
    jac = jacobian(cutmesh)
    vectosymmconverter = vector_to_symmetric_matrix_converter()

    for cellid = 1:ncells
        s = cell_sign(cutmesh, cellid)
        @assert s == -1 || s == 0 || s == +1
        if s == +1 || s == 0
            nodeids = nodal_connectivity(cutmesh, +1, cellid)
            celldofs = element_dofs(nodeids, dim)
            celldisp = nodaldisplacement[celldofs]
            points = cellquads[+1, cellid].points
            update_product_stress!(
                qpstress,
                basis,
                stiffness,
                transfstress,
                theta0,
                celldisp,
                points,
                jac,
                vectosymmconverter,
            )
        end
        if s == -1 || s == 0
            nodeids = nodal_connectivity(cutmesh, -1, cellid)
            celldofs = element_dofs(nodeids, dim)
            celldisp = nodaldisplacement[celldofs]
            points = cellquads[-1, cellid].points
            update_parent_stress!(
                qpstress,
                basis,
                stiffness,
                celldisp,
                points,
                jac,
                vectosymmconverter,
            )
        end
    end
    return reshape(qpstress, 4, :)
end

function parent_stress_at_interface_quadrature_points(
    nodaldisplacement,
    basis,
    stiffness,
    interfacequads,
    cutmesh,
)

    dim = dimension(basis)
    qpstress = zeros(0)
    jac = jacobian(cutmesh)
    vectosymmconverter = vector_to_symmetric_matrix_converter()
    cellsign = cell_sign(cutmesh)

    cellids = findall(cellsign .== 0)

    for cellid in cellids
        nodeids = nodal_connectivity(cutmesh, -1, cellid)
        celldofs = element_dofs(nodeids, dim)
        celldisp = nodaldisplacement[celldofs]
        points = interfacequads[-1, cellid].points
        update_parent_stress!(
            qpstress,
            basis,
            stiffness,
            celldisp,
            points,
            jac,
            vectosymmconverter,
        )
    end
    return reshape(qpstress, 4, :)
end

function product_stress_at_interface_quadrature_points(
    nodaldisplacement,
    basis,
    stiffness,
    transfstress,
    theta0,
    interfacequads,
    cutmesh,
)

    dim = dimension(basis)
    qpstress = zeros(0)
    jac = jacobian(cutmesh)
    vectosymmconverter = vector_to_symmetric_matrix_converter()
    cellsign = cell_sign(cutmesh)

    cellids = findall(cellsign .== 0)

    for cellid in cellids
        nodeids = nodal_connectivity(cutmesh, +1, cellid)
        celldofs = element_dofs(nodeids, dim)
        celldisp = nodaldisplacement[celldofs]
        points = interfacequads[+1, cellid].points
        update_product_stress!(
            qpstress,
            basis,
            stiffness,
            transfstress,
            theta0,
            celldisp,
            points,
            jac,
            vectosymmconverter,
        )
    end
    return reshape(qpstress, 4, :)
end

function cell_quadrature_points(cellquads, cutmesh)
    ncells = number_of_cells(cutmesh)
    coords = zeros(2, 0)
    for cellid = 1:ncells
        s = cell_sign(cutmesh, cellid)
        cellmap = cell_map(cutmesh, cellid)
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

function interface_quadrature_points(interfacequads, cutmesh)
    cellsign = cell_sign(cutmesh)
    cellids = findall(cellsign .== 0)
    numcellqps = length(interfacequads.quads[1])
    numqps = numcellqps * length(cellids)
    points = zeros(2, numqps)
    counter = 1
    for cellid in cellids
        cellmap = cell_map(cutmesh, cellid)
        qp = cellmap(interfacequads[1, cellid].points)
        points[:, counter:(counter+numcellqps-1)] .= qp
        counter += numcellqps
    end
    return points
end
