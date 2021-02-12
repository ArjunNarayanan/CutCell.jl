function pressure_at_points(stresses)
    return -1.0 / 3.0 * (stresses[1, :] + stresses[2, :] + stresses[4, :])
end

function stress_inner_product_over_points(stresses)
    return stresses[1,:].^2 + stresses[2,:].^2 + 2.0*stresses[3,:].^2 + stresses[4,:].^2
end

function deviatoric_stress(stressvector, p)
    return stressvector + p * [1.0, 1.0, 0.0, 1.0]
end

function deviatoric_stress_at_points(stresses, p)
    devstress = copy(stresses)
    devstress[1, :] .+= p
    devstress[2, :] .+= p
    devstress[4, :] .+= p

    return devstress
end

function traction_force(stressvector, normal)
    return [
        stressvector[1] * normal[1] + stressvector[3] * normal[2],
        stressvector[3] * normal[1] + stressvector[2] * normal[2],
    ]
end

function traction_force_at_points(stresses, normals)
    npts = size(stresses)[2]
    @assert size(normals)[2] == npts

    return hcat(
        [traction_force(stresses[:, i], normals[:, i]) for i = 1:npts]...,
    )
end

function normal_stress_component(stressvector, normal)
    tn = traction_force(stressvector,normal)
    return dot(tn,normal)
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

function product_stress(
    point,
    basis,
    stiffness,
    transfstress,
    theta0,
    celldisp,
    jac,
    vectosymmconverter,
)

    dim = dimension(basis)
    lambda, mu = lame_coefficients(stiffness, +1)

    grad = transform_gradient(gradient(basis, point), jac)
    NK = sum([make_row_matrix(vectosymmconverter[k], grad[:, k]) for k = 1:dim])
    symmdispgrad = NK * celldisp

    inplanestress = (stiffness[+1] * symmdispgrad) - transfstress
    s33 =
        lambda * (symmdispgrad[1] + symmdispgrad[2]) -
        (lambda + 2mu / 3) * theta0

    stress = vcat(inplanestress, s33)

    return stress
end

function product_stress_at_reference_points(
    refpoints,
    refcellids,
    basis,
    stiffness,
    transfstress,
    theta0,
    nodaldisplacement,
    cutmesh,
)

    dim, numpts = size(refpoints)
    productstress = zeros(4, numpts)
    vectosymmconverter = CutCell.vector_to_symmetric_matrix_converter()
    jac = CutCell.jacobian(cutmesh)

    for i = 1:numpts
        cellid = refcellids[i]
        nodeids = CutCell.nodal_connectivity(cutmesh, +1, cellid)
        celldofs = CutCell.element_dofs(nodeids, dim)
        celldisp = nodaldisplacement[celldofs]
        point = refpoints[:, i]

        productstress[:, i] .= product_stress(
            point,
            basis,
            stiffness,
            transfstress,
            theta0,
            celldisp,
            jac,
            vectosymmconverter,
        )
    end
    return productstress
end

function interpolate_at_reference_points(
    refpoints,
    refcellids,
    levelsetsign,
    basis,
    nodalvalues,
    dofspernode,
    cutmesh,
)
    dim, numpts = size(refpoints)
    interpolatedvals = zeros(dofspernode, numpts)

    for i = 1:numpts
        cellid = refcellids[i]
        nodeids = CutCell.nodal_connectivity(cutmesh, levelsetsign, cellid)
        celldofs = CutCell.element_dofs(nodeids, dofspernode)
        cellvals = nodalvalues[celldofs]

        vals = basis(refpoints[:, i])
        NI = interpolation_matrix(vals, dofspernode)

        interpolatedvals[:, i] = NI * cellvals
    end

    return interpolatedvals
end

function displacement_at_reference_points(
    refpoints,
    refcellids,
    levelsetsign,
    basis,
    nodaldisplacement,
    cutmesh,
)

    dim = dimension(basis)
    return interpolate_at_reference_points(
        refpoints,
        refcellids,
        levelsetsign,
        basis,
        nodaldisplacement,
        dim,
        cutmesh,
    )
end

function parent_stress(
    point,
    basis,
    stiffness,
    celldisp,
    jac,
    vectosymmconverter,
)
    dim = dimension(basis)
    lambda, mu = lame_coefficients(stiffness, -1)

    grad = transform_gradient(gradient(basis, point), jac)
    NK = sum([make_row_matrix(vectosymmconverter[k], grad[:, k]) for k = 1:dim])
    symmdispgrad = NK * celldisp

    inplanestress = stiffness[-1] * symmdispgrad
    s33 = lambda * (symmdispgrad[1] + symmdispgrad[2])

    stress = vcat(inplanestress, s33)

    return stress
end

function parent_stress_at_reference_points(
    refpoints,
    refcellids,
    basis,
    stiffness,
    nodaldisplacement,
    cutmesh,
)

    dim, numpts = size(refpoints)
    parentstress = zeros(4, numpts)
    vectosymmconverter = CutCell.vector_to_symmetric_matrix_converter()
    jac = CutCell.jacobian(cutmesh)

    for i = 1:numpts
        cellid = refcellids[i]
        nodeids = CutCell.nodal_connectivity(cutmesh, -1, cellid)
        celldofs = CutCell.element_dofs(nodeids, dim)
        celldisp = nodaldisplacement[celldofs]
        point = refpoints[:, i]

        parentstress[:, i] .= parent_stress(
            point,
            basis,
            stiffness,
            celldisp,
            jac,
            vectosymmconverter,
        )
    end
    return parentstress
end

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

        numericalstress = product_stress(
            p,
            basis,
            stiffness,
            transfstress,
            theta0,
            celldisp,
            jac,
            vectosymmconverter,
        )

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

        numericalstress = parent_stress(
            p,
            basis,
            stiffness,
            celldisp,
            jac,
            vectosymmconverter,
        )

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
