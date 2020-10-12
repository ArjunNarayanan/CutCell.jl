function apply_homogeneous_linear_constraint!(matrix, row, cols, coeffs)
    @assert length(cols) == length(coeffs)
    @assert row in cols

    set_to_zero = matrix[row, :]
    for idx in set_to_zero.nzind
        if idx != row
            matrix[row, idx] = 0.0
        end
    end

    scale = matrix[row, row]
    for (idx, col) in enumerate(cols)
        matrix[row, col] = scale * coeffs[idx]
    end
end

function couple_node_degrees_of_freedom!(
    matrix,
    positivenodeids,
    negativenodeids,
    equationnodeid,
    dof,
    basisvals,
)
    @assert length(positivenodeids) == length(negativenodeids) == length(basisvals)

    positivedofs = [node_to_dof_id(n, dof, 2) for n in positivenodeids]
    negativedofs = [node_to_dof_id(n, dof, 2) for n in negativenodeids]
    equationdof = node_to_dof_id(equationnodeid, dof, 2)

    coeffs = vcat(basisvals, -basisvals)
    cols = vcat(positivedofs, negativedofs)

    apply_homogeneous_linear_constraint!(matrix, equationdof, cols, coeffs)
end

function apply_coherent_displacement_constraint!(matrix, basis, interfacequads, cutmesh)

    cellsign = cell_sign(cutmesh)
    cellid = findfirst(x -> x == 0, cellsign)

    positivenodeids = nodal_connectivity(cutmesh, +1, cellid)
    negativenodeids = nodal_connectivity(cutmesh, -1, cellid)

    quad = interfacequads[cellid]
    @assert length(quad) > 1

    p1 = quad[1][1]
    basisvals = basis(p1)
    CutCell.couple_node_degrees_of_freedom!(
        matrix,
        positivenodeids,
        negativenodeids,
        positivenodeids[1],
        1,
        basisvals,
    )
    CutCell.couple_node_degrees_of_freedom!(
        matrix,
        positivenodeids,
        negativenodeids,
        positivenodeids[1],
        2,
        basisvals,
    )

    p2 = quad[2][1]
    basisvals = basis(p2)
    CutCell.couple_node_degrees_of_freedom!(
        matrix,
        positivenodeids,
        negativenodeids,
        positivenodeids[2],
        1,
        basisvals,
    )
end

function coherent_traction_operator(basis, quad, normals, stiffness, cellmap)

    @assert size(normals)[2] == length(quad)
    dim = dimension(basis)
    nf = number_of_basis_functions(basis)
    ndofs = dim * nf
    matrix = zeros(ndofs, ndofs)
    vectosymmconverter = vector_to_symmetric_matrix_converter()

    jac = jacobian(cellmap)
    scalearea = scale_area(cellmap, normals)

    for (idx, (p, w)) in enumerate(quad)
        grad = transform_gradient(gradient(basis, p), jac)
        vals = basis(p)
        normal = normals[:, idx]
        NK = zeros(3, 2nf)
        N = zeros(2, 3)
        for k = 1:dim
            NK .+= make_row_matrix(vectosymmconverter[k], grad[:, k])
            N .+= normal[k] * vectosymmconverter[k]'
        end
        NI = interpolation_matrix(vals, dim)
        matrix .+= NI' * N * stiffness * NK * scalearea[idx] * w
    end
    return matrix
end

function coherent_traction_operator(basis, testquad, trialquad, normals, stiffness, cellmap)
    numqp = length(testquad)
    @assert length(trialquad) == numqp
    @assert size(normals)[2] == numqp

    dim = dimension(basis)
    nf = number_of_basis_functions(basis)
    ndofs = dim * nf
    matrix = zeros(ndofs, ndofs)
    vectorsymmconverter = vector_to_symmetric_matrix_converter()

    jac = jacobian(cellmap)
    scalearea = scale_area(cellmap, normals)

    for idx = 1:numqp
        testp, testw = testquad[idx]
        trialp, trialw = trialquad[idx]

        grad = transform_gradient(gradient(basis, trialp), jac)
        vals = basis(testp)

        normal = normals[:, idx]
        NK = zeros(3, 2nf)
        N = zeros(2, 3)
        for k = 1:dim
            NK .+= make_row_matrix(vectorsymmconverter[k], grad[:, k])
            N .+= normal[k] * vectorsymmconverter[k]'
        end
        NI = interpolation_matrix(vals, dim)
        matrix .+= NI' * N * stiffness * NK * scalearea[idx] * testw
    end
    return matrix
end

function assemble_coherent_traction_operator!(
    sysmatrix,
    basis,
    interfacequads,
    stiffness,
    cutmesh,
)

    cellsign = cell_sign(cutmesh)
    ncells = number_of_cells(cutmesh)
    dim = dimension(cutmesh)

    for cellid = 1:ncells
        s = cellsign[cellid]
        if s == 0
            cellmap = cell_map(cutmesh, cellid)
            normals = interface_normals(interfacequads, cellid)
            quad = interfacequads[cellid]
            positivestiffness = stiffness[1]
            negativestiffness = stiffness[-1]
            positivenodeids = nodal_connectivity(cutmesh, +1, cellid)
            negativenodeids = nodal_connectivity(cutmesh, -1, cellid)

            positiveop =
                +coherent_traction_operator(
                    basis,
                    quad,
                    normals,
                    negativestiffness,
                    cellmap,
                )
            assemble_couple_cell_matrix!(
                sysmatrix,
                positivenodeids,
                negativenodeids,
                dim,
                vec(positiveop),
            )

            # negativeop =
            #     -coherent_traction_operator(
            #         basis,
            #         quad,
            #         normals,
            #         positivestiffness,
            #         cellmap,
            #     )
            # assemble_couple_cell_matrix!(
            #     sysmatrix,
            #     negativenodeids,
            #     positivenodeids,
            #     dim,
            #     vec(negativeop),
            # )
        end
    end
end
