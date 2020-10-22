struct InterfaceOperators
    operators::Any
    celltooperator::Any
    ncells::Any
    numoperators::Any
    function InterfaceOperators(operators, celltooperator)
        ncells = length(celltooperator)
        numcouples, numoperators = size(operators)

        @assert numcouples == 4
        @assert all(celltooperator .>= 0)
        @assert all(celltooperator .<= numoperators)
        new(operators, celltooperator, ncells, numoperators)
    end
end

function Base.getindex(interfaceoperator::InterfaceOperators, s1, s2, cellid)
    row = cell_couple_sign_to_row(s1, s2)
    idx = interfaceoperator.celltooperator[cellid]
    idx > 0 || error("Cell $cellid does not have an operator")
    return interfaceoperator.operators[row, idx]
end

function Base.show(io::IO, interfaceoperator::InterfaceOperators)
    ncells = number_of_cells(interfaceoperator)
    nops = number_of_operators(interfaceoperator)
    str = "InterfaceOperator\n\tNum. Cells: $ncells\n\tNum. Operators: $nops"
    print(io, str)
end

function number_of_cells(interfaceoperator::InterfaceOperators)
    return interfaceoperator.ncells
end

function number_of_operators(interfaceoperator::InterfaceOperators)
    return interfaceoperator.numoperators
end

function interface_mass_operators(basis, interfacequads, cellmap, cellsign, penalty)

    ncells = length(cellsign)
    hasinterface = cellsign .== 0
    numinterfaces = count(hasinterface)
    operators = Matrix{Any}(undef, 4, numinterfaces)
    celltooperator = zeros(Int, ncells)
    dim = dimension(basis)

    cellids = findall(hasinterface)

    for (idx, cellid) in enumerate(cellids)
        normal = interface_normals(interfacequads, cellid)
        facescale = scale_area(cellmap, normal)
        for s1 in [+1, -1]
            quad1 = interfacequads[s1, cellid]
            for s2 in [+1, -1]
                row = cell_couple_sign_to_row(s1, s2)
                quad2 = interfacequads[s2, cellid]
                mass = penalty * interface_mass_matrix(basis, quad1, quad2, facescale)
                operators[row, idx] = mass
            end
        end
        celltooperator[cellid] = idx
    end
    return InterfaceOperators(operators, celltooperator)
end

function interface_mass_operators(basis, interfacequads, cutmesh, penalty)
    cellmap = cell_map(cutmesh, 1)
    cellsign = cell_sign(cutmesh)
    return interface_mass_operators(basis, interfacequads, cellmap, cellsign, penalty)
end

function interface_traction_operators(basis, interfacequads, stiffness, cellmap, cellsign)

    ncells = length(cellsign)
    hasinterface = cellsign .== 0
    numinterfaces = count(hasinterface)
    operators = Matrix{Any}(undef, 4, numinterfaces)
    celltooperator = zeros(Int, ncells)

    cellids = findall(hasinterface)

    for (idx, cellid) in enumerate(cellids)
        normal = interface_normals(interfacequads, cellid)
        for s1 in [+1, -1]
            quad1 = interfacequads[s1, cellid]
            for s2 in [+1, -1]
                row = cell_couple_sign_to_row(s1, s2)
                quad2 = interfacequads[s2, cellid]
                top = coherent_traction_operator(
                    basis,
                    quad1,
                    quad2,
                    normal,
                    stiffness[s2],
                    cellmap,
                )
                operators[row, idx] = top
            end
        end
        celltooperator[cellid] = idx
    end
    return InterfaceOperators(operators, celltooperator)
end

function interface_traction_operators(basis, interfacequads, stiffness, cutmesh)
    cellmap = cell_map(cutmesh, 1)
    cellsign = cell_sign(cutmesh)
    return interface_traction_operators(basis, interfacequads, stiffness, cellmap, cellsign)
end

struct InterfaceCondition
    tractionoperator::Any
    massoperator::Any
    penalty::Any
    ncells::Any
    function InterfaceCondition(
        tractionoperator::InterfaceOperators,
        massoperator::InterfaceOperators,
        penalty,
    )

        ncells = number_of_cells(massoperator)
        @assert number_of_cells(tractionoperator) == ncells
        new(tractionoperator, massoperator, penalty, ncells)
    end
end

function InterfaceCondition(basis, interfacequads, stiffness, cutmesh, penalty)
    tractionoperator =
        interface_traction_operators(basis, interfacequads, stiffness, cutmesh)
    massoperator = interface_mass_operators(basis, interfacequads, cutmesh, penalty)
    return InterfaceCondition(tractionoperator, massoperator, penalty)
end

function traction_operator(interfacecondition::InterfaceCondition, s, cellid)
    return interfacecondition.tractionoperator[s, cellid]
end

function mass_operator(interfacecondition::InterfaceCondition, s, cellid)
    return interfacecondition.massoperator[s, cellid]
end

function Base.show(io::IO, interfacecondition::InterfaceCondition)
    ncells = interfacecondition.ncells
    penalty = interfacecondition.penalty
    str = "InterfaceCondition\n\tNum. Cells: $ncells\n\tDisplacement Penalty: $penalty"
    print(io, str)
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

function coherent_traction_operator(basis, quad1, quad2, normals, stiffness, cellmap)
    numqp = length(quad1)
    @assert length(quad2) == size(normals)[2] == numqp
    dim = dimension(basis)
    nf = number_of_basis_functions(basis)
    ndofs = dim * nf
    matrix = zeros(ndofs, ndofs)
    vectosymmconverter = vector_to_symmetric_matrix_converter()

    jac = jacobian(cellmap)
    scalearea = scale_area(cellmap, normals)

    for qpidx = 1:numqp
        p1, w1 = quad1[qpidx]
        p2, w2 = quad2[qpidx]
        @assert w1 ≈ w2

        vals = basis(p1)
        grad = transform_gradient(gradient(basis, p2), jac)
        normal = normals[:, qpidx]
        NK = sum([make_row_matrix(vectosymmconverter[k], grad[:, k]) for k = 1:dim])
        N = sum([normal[k] * vectosymmconverter[k]' for k = 1:dim])
        NI = interpolation_matrix(vals, dim)

        matrix .+= NI' * N * stiffness * NK * scalearea[qpidx] * w1
    end
    return matrix
end

function interface_mass_matrix(basis, quad1, quad2, scale)
    numqp = length(quad1)
    @assert length(quad2) == length(scale) == numqp
    nf = number_of_basis_functions(basis)
    dim = dimension(basis)
    totaldofs = dim * nf
    matrix = zeros(totaldofs, totaldofs)
    for qpidx = 1:numqp
        p1, w1 = quad1[qpidx]
        p2, w2 = quad2[qpidx]
        @assert w1 ≈ w2

        vals1 = basis(p1)
        vals2 = basis(p2)

        NI1 = interpolation_matrix(vals1, dim)
        NI2 = interpolation_matrix(vals2, dim)

        matrix .+= NI1' * NI2 * scale[qpidx] * w1
    end
    return matrix
end
