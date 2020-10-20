struct InterfaceOperators
    operators::Any
    celltooperator::Any
    ncells::Any
    nops::Any
    function InterfaceOperators(operators, celltooperator)
        ncells = length(celltooperator)
        nphase, nops = size(operators)

        @assert nphase == 2
        @assert all(celltooperator .>= 0)
        @assert all(celltooperator .<= nops)
        new(operators, celltooperator, ncells, nops)
    end
end

function Base.getindex(interfaceoperator::InterfaceOperators, s, cellid)
    row = cell_sign_to_row(s)
    idx = interfaceoperator.celltooperator[cellid]
    idx > 0 || error("Cell $cellid with cellsign $s does not have an operator")
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
    return interfaceoperator.nops
end

function interface_mass_operators(basis, interfacequads, cellmap, cellsign, penalty)

    ncells = length(cellsign)
    hasinterface = cellsign .== 0
    numinterfaces = count(hasinterface)
    operators = Matrix{Any}(undef, 2, numinterfaces)
    celltooperator = zeros(Int, ncells)
    dim = dimension(basis)

    cellids = findall(hasinterface)
    counter = 1
    for cellid in cellids
        normal = interface_normals(interfacequads, cellid)
        facescale = scale_area(cellmap, normal)
        for s in [+1, -1]
            row = cell_sign_to_row(s)
            quad = interfacequads[s, cellid]
            mass = penalty * mass_matrix(basis, quad, facescale, dim)
            operators[row, counter] = mass
        end
        celltooperator[cellid] = counter
        counter += 1
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
    operators = Matrix{Any}(undef,2,numinterfaces)
    celltooperator = zeros(Int, ncells)

    cellids = findall(hasinterface)
    counter = 1
    for (cellid, s) in enumerate(cellsign)
        normal = interface_normals(interfacequads,cellid)
        for s in [+1,-1]
            row = cell_sign_to_row(s)
            quad = interfacequads[s,cellid]
            top = coherent_traction_operator(basis,quad,normal,stiffness[s],cellmap)
            operators[row,counter] = top
        end
        celltooperator[cellid] = counter
        counter += 1
    end
    return InterfaceOperators(operators, celltooperator)
end

function interface_traction_operators(basis, interfacequads, stiffness, cutmesh)
    cellmap = cell_map(cutmesh, 1)
    cellsign = cell_sign(cutmesh)
    return InterfaceTractionOperator(basis, interfacequads, stiffness, cellmap, cellsign)
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
    tractionoperator = interface_traction_operators(basis, interfacequads, stiffness, cutmesh)
    massoperator = interface_mass_operators(basis, interfacequads, cutmesh, penalty)
    return InterfaceCondition(tractionoperator, massoperator, penalty)
end

function traction_operator(interfacecondition::InterfaceCondition, s, cellid)
    return interfacecondition.tractionoperator[s, cellid]
end

function mass_operator(interfacecondition::InterfaceCondition, cellid)
    return interfacecondition.massoperator[cellid]
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
