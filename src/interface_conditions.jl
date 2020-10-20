struct InterfaceMassOperator
    operators::Any
    celltooperator::Any
    ncells::Any
    function InterfaceMassOperator(operators, celltooperator)
        ncells = length(celltooperator)
        @assert all(celltooperator .>= 0)
        @assert all(celltooperator .<= length(operators))
        new(operators, celltooperator, ncells)
    end
end

function InterfaceMassOperator(basis, interfacequads, cellmap, cellsign, penalty)
    ncells = length(cellsign)
    operators = []
    celltooperator = zeros(Int, ncells)
    dim = dimension(basis)

    for (cellid, s) in enumerate(cellsign)
        if s == 0
            fquad = interfacequads[cellid]
            normal = interface_normals(interfacequads, cellid)
            facescale = scale_area(cellmap, normal)

            mass = penalty * mass_matrix(basis, fquad, facescale, dim)
            push!(operators, mass)
            celltooperator[cellid] = length(operators)
        end
    end
    return InterfaceMassOperator(operators, celltooperator)
end

function InterfaceMassOperator(basis, interfacequads, cutmesh, penalty)
    cellmap = cell_map(cutmesh, 1)
    cellsign = cell_sign(cutmesh)
    return InterfaceMassOperator(basis, interfacequads, cellmap, cellsign, penalty)
end

function Base.getindex(massoperator::InterfaceMassOperator, cellid)
    return massoperator.operators[massoperator.celltooperator[cellid]]
end

function Base.show(io::IO, massoperator::InterfaceMassOperator)
    ncells = massoperator.ncells
    nuniqueoperators = length(massoperator.operators)
    str = "InterfaceMassOperator\n\tNum. Cells: $ncells\n\tNum. Unique Operators: $nuniqueoperators"
    print(io, str)
end

function number_of_cells(massoperator::InterfaceMassOperator)
    return massoperator.ncells
end

struct InterfaceTractionOperator
    operators::Any
    celltooperator::Any
    ncells::Any
    function InterfaceTractionOperator(operators, celltooperator)
        nphase, ncells = size(celltooperator)
        @assert nphase == 2
        @assert all(celltooperator .>= 0)
        @assert all(celltooperator .<= length(operators))
        new(operators, celltooperator, ncells)
    end
end

function InterfaceTractionOperator(basis, interfacequads, stiffness, cellmap, cellsign)
    ncells = length(cellsign)
    operators = []
    celltooperator = zeros(Int, 2, ncells)

    for (cellid, s) in enumerate(cellsign)
        if s == 0
            fquad = interfacequads[cellid]
            negativenormal = interface_normals(interfacequads, cellid)

            positivetop = coherent_traction_operator(
                basis,
                fquad,
                negativenormal,
                stiffness[+1],
                cellmap,
            )
            push!(operators, positivetop)
            celltooperator[1, cellid] = length(operators)

            negativetop = coherent_traction_operator(
                basis,
                fquad,
                negativenormal,
                stiffness[-1],
                cellmap,
            )
            push!(operators, negativetop)
            celltooperator[2, cellid] = length(operators)
        end
    end
    return InterfaceTractionOperator(operators, celltooperator)
end

function InterfaceTractionOperator(basis, interfacequads, stiffness, cutmesh)
    cellmap = cell_map(cutmesh, 1)
    cellsign = cell_sign(cutmesh)
    return InterfaceTractionOperator(basis, interfacequads, stiffness, cellmap, cellsign)
end

function Base.getindex(tractionoperator::InterfaceTractionOperator, s, cellid)
    row = cell_sign_to_row(s)
    return tractionoperator.operators[tractionoperator.celltooperator[row, cellid]]
end

function Base.show(io::IO, tractionoperator::InterfaceTractionOperator)
    ncells = tractionoperator.ncells
    nuniqueoperators = length(tractionoperator.operators)
    str = "InterfaceTractionOperator\n\tNum. Cells: $ncells\n\tNum. Unique Operators: $nuniqueoperators"
    print(io, str)
end

function number_of_cells(tractionoperator::InterfaceTractionOperator)
    return tractionoperator.ncells
end

struct InterfaceCondition
    tractionoperator::Any
    massoperator::Any
    penalty::Any
    ncells::Any
    function InterfaceCondition(
        tractionoperator::InterfaceTractionOperator,
        massoperator::InterfaceMassOperator,
        penalty,
    )

        ncells = number_of_cells(massoperator)
        @assert number_of_cells(tractionoperator) == ncells
        new(tractionoperator,massoperator,penalty,ncells)
    end
end

function InterfaceCondition(basis,interfacequads,stiffness,cutmesh,penalty)
    tractionoperator = InterfaceTractionOperator(basis,interfacequads,stiffness,cutmesh)
    massoperator = InterfaceMassOperator(basis,interfacequads,cutmesh,penalty)
    return InterfaceCondition(tractionoperator,massoperator,penalty)
end

function traction_operator(interfacecondition::InterfaceCondition,s,cellid)
    return interfacecondition.tractionoperator[s,cellid]
end

function mass_operator(interfacecondition::InterfaceCondition,cellid)
    return interfacecondition.massoperator[cellid]
end

function Base.show(io::IO,interfacecondition::InterfaceCondition)
    ncells = interfacecondition.ncells
    penalty = interfacecondition.penalty
    str = "InterfaceCondition\n\tNum. Cells: $ncells\n\tDisplacement Penalty: $penalty"
    print(io,str)
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
