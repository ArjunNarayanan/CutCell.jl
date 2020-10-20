abstract type AbstractBoundaryOperator end

struct BoundaryOperator <: AbstractBoundaryOperator
    operators::Any
    facetooperator::Any
    ncells::Any
    function BoundaryOperator(operators, facetooperator)
        nphase, nface, ncells = size(facetooperator)
        @assert nphase == 2
        @assert nface == 4
        @assert all(facetooperator .>= 0)
        @assert all(facetooperator .<= length(operators))
        new(operators, facetooperator, ncells)
    end
end

function boundary_mass_operator(basis, facequads, cutmesh, onboundary, penalty)
    cellids = findall(is_boundary_cell(cutmesh))
    cellconnectivity = cell_connectivity(cutmesh)
    facemidpoints = reference_face_midpoints()
    nfaces = length(facemidpoints)
    cellsign = cell_sign(cutmesh)
    ncells = length(cellsign)
    facedetjac = face_determinant_jacobian(cell_map(cutmesh, 1))
    dim = dimension(basis)

    refquads = reference_face_quadrature(facequads)
    operators = [
        penalty * mass_matrix(basis, q, detjac, dim)
        for (q, detjac) in zip(refquads, facedetjac)
    ]
    facetooperator = zeros(Int, 2, 4, ncells)

    for cellid in cellids
        cellmap = cell_map(cutmesh, cellid)
        s = cellsign[cellid]
        for faceid = 1:nfaces
            if cellconnectivity[faceid, cellid] == 0
                if onboundary(cellmap(facemidpoints[faceid]))
                    if s == 1
                        facetooperator[1, faceid, cellid] = faceid
                    elseif s == -1
                        facetooperator[2, faceid, cellid] = faceid
                    elseif s == 0
                        pquad = facequads[+1, faceid, cellid]
                        positiveop =
                            penalty * mass_matrix(basis, pquad, facedetjac[faceid], dim)
                        push!(operators, positiveop)
                        facetooperator[1, faceid, cellid] = length(operators)

                        nquad = facequads[-1, faceid, cellid]
                        negativeop =
                            penalty * mass_matrix(basis, nquad, facedetjac[faceid], dim)
                        push!(operators, negativeop)
                        facetooperator[2, faceid, cellid] = length(operators)
                    else
                        error("Expected cellsign ∈ {-1,0,1}, got cellsign = $s")
                    end
                end
            end
        end
    end
    BoundaryOperator(operators, facetooperator)
end

function boundary_traction_operator(basis, facequads, stiffness, cutmesh, onboundary)

    cellids = findall(is_boundary_cell(cutmesh))
    cellconnectivity = cell_connectivity(cutmesh)
    facemidpoints = reference_face_midpoints()
    nfaces = length(facemidpoints)
    cellsign = cell_sign(cutmesh)
    ncells = length(cellsign)

    cellmap = cell_map(cutmesh, 1)
    facedetjac = face_determinant_jacobian(cellmap)
    dim = dimension(basis)

    refnormals = reference_face_normals()
    refquads = reference_face_quadrature(facequads)
    operators1 = [
        face_traction_operator(basis, q, n, stiffness[+1], detjac, cellmap)
        for (q, n, detjac) in zip(refquads, refnormals, facedetjac)
    ]
    operators2 = [
        face_traction_operator(basis, q, n, stiffness[-1], detjac, cellmap)
        for (q, n, detjac) in zip(refquads, refnormals, facedetjac)
    ]

    operators = vcat(operators1, operators2)
    facetooperator = zeros(Int, 2, 4, ncells)

    for cellid in cellids
        cellmap = cell_map(cutmesh, cellid)
        s = cellsign[cellid]
        for faceid = 1:nfaces
            if cellconnectivity[faceid, cellid] == 0
                if onboundary(cellmap(facemidpoints[faceid]))
                    if s == 1
                        facetooperator[1, faceid, cellid] = faceid
                    elseif s == -1
                        facetooperator[2, faceid, cellid] = nfaces + faceid
                    elseif s == 0
                        pquad = facequads[+1, faceid, cellid]
                        positiveop = face_traction_operator(
                            basis,
                            pquad,
                            refnormals[faceid],
                            stiffness[1],
                            facedetjac[faceid],
                            cellmap,
                        )
                        push!(operators, positiveop)
                        facetooperator[1, faceid, cellid] = length(operators)

                        nquad = facequads[-1, faceid, cellid]
                        negativeop = face_traction_operator(
                            basis,
                            nquad,
                            refnormals[faceid],
                            stiffness[-1],
                            facedetjac[faceid],
                            cellmap,
                        )
                        push!(operators, negativeop)
                        facetooperator[2, faceid, cellid] = length(operators)
                    else
                        error("Expected cellsign ∈ {-1,0,+1}, got cellsign = $s")
                    end
                end
            end
        end
    end
    BoundaryOperator(operators, facetooperator)
end

function boundary_displacement_rhs(bcfunc, basis, facequads, cutmesh, onboundary, penalty)
    cellids = findall(is_boundary_cell(cutmesh))
    cellconnectivity = cell_connectivity(cutmesh)
    facemidpoints = reference_face_midpoints()
    nfaces = length(facemidpoints)
    cellsign = cell_sign(cutmesh)
    facedetjac = face_determinant_jacobian(cell_map(cutmesh, 1))
    ncells = length(cellsign)

    vectors = []
    facetovectors = zeros(Int, 2, 4, ncells)

    for cellid in cellids
        cellmap = cell_map(cutmesh, cellid)
        s = cellsign[cellid]
        for faceid = 1:nfaces
            if cellconnectivity[faceid, cellid] == 0
                if onboundary(cellmap(facemidpoints[faceid]))
                    @assert s == +1 || s == 0 || s == -1
                    if s == +1 || s == 0
                        pquad = facequads[+1, faceid, cellid]
                        rhs =
                            penalty *
                            linear_form(bcfunc, basis, pquad, cellmap, facedetjac[faceid])
                        push!(vectors, rhs)
                        facetovectors[1, faceid, cellid] = length(vectors)
                    end
                    if s == -1 || s == 0
                        nquad = facequads[-1, faceid, cellid]
                        rhs =
                            penalty *
                            linear_form(bcfunc, basis, nquad, cellmap, facedetjac[faceid])
                        push!(vectors, rhs)
                        facetovectors[2, faceid, cellid] = length(vectors)
                    end
                end
            end
        end
    end
    BoundaryOperator(vectors, facetovectors)
end

struct BoundaryComponentOperator <: AbstractBoundaryOperator
    operators::Any
    facetooperator::Any
    ncells::Any
    component::Any
    function BoundaryComponentOperator(operators, facetooperator, component)
        nphase, nface, ncells = size(facetooperator)
        @assert nphase == 2
        @assert nface == 4
        @assert all(facetooperator .>= 0)
        @assert all(facetooperator .<= length(operators))
        new(operators, facetooperator, ncells, component)
    end
end

function boundary_mass_component_operator(
    basis,
    facequads,
    cutmesh,
    onboundary,
    component,
    penalty,
)
    cellids = findall(is_boundary_cell(cutmesh))
    cellconnectivity = cell_connectivity(cutmesh)
    facemidpoints = reference_face_midpoints()
    nfaces = length(facemidpoints)
    cellsign = cell_sign(cutmesh)
    ncells = length(cellsign)
    facedetjac = face_determinant_jacobian(cell_map(cutmesh, 1))
    dim = dimension(basis)

    refquads = reference_face_quadrature(facequads)
    operators = [
        penalty * component_mass_matrix(basis, q, component, detjac)
        for (q, detjac) in zip(refquads, facedetjac)
    ]
    facetooperator = zeros(Int, 2, 4, ncells)

    for cellid in cellids
        cellmap = cell_map(cutmesh, cellid)
        s = cellsign[cellid]
        for faceid = 1:nfaces
            if cellconnectivity[faceid, cellid] == 0
                if onboundary(cellmap(facemidpoints[faceid]))
                    if s == 1
                        facetooperator[1, faceid, cellid] = faceid
                    elseif s == -1
                        facetooperator[2, faceid, cellid] = faceid
                    elseif s == 0
                        pquad = facequads[+1, faceid, cellid]
                        positiveop =
                            penalty * component_mass_matrix(
                                basis,
                                pquad,
                                component,
                                facedetjac[faceid],
                            )
                        push!(operators, positiveop)
                        facetooperator[1, faceid, cellid] = length(operators)

                        nquad = facequads[-1, faceid, cellid]
                        negativeop =
                            penalty * component_mass_matrix(
                                basis,
                                nquad,
                                component,
                                facedetjac[faceid],
                            )
                        push!(operators, negativeop)
                        facetooperator[2, faceid, cellid] = length(operators)
                    else
                        error("Expected cellsign ∈ {-1,0,1}, got cellsign = $s")
                    end
                end
            end
        end
    end
    BoundaryComponentOperator(operators, facetooperator, component)
end

function boundary_traction_component_operator(
    basis,
    facequads,
    stiffness,
    cutmesh,
    onboundary,
    component,
)

    cellids = findall(is_boundary_cell(cutmesh))
    cellconnectivity = cell_connectivity(cutmesh)
    facemidpoints = reference_face_midpoints()
    nfaces = length(facemidpoints)
    cellsign = cell_sign(cutmesh)
    ncells = length(cellsign)

    cellmap = cell_map(cutmesh, 1)
    facedetjac = face_determinant_jacobian(cellmap)
    dim = dimension(basis)

    refnormals = reference_face_normals()
    refquads = reference_face_quadrature(facequads)
    operators1 = [
        face_traction_component_operator(
            basis,
            q,
            component,
            n,
            stiffness[+1],
            detjac,
            cellmap,
        ) for (q, n, detjac) in zip(refquads, refnormals, facedetjac)
    ]
    operators2 = [
        face_traction_component_operator(
            basis,
            q,
            component,
            n,
            stiffness[-1],
            detjac,
            cellmap,
        ) for (q, n, detjac) in zip(refquads, refnormals, facedetjac)
    ]

    operators = vcat(operators1, operators2)
    facetooperator = zeros(Int, 2, 4, ncells)

    for cellid in cellids
        cellmap = cell_map(cutmesh, cellid)
        s = cellsign[cellid]
        for faceid = 1:nfaces
            if cellconnectivity[faceid, cellid] == 0
                if onboundary(cellmap(facemidpoints[faceid]))
                    if s == 1
                        facetooperator[1, faceid, cellid] = faceid
                    elseif s == -1
                        facetooperator[2, faceid, cellid] = nfaces + faceid
                    elseif s == 0
                        pquad = facequads[+1, faceid, cellid]
                        positiveop = face_traction_component_operator(
                            basis,
                            pquad,
                            component,
                            refnormals[faceid],
                            stiffness[+1],
                            facedetjac[faceid],
                            cellmap,
                        )
                        push!(operators, positiveop)
                        facetooperator[1, faceid, cellid] = length(operators)

                        nquad = facequads[-1, faceid, cellid]
                        negativeop = face_traction_component_operator(
                            basis,
                            nquad,
                            component,
                            refnormals[faceid],
                            stiffness[-1],
                            facedetjac[faceid],
                            cellmap,
                        )
                        push!(operators, negativeop)
                        facetooperator[2, faceid, cellid] = length(operators)
                    else
                        error("Expected cellsign ∈ {-1,0,+1}, got cellsign = $s")
                    end
                end
            end
        end
    end
    BoundaryComponentOperator(operators, facetooperator, component)
end

function boundary_displacement_component_rhs(
    bcfunc,
    basis,
    facequads,
    cutmesh,
    onboundary,
    component,
    penalty,
)

    cellids = findall(is_boundary_cell(cutmesh))
    cellconnectivity = cell_connectivity(cutmesh)
    facemidpoints = reference_face_midpoints()
    nfaces = length(facemidpoints)
    cellsign = cell_sign(cutmesh)
    facedetjac = face_determinant_jacobian(cell_map(cutmesh, 1))
    ncells = length(cellsign)

    vectors = []
    facetovectors = zeros(Int, 2, 4, ncells)

    for cellid in cellids
        cellmap = cell_map(cutmesh, cellid)
        s = cellsign[cellid]
        for faceid = 1:nfaces
            if cellconnectivity[faceid, cellid] == 0
                if onboundary(cellmap(facemidpoints[faceid]))
                    @assert s == +1 || s == 0 || s == -1
                    if s == +1 || s == 0
                        pquad = facequads[+1, faceid, cellid]
                        rhs =
                            penalty * component_linear_form(
                                bcfunc,
                                basis,
                                pquad,
                                component,
                                cellmap,
                                facedetjac[faceid],
                            )
                        push!(vectors, rhs)
                        facetovectors[1, faceid, cellid] = length(vectors)
                    end
                    if s == -1 || s == 0
                        nquad = facequads[-1, faceid, cellid]
                        rhs =
                            penalty * component_linear_form(
                                bcfunc,
                                basis,
                                nquad,
                                component,
                                cellmap,
                                facedetjac[faceid],
                            )
                        push!(vectors, rhs)
                        facetovectors[2, faceid, cellid] = length(vectors)
                    end
                end
            end
        end
    end
    BoundaryComponentOperator(vectors, facetovectors, component)
end

function Base.getindex(bm::BO, s, faceid, cellid) where {BO<:AbstractBoundaryOperator}
    phaseid = cell_sign_to_row(s)
    idx = bm.facetooperator[phaseid, faceid, cellid]
    idx > 0 ||
        error("Expected idx > 0, got idx = $idx. Check if [s,faceid,cellid] =[$s,$faceid,$cellid] has a valid operator")
    return bm.operators[idx]
end

function Base.show(io::IO, bm::BoundaryOperator)
    ncells = bm.ncells
    noperators = length(bm.operators)
    str = "BoundaryOperator\n\tNum. Cells: $ncells\n\tNum. Unique Operators: $noperators"
    print(io, str)
end

function Base.show(io::IO, bm::BoundaryComponentOperator)
    ncells = bm.ncells
    noperators = length(bm.operators)
    component = bm.component
    str = "BoundaryComponentOperator\n\tNum. Cells: $ncells\n\tNum. Unique Operators: $noperators\n\tComponent: $component"
    print(io, str)
end

function number_of_cells(bm::BO) where {BO<:AbstractBoundaryOperator}
    return bm.ncells
end

function has_operator(bm::BO, s, faceid, cellid) where {BO<:AbstractBoundaryOperator}
    @assert s == -1 || s == +1
    phaseid = s == +1 ? 1 : 2
    return bm.facetooperator[phaseid, faceid, cellid] > 0
end

abstract type AbstractDisplacementCondition end

struct DisplacementCondition <: AbstractDisplacementCondition
    tractionoperator::Any
    massoperator::Any
    displacementrhs::Any
    penalty::Any
    ncells::Any
    function DisplacementCondition(
        tractionoperator::BoundaryOperator,
        massoperator::BoundaryOperator,
        displacementrhs::BoundaryOperator,
        penalty,
    )
        ncells = number_of_cells(massoperator)
        @assert number_of_cells(tractionoperator) == ncells
        @assert number_of_cells(displacementrhs) == ncells
        new(tractionoperator, massoperator, displacementrhs, penalty, ncells)
    end
end

function DisplacementCondition(
    displacementfunc,
    basis,
    facequads,
    stiffness,
    cutmesh,
    onboundary,
    penalty,
)
    massoperator = boundary_mass_operator(basis, facequads, cutmesh, onboundary, penalty)
    tractionoperator =
        boundary_traction_operator(basis, facequads, stiffness, cutmesh, onboundary)
    displacementrhs = boundary_displacement_rhs(
        displacementfunc,
        basis,
        facequads,
        cutmesh,
        onboundary,
        penalty,
    )
    return DisplacementCondition(tractionoperator, massoperator, displacementrhs, penalty)
end

struct DisplacementComponentCondition <: AbstractDisplacementCondition
    tractionoperator::Any
    massoperator::Any
    displacementrhs::Any
    penalty::Any
    component::Any
    ncells::Any
    function DisplacementComponentCondition(
        tractionoperator::BoundaryComponentOperator,
        massoperator::BoundaryComponentOperator,
        displacementrhs::BoundaryComponentOperator,
        penalty,
        component,
    )
        ncells = number_of_cells(massoperator)
        @assert number_of_cells(tractionoperator) == ncells
        @assert number_of_cells(displacementrhs) == ncells
        new(tractionoperator, massoperator, displacementrhs, penalty, component, ncells)
    end
end

function DisplacementComponentCondition(
    displacementfunc,
    basis,
    facequads,
    stiffness,
    cutmesh,
    onboundary,
    component,
    penalty,
)

    massoperator = boundary_mass_component_operator(
        basis,
        facequads,
        cutmesh,
        onboundary,
        component,
        penalty,
    )
    tractionoperator = boundary_traction_component_operator(
        basis,
        facequads,
        stiffness,
        cutmesh,
        onboundary,
        component,
    )
    displacementrhs = boundary_displacement_component_rhs(
        displacementfunc,
        basis,
        facequads,
        cutmesh,
        onboundary,
        component,
        penalty,
    )

    return DisplacementComponentCondition(
        tractionoperator,
        massoperator,
        displacementrhs,
        penalty,
        component,
    )
end

function traction_operator(
    dispcondition::DC,
    s,
    faceid,
    cellid,
) where {DC<:AbstractDisplacementCondition}
    return dispcondition.tractionoperator[s, faceid, cellid]
end

function mass_operator(
    dispcondition::DC,
    s,
    faceid,
    cellid,
) where {DC<:AbstractDisplacementCondition}
    return dispcondition.massoperator[s, faceid, cellid]
end

function displacement_rhs(
    dispcondition::DC,
    s,
    faceid,
    cellid,
) where {DC<:AbstractDisplacementCondition}
    return dispcondition.displacementrhs[s, faceid, cellid]
end

function Base.show(io::IO, dispcondition::DisplacementCondition)
    ncells = dispcondition.ncells
    penalty = dispcondition.penalty
    str = "DisplacementCondition\n\tNum. Cells: $ncells\n\tPenalty: $penalty"
    print(io, str)
end

function Base.show(io::IO, dispcondition::DisplacementComponentCondition)
    ncells = dispcondition.ncells
    penalty = dispcondition.penalty
    component = dispcondition.component
    str = "DisplacementComponentCondition\n\tNum. Cells: $ncells\n\tPenalty: $penalty\n\tComponent: $component"
    print(io, str)
end

function has_operator(
    dispcondition::DC,
    s,
    faceid,
    cellid,
) where {DC<:AbstractDisplacementCondition}
    flag = has_operator(dispcondition.tractionoperator, s, faceid, cellid)
    @assert has_operator(dispcondition.massoperator, s, faceid, cellid) == flag
    @assert has_operator(dispcondition.displacementrhs, s, faceid, cellid) == flag
    return flag
end
