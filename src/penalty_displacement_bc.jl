abstract type BoundaryOperator end

struct BoundaryMassOperator <: BoundaryOperator
    operators::Any
    facetooperator::Any
    ncells::Any
    function BoundaryMassOperator(operators, facetooperator)
        nphase, nface, ncells = size(facetooperator)
        @assert nphase == 2
        @assert nface == 4
        @assert all(facetooperator .>= 0)
        @assert all(facetooperator .<= length(operators))
        new(operators, facetooperator, ncells)
    end
end

function BoundaryMassOperator(basis, facequads, cutmesh, onboundary, penalty)
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
    BoundaryMassOperator(operators, facetooperator)
end

struct BoundaryComponentMassOperator <: BoundaryOperator
    operators::Any
    facetooperator::Any
    ncells::Any
    component::Any
    function BoundaryComponentMassOperator(operators, facetooperator, component)
        nphase, nface, ncells = size(facetooperator)
        @assert nphase == 2
        @assert nface == 4
        @assert all(facetooperator .>= 0)
        @assert all(facetooperator .<= length(operators))
        new(operators, facetooperator, ncells, component)
    end
end

function BoundaryComponentMassOperator(
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
                                dim,
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
    BoundaryComponentMassOperator(operators, facetooperator, component)
end

function Base.getindex(bm::BO, s, faceid, cellid) where {BO<:BoundaryOperator}
    (s == -1 || s == +1) ||
        error("Use ±1 to index into 1st dimension of $BO, got index = $s")
    phaseid = s == +1 ? 1 : 2
    idx = bm.facetooperator[phaseid,faceid,cellid]
    idx > 0 || error("Expected idx > 0, got idx = $idx. Check if [s,faceid,cellid] =[$s,$faceid,$cellid] has a valid operator")
    return bm.operators[idx]
end

function Base.show(io::IO,bm::BO) where {BO<:BoundaryOperator}
    ncells = bm.ncells
    noperators = length(bm.operators)
    str = "$BO\n\tNum. Cells: $ncells\n\tNum. Unique Operators: $noperators"
    print(io,str)
end
