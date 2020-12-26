struct CellQuadratures
    quads::Any
    celltoquad::Any
    ncells::Any
    function CellQuadratures(quads, celltoquad)
        nphase, ncells = size(celltoquad)
        @assert nphase == 2
        @assert all(celltoquad .>= 0)
        @assert all(celltoquad .<= length(quads))
        new(quads, celltoquad, ncells)
    end
end

function CellQuadratures(
    cellsign,
    levelset,
    levelsetcoeffs,
    nodalconnectivity,
    numuniformqp,
    numcutqp,
)
    numcells = length(cellsign)
    @assert size(nodalconnectivity)[2] == numcells

    tpq = tensor_product_quadrature(2, numuniformqp)

    quads = [tpq]

    celltoquad = zeros(Int, 2, numcells)
    xL, xR = [-1.0, -1.0], [1.0, 1.0]

    for cellid = 1:numcells
        s = cellsign[cellid]
        if s == +1
            celltoquad[1, cellid] = 1
        elseif s == -1
            celltoquad[2, cellid] = 1
        elseif s == 0
            nodeids = nodalconnectivity[:, cellid]
            update!(levelset, levelsetcoeffs[nodeids])

            try
                pquad = area_quadrature(levelset, +1, xL, xR, numcutqp, numsplits = 2)
                push!(quads, pquad)
                celltoquad[1, cellid] = length(quads)

                nquad = area_quadrature(levelset, -1, xL, xR, numcutqp, numsplits = 2)
                push!(quads, nquad)
                celltoquad[2, cellid] = length(quads)
            catch e
                pquad = area_quadrature(levelset, +1, xL, xR, numcutqp, numsplits = 3)
                push!(quads, pquad)
                celltoquad[1, cellid] = length(quads)

                nquad = area_quadrature(levelset, -1, xL, xR, numcutqp, numsplits = 3)
                push!(quads, nquad)
                celltoquad[2, cellid] = length(quads)
            end
        else
            error("Expected cellsign ∈ {-1,0,+1}, got cellsign = $s")
        end
    end
    return CellQuadratures(quads, celltoquad)
end

function CellQuadratures(levelset, levelsetcoeffs, cutmesh, numuniformqp, numcutqp)

    cellsign = cell_sign(cutmesh)
    nodalconnectivity = nodal_connectivity(cutmesh.mesh)
    return CellQuadratures(
        cellsign,
        levelset,
        levelsetcoeffs,
        nodalconnectivity,
        numuniformqp,
        numcutqp,
    )
end

function CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
    return CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp, numqp)
end

function Base.getindex(vquads::CellQuadratures, s, cellid)
    row = cell_sign_to_row(s)
    idx = vquads.celltoquad[row, cellid]
    idx > 0 || error("Cell $cellid, cellsign $s, does not have a cell quadrature")
    return vquads.quads[vquads.celltoquad[row, cellid]]
end

function Base.show(io::IO, cellquads::CellQuadratures)
    ncells = cellquads.ncells
    nuniquequads = length(cellquads.quads)
    str = "CellQuadratures\n\tNum. Cells: $ncells\n\tNum. Unique Quadratures: $nuniquequads"
    print(io, str)
end

function uniform_cell_quadrature(vquads::CellQuadratures)
    return vquads.quads[1]
end

function number_of_cells(cellquads::CellQuadratures)
    return cellquads.ncells
end

function has_quadrature(cellquads::CellQuadratures, s, cellid)
    row = cell_sign_to_row(s)
    return cellquads.celltoquad[row, cellid] != 0
end
