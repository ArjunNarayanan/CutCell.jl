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

function Base.getindex(vquads::CellQuadratures, s, cellid)
    (s == -1 || s == +1) ||
        error("Use ±1 to index into rows (i.e. phase) of CellQuadratures")
    row = s == +1 ? 1 : 2
    (1 <= cellid <= vquads.ncells) ||
        throw(BoundsError(vquads.celltoquad, [row, cellid]))
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
    quad1d = ImplicitDomainQuadrature.ReferenceQuadratureRule(numcutqp)

    quads = [tpq]

    celltoquad = zeros(Int, 2, numcells)
    box = IntervalBox(-1..1, 2)

    for cellid = 1:numcells
        if cellsign[cellid] == +1
            celltoquad[1, cellid] = 1
        elseif cellsign[cellid] == -1
            celltoquad[2, cellid] = 1
        elseif cellsign[cellid] == 0
            nodeids = nodalconnectivity[:, cellid]
            update!(levelset, levelsetcoeffs[nodeids])

            pquad = area_quadrature(levelset, +1, box, quad1d)
            push!(quads, pquad)
            celltoquad[1, cellid] = length(quads)

            nquad = area_quadrature(levelset, -1, box, quad1d)
            push!(quads, nquad)
            celltoquad[2, cellid] = length(quads)
        end
    end
    return CellQuadratures(quads, celltoquad)
end

function CellQuadratures(
    levelset,
    levelsetcoeffs,
    cutmesh,
    numuniformqp,
    numcutqp,
)

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
