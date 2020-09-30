struct CutMeshCellQuadratures
    quads
    celltoquad
    ncells
    function CutMeshCellQuadratures(quads,celltoquad)
        nphase,ncells = size(celltoquad)
        @assert nphase == 2
        @assert all(celltoquad .>= 0)
        @assert all(celltoquad .<= length(quads))
        new(quads,celltoquad,ncells)
    end
end

function Base.getindex(vquads::CutMeshCellQuadratures,row,col)
    flag = (1 <= row <= 2) && (1 <= col <= vquads.ncells)
    flag || throw(BoundsError(vquads.celltoquad,[row,col]))
    return vquads.quads[vquads.celltoquad[row,col]]
end

function cell_sign(levelset,levelsetcoeffs,nodalconnectivity)
    ncells = size(nodalconnectivity)[2]
    cellsign = zeros(Int,ncells)
    box = IntervalBox(-1..1,2)
    for cellid in 1:ncells
        nodeids = nodalconnectivity[:,cellid]
        update!(levelset,levelsetcoeffs[nodeids])
        cellsign[cellid] = sign(levelset,box)
    end
    return cellsign
end

function CutMeshCellQuadratures(
    cellsign,
    levelset,
    levelsetcoeffs,
    nodalconnectivity,
    numqp,
)
    numcells = length(cellsign)
    @assert size(nodalconnectivity)[2] == numcells

    quad1d = ImplicitDomainQuadrature.ReferenceQuadratureRule(numqp)
    tpq = tensor_product_quadrature(2, numqp)
    quads = Vector{QuadratureRule{2}}(undef, 0)
    push!(quads,tpq)
    celltoquad = zeros(Int, 2, numcells)
    box = IntervalBox(-1..1,2)

    for cellid = 1:numcells
        if cellsign[cellid] == +1
            celltoquad[1,cellid] = 1
        elseif cellsign[cellid] == -1
            celltoquad[2,cellid] = 1
        elseif cellsign[cellid] == 0
            nodeids = nodalconnectivity[:,cellid]
            update!(levelset,levelsetcoeffs[nodeids])

            pquad = quadrature(levelset,+1,false,box,quad1d)
            push!(quads,pquad)
            celltoquad[1,cellid] = length(quads)

            nquad = quadrature(levelset,-1,false,box,quad1d)
            push!(quads,nquad)
            celltoquad[2,cellid] = length(quads)
        end
    end
    return CutMeshCellQuadratures(quads,celltoquad)
end

function active_node_ids(s,cellsign,nodalconnectivity)
    @assert s ∈ [-1,1]
    activenodeids = Int[]
    for cellid in 1:numcells
        if cellsign[cellid] == s || cellsign[cellid] == 0
            append!(activenodeids,nodalconnectivity[:,cellid])
        end
    end
    sort!(activenodeids)
    unique!(activenodeids)
    return activenodeids
end
