struct InterfaceQuadratures
    quads::Any
    normals::Any
    celltoquad::Any
    ncells::Any
    function InterfaceQuadratures(quads, normals, celltoquad)
        ncells = length(celltoquad)
        @assert all(celltoquad .>= 0)
        @assert all(celltoquad .<= length(quads))
        @assert length(quads) == length(normals)
        new(quads, normals, celltoquad, ncells)
    end
end

function InterfaceQuadratures(
    cellsign,
    levelset,
    levelsetcoeffs,
    nodalconnectivity,
    numqp,
    cellmap,
)

    numcells = length(cellsign)
    @assert size(nodalconnectivity)[2] == numcells

    box = IntervalBox(-1..1, 2)
    invjac = inverse_jacobian(cellmap)
    quad1d = ImplicitDomainQuadrature.ReferenceQuadratureRule(numqp)

    quads = []
    normals = []
    celltoquad = zeros(Int, numcells)

    for cellid = 1:numcells
        if cellsign[cellid] == 0
            nodeids = nodalconnectivity[:, cellid]
            update!(levelset, levelsetcoeffs[nodeids])

            squad = surface_quadrature(levelset, box, quad1d)
            push!(quads, squad)
            n = levelset_normal(levelset, squad.points, invjac)
            push!(normals, n)

            celltoquad[cellid] = length(quads)
        end
    end
    return InterfaceQuadratures(quads, normals, celltoquad)
end

function InterfaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
    cellsign = cell_sign(cutmesh)
    nodalconnectivity = nodal_connectivity(cutmesh.mesh)
    cellmap = cell_map(cutmesh, 1)
    return InterfaceQuadratures(
        cellsign,
        levelset,
        levelsetcoeffs,
        nodalconnectivity,
        numqp,
        cellmap,
    )
end

function Base.getindex(iquads::InterfaceQuadratures, cellid)
    1 <= cellid <= iquads.ncells ||
        throw(BoundsError(iquads.celltoquad, [cellid]))
    idx = iquads.celltoquad[cellid]
    return iquads.quads[idx]
end

function Base.show(io::IO, interfacequads::InterfaceQuadratures)
    ncells = interfacequads.ncells
    numinterfacequads = length(interfacequads.quads)
    str = "InterfaceQuadratures\n\tNum. Cells: $ncells\n\tNum. Interface Quadratures: $numinterfacequads"
    print(io, str)
end

function interface_normals(iquads::InterfaceQuadratures, cellid)
    1 <= cellid <= iquads.ncells ||
        throw(BoundsError(iquads.celltoquad, [cellid]))
    idx = iquads.celltoquad[cellid]
    return iquads.normals[idx]
end
