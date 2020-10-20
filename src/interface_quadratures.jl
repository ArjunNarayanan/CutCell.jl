struct InterfaceQuadratures
    positivequads::Any
    negativequads
    normals::Any
    celltoquad::Any
    ncells::Any
    function InterfaceQuadratures(positivequads, negativequads, normals, celltoquad)
        nphase,ncells = size(celltoquad)
        @assert all(celltoquad .>= 0)
        @assert length(positivequads) == length(negativequads) == length(normals)
        @assert all(celltoquad .<= length(positivequads))
        new(positivequads, negativequads, normals, celltoquad, ncells)
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

    positivequads = []
    negativequads = []
    normals = []
    celltoquad = zeros(Int, 2, numcells)

    counter = 0
    for cellid = 1:numcells
        if cellsign[cellid] == 0
            nodeids = nodalconnectivity[:, cellid]
            update!(levelset, levelsetcoeffs[nodeids])

            squad = surface_quadrature(levelset, box, quad1d)
            n = levelset_normal(levelset, squad.points, invjac)
            push!(positivequads, squad)
            push!(negativequads, squad)
            push!(normals, n)
            counter += 1

            celltoquad[cellid] = counter
        end
    end
    return InterfaceQuadratures(positivequads, negativequads, normals, celltoquad)
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

function Base.getindex(iquads::InterfaceQuadratures, s, cellid)
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
