struct InterfaceQuadratures
    quads::Any
    normals::Any
    celltoquad::Any
    ncells::Any
    function InterfaceQuadratures(quads, normals, celltoquad)
        ncells = length(celltoquad)
        nphase, nquads = size(quads)

        @assert nphase == 2
        @assert length(normals) == nquads
        @assert all(celltoquad .>= 0)
        @assert all(celltoquad .<= nquads)

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

    xL, xR = [-1.0, -1.0], [1.0, 1.0]
    invjac = inverse_jacobian(cellmap)
    quad1d = ImplicitDomainQuadrature.ReferenceQuadratureRule(numqp)

    hasinterface = cellsign .== 0
    numinterfaces = count(hasinterface)
    quads = Matrix{Any}(undef, 2, numinterfaces)
    normals = Vector{Any}(undef, numinterfaces)
    celltoquad = zeros(Int, numcells)

    counter = 1
    for cellid = 1:numcells
        if cellsign[cellid] == 0
            nodeids = nodalconnectivity[:, cellid]
            update!(levelset, levelsetcoeffs[nodeids])

            squad = surface_quadrature(levelset, xL, xR, quad1d)
            n = levelset_normal(levelset, squad.points, invjac)

            quads[1, counter] = squad
            quads[2, counter] = squad

            normals[counter] = n

            celltoquad[cellid] = counter

            counter += 1
        end
    end
    return InterfaceQuadratures(quads, normals, celltoquad)
end

function InterfaceQuadratures(levelset, levelsetcoeffs, cutmesh::CutMesh, numqp)
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
    row = cell_sign_to_row(s)
    idx = iquads.celltoquad[cellid]
    idx > 0 || error("Cell $cellid does not have an interface quadrature rule")
    return iquads.quads[row, idx]
end

function Base.show(io::IO, interfacequads::InterfaceQuadratures)
    ncells = interfacequads.ncells
    numinterfaces = length(interfacequads.normals)
    str = "InterfaceQuadratures\n\tNum. Cells: $ncells\n\tNum. Interfaces: $numinterfaces"
    print(io, str)
end

function interface_normals(iquads::InterfaceQuadratures, cellid)
    idx = iquads.celltoquad[cellid]
    return iquads.normals[idx]
end
