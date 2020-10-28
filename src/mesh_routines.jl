struct Mesh
    dim::Any
    cellmaps::Any
    nodalcoordinates::Any
    nodalconnectivity::Any
    cellconnectivity::Any
    ncells::Any
    nfmside::Any
    totalnodes::Any
    nodesperelement::Any
    isinteriorcell::Any
end

function Mesh(mesh, nf::Int)
    dim = dimension(mesh)
    cellmaps = cell_maps(mesh)
    x0 = reference_corner(mesh)
    nelements = elements_per_mesh_side(mesh)
    nfeside = nodes_per_element_side(nf)
    nfmside = nodes_per_mesh_side(nelements, nfeside)
    totalnodes = prod(nfmside)

    nodalcoordinates = nodal_coordinates(x0, widths(mesh), nelements, nfmside)
    nodalconnectivity = nodal_connectivity(nfmside, nfeside, nf, nelements)
    cellconnectivity = cell_connectivity(mesh)
    isinteriorcell = is_interior_cell(cellconnectivity)
    ncells = number_of_elements(mesh)
    Mesh(
        dim,
        cellmaps,
        nodalcoordinates,
        nodalconnectivity,
        cellconnectivity,
        ncells,
        nfmside,
        totalnodes,
        nf,
        isinteriorcell,
    )
end

function Mesh(mesh, basis)
    nf = number_of_basis_functions(basis)
    return Mesh(mesh, nf)
end

function Mesh(x0, widths, nelements, nf::Int)
    mesh = UniformMesh(x0, widths, nelements)
    return Mesh(mesh, nf)
end

function Mesh(x0, widths, nelements, basis)
    nf = number_of_basis_functions(basis)
    return Mesh(x0, widths, nelements, nf)
end

function Base.show(io::IO, mesh::Mesh)
    ncells = number_of_cells(mesh)
    numnodes = number_of_nodes(mesh)
    dim = dimension(mesh)
    nf = nodes_per_element(mesh)
    str = "Mesh\n\tDimension : $dim\n\tNum. Cells: $ncells\n\tNum. Nodes: $numnodes\n\tNodes/Cell: $nf"
    print(io, str)
end

function number_of_cells(mesh::Mesh)
    return mesh.ncells
end

function nodes_per_mesh_side(mesh::Mesh)
    return mesh.nfmside
end

function number_of_nodes(mesh::Mesh)
    return mesh.totalnodes
end

function nodes_per_element(mesh::Mesh)
    return mesh.nodesperelement
end

function reference_corner(mesh::UniformMesh)
    return mesh.x0
end

function widths(mesh::UniformMesh)
    return mesh.widths
end

function elements_per_mesh_side(mesh::UniformMesh)
    return mesh.nelements
end

function cell_maps(mesh)
    ncells = number_of_elements(mesh)
    return [CellMap(element(mesh, i)...) for i = 1:ncells]
end

function cell_maps(mesh::Mesh)
    return mesh.cellmaps
end

function cell_map(mesh::Mesh, i)
    return mesh.cellmaps[i]
end

function determinant_jacobian(mesh::Mesh)
    return determinant_jacobian(cell_map(mesh,1))
end

function jacobian(mesh::Mesh)
    return jacobian(cell_map(mesh,1))
end

function face_determinant_jacobian(mesh::Mesh)
    return face_determinant_jacobian(cell_map(mesh, 1))
end

function nodes_per_mesh_side(nelements, nfeside)
    return (nfeside - 1) * nelements .+ 1
end

function nodes_per_element_side(nodesperelement)
    nfeside = sqrt(nodesperelement)
    @assert isinteger(nfeside)
    return round(Int, nfeside)
end

function nodal_coordinates(x0, widths, nelements, nfmside)
    @assert length(x0) == length(widths) == length(nelements) == 2

    xrange = range(x0[1], stop = x0[1] + widths[1], length = nfmside[1])
    yrange = range(x0[2], stop = x0[2] + widths[2], length = nfmside[2])

    ycoords = repeat(yrange, outer = nfmside[1])
    xcoords = repeat(xrange, inner = nfmside[2])
    return vcat(xcoords', ycoords')
end

function nodal_coordinates(mesh::Mesh)
    return mesh.nodalcoordinates
end

function nodal_connectivity(mesh::Mesh)
    return mesh.nodalconnectivity
end

function dimension(mesh::Mesh)
    return mesh.dim
end

function dimension(mesh::UniformMesh)
    return CartesianMesh.dimension(mesh)
end

function faces_per_cell(mesh::UniformMesh)
    return CartesianMesh.faces_per_cell(mesh)
end

function number_of_elements(mesh::UniformMesh)
    return CartesianMesh.number_of_elements(mesh)
end

function nodal_connectivity(nfmside, nfeside, nodesperelement, nelements)
    numnodes = prod(nfmside)
    ncells = prod(nelements)

    nodeids = reshape(1:numnodes, reverse(nfmside)...)
    connectivity = zeros(Int, nodesperelement, ncells)
    colstart = 1
    colend = nfeside

    cellid = 1
    for col = 1:nelements[1]
        rowstart = 1
        rowend = nfeside
        for row = 1:nelements[2]
            connectivity[:, cellid] .= vec(nodeids[rowstart:rowend, colstart:colend])
            cellid += 1
            rowstart = rowend
            rowend += nfeside - 1
        end
        colstart = colend
        colend += nfeside - 1
    end
    return connectivity
end

function cell_connectivity(mesh)
    ncells = number_of_elements(mesh)
    nfaces = faces_per_cell(mesh)
    connectivity = zeros(Int, nfaces, ncells)
    for cellid = 1:ncells
        connectivity[:, cellid] .= neighbors(mesh, cellid)
    end
    return connectivity
end

function cell_connectivity(mesh::Mesh)
    return mesh.cellconnectivity
end

function bottom_boundary_node_ids(mesh)
    nfmside = nodes_per_mesh_side(mesh)
    return range(1, step = nfmside[2], length = nfmside[1])
end

function right_boundary_node_ids(mesh)
    nfmside = nodes_per_mesh_side(mesh)
    start = nfmside[2] * (nfmside[1] - 1) + 1
    stop = nfmside[1] * nfmside[2]
    return start:stop
end

function top_boundary_node_ids(mesh)
    nfmside = nodes_per_mesh_side(mesh)
    return range(nfmside[2], step = nfmside[2], length = nfmside[1])
end

function left_boundary_node_ids(mesh)
    nfmside = nodes_per_mesh_side(mesh)
    return 1:nfmside[2]
end

function number_of_degrees_of_freedom(mesh::Mesh)
    dim = dimension(mesh)
    numnodes = number_of_nodes(mesh)
    return numnodes * dim
end

function boundary_node_ids(mesh::Mesh)
    nodeids = vcat(
        bottom_boundary_node_ids(mesh),
        right_boundary_node_ids(mesh),
        top_boundary_node_ids(mesh),
        left_boundary_node_ids(mesh),
    )
    return unique!(sort!(nodeids))
end

function is_interior_cell(cellconnectivity)
    numcells = size(cellconnectivity)[2]
    isinteriorcell = [all(cellconnectivity[:, i] .!= 0) for i = 1:numcells]
    return isinteriorcell
end

function is_interior_cell(mesh::Mesh)
    return mesh.isinteriorcell
end

function is_boundary_cell(x)
    return .!is_interior_cell(x)
end
