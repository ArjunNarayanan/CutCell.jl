struct CellMap
    yL
    yR
    jacobian
    dim
    function CellMap(yL,yR)
        dim = length(yL)
        @assert length(yR) == dim
        @assert all(yR .>= yL)
        jacobian = 0.5*(yR-yL)
        new(yL,yR,jacobian,dim)
    end
end

function dimension(C::CellMap)
    return C.dim
end

function jacobian(C::CellMap)
    return C.jacobian
end

function determinant_jacobian(C::CellMap)
    return prod(jacobian(C))
end

function (C::CellMap)(x)
    dim = dimension(C)
    @assert length(x) == dim
    return C.yL + (jacobian(C) .* (x + ones(dim)))
end

struct Mesh
    cellmaps
    nodalcoordinates
    nodalconnectivity
    cellconnectivity
    ncells
    nfmside
    nodesperelement
end

function Mesh(mesh,nf::Int)
    cellmaps = cell_maps(mesh)
    x0 = reference_corner(mesh)
    nelements = elements_per_mesh_side(mesh)
    nfeside = nodes_per_element_side(nf)
    nfmside = nodes_per_mesh_side(nelements,nfeside)

    nodalcoordinates = nodal_coordinates(x0,widths(mesh),nelements,nfmside)
    nodalconnectivity = nodal_connectivity(nfmside,nfeside,nf,nelements)
    cellconnectivity = cell_connectivity(mesh)
    ncells = number_of_elements(mesh)
    Mesh(cellmaps,nodalcoordinates,nodalconnectivity,
        cellconnectivity,ncells,nfmside,nf)
end

function nodes_per_mesh_side(mesh::Mesh)
    return mesh.nfmside
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

function Mesh(mesh,basis)
    nf = number_of_basis_functions(basis)
    return Mesh(mesh,nf)
end

function elements_per_mesh_side(mesh::UniformMesh)
    return mesh.nelements
end

function cell_maps(mesh)
    ncells = number_of_elements(mesh)
    return [CellMap(element(mesh,i)...) for i = 1:ncells]
end

function cell_maps(mesh::Mesh)
    return mesh.cellmaps
end

function cellmap(mesh::Mesh,i)
    return mesh.cellmaps[i]
end

function nodes_per_mesh_side(nelements,nfeside)
    return (nfeside-1)*nelements .+ 1
end

function nodes_per_element_side(nodesperelement)
    nfeside = sqrt(nodesperelement)
    @assert isinteger(nfeside)
    return round(Int,nfeside)
end

function nodal_coordinates(x0,widths,nelements,nfmside)
    @assert length(x0) == length(widths) == length(nelements) == 2

    xrange = range(x0[1],stop=x0[1]+widths[1],length=nfmside[1])
    yrange = range(x0[2],stop=x0[2]+widths[2],length=nfmside[2])

    ycoords = repeat(yrange,outer=nfmside[1])
    xcoords = repeat(xrange,inner=nfmside[2])
    return vcat(xcoords',ycoords')
end

function nodal_coordinates(mesh::Mesh)
    return mesh.nodalcoordinates
end

function nodal_connectivity(mesh::Mesh)
    return mesh.nodalconnectivity
end

function dimension(mesh::UniformMesh)
    return CartesianMesh.dimension(mesh)
end

function nodal_connectivity(nfmside,nfeside,nodesperelement,nelements)
    numnodes = prod(nfmside)
    ncells = prod(nelements)

    nodeids = reshape(1:numnodes,reverse(nfmside)...)
    connectivity = zeros(Int,nodesperelement,ncells)
    colstart = 1
    colend = nfeside

    cellid = 1
    for col in 1:nelements[1]
        rowstart = 1
        rowend = nfeside
        for row in 1:nelements[2]
            connectivity[:,cellid] .= vec(nodeids[rowstart:rowend,colstart:colend])
            cellid += 1
            rowstart = rowend
            rowend += nfeside-1
        end
        colstart = colend
        colend += nfeside-1
    end
    return connectivity
end

function faces_per_cell(mesh::UniformMesh)
    return CartesianMesh.faces_per_cell(mesh)
end

function cell_connectivity(mesh)
    ncells = number_of_elements(mesh)
    nfaces = faces_per_cell(mesh)
    connectivity = zeros(Int,nfaces,ncells)
    for cellid = 1:ncells
        connectivity[:,cellid] .= neighbors(mesh,cellid)
    end
    return connectivity
end

function bottom_boundary_node_ids(mesh)
    nfmside = nodes_per_mesh_side(mesh)
    return range(1,step=nfmside[2],length=nfmside[1])
end

function right_boundary_node_ids(mesh)
    nfmside = nodes_per_mesh_side(mesh)
    start = nfmside[2]*(nfmside[1]-1)+1
    stop = nfmside[1]*nfmside[2]
    return start:stop
end

function top_boundary_node_ids(mesh)
    nfmside = nodes_per_mesh_side(mesh)
    return range(nfmside[2],step=nfmside[2],length=nfmside[1])
end

function left_boundary_node_ids(mesh)
    nfmside = nodes_per_mesh_side(mesh)
    return 1:nfmside[2]
end

function number_of_degrees_of_freedom(mesh::Mesh)
    dim = size(mesh.nodalcoordinates)[1]
    numnodes = prod(mesh.nfmside)
    return numnodes*dim
end
