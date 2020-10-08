struct InterfaceConstraints
    cellmatrices::Any
    celltomatrix::Any
    ncells::Any
    function InterfaceConstraints(cellmatrices, celltomatrix)
        ncells = length(celltomatrix)
        @assert all(celltomatrix .>= 0)
        @assert all(celltomatrix .<= length(cellmatrices))
        new(cellmatrices, celltomatrix, ncells)
    end
end

function Base.getindex(ic::InterfaceConstraints, cellid)
    idx = ic.celltomatrix[cellid]
    return ic.cellmatrices[idx]
end

function Base.show(io::IO, ic::InterfaceConstraints)
    ncells = ic.ncells
    nmatrix = length(ic.cellmatrices)
    str = "InterfaceConstraints\n\tNum. Cells: $ncells\n\tNum. Unique Constraint Matrices: $nmatrix"
    print(io,str)
end

function assemble_cell_interface_constraint!(
    sysmatrix::SystemMatrix,
    cellmatrix,
    nodeids1,
    nodeids2,
    dofspernode,
)

    edofs1 = element_dofs(nodeids1,dofspernode)
    edofs2 = element_dofs(nodeids2,dofspernode)
    vals = vec(cellmatrix)

    r1,c1 = element_dofs_to_operator_dofs(edofs1,edofs1)
    assemble!(sysmatrix,r1,c1,vals)
    r2,c2 = element_dofs_to_operator_dofs(edofs1,edofs2)
    assemble!(sysmatrix,r2,c2,-vals)
    r3,c3 = element_dofs_to_operator_dofs(edofs2,edofs1)
    assemble!(sysmatrix,r3,c3,-vals)
    r4,c4 = element_dofs_to_operator_dofs(edofs2,edofs2)
    assemble!(sysmatrix,r4,c4,vals)
end

function assemble_interface_constraints!(
    sysmatrix::SystemMatrix,
    interfaceconstraints,
    cutmesh,
)

    dofspernode = dimension(cutmesh)
    ncells = number_of_cells(cutmesh)
    cellsign = cell_sign(cutmesh)

    for cellid = 1:ncells
        if cellsign[cellid] == 0
            constraintmatrix = interfaceconstraints[cellid]

            nodeids1 = nodal_connectivity(cutmesh, +1, cellid)
            nodeids2 = nodal_connectivity(cutmesh, -1, cellid)

            assemble_cell_interface_constraint!(
                sysmatrix,
                constraintmatrix,
                nodeids1,
                nodeids2,
                2,
            )
        end
    end
end

function coherent_interface_constraint(
    basis,
    interfacequads,
    cellsign,
    cellmap,
    penalty,
)

    ncells = length(cellsign)
    cellmatrices = []
    celltomatrix = zeros(Int, ncells)

    for cellid = 1:ncells
        if cellsign[cellid] == 0
            squad = interfacequads[cellid]
            normals = interface_normals(interfacequads, cellid)

            scale = scale_area(cellmap, normals)
            matrix = penalty * mass_matrix(basis, squad, scale, 2)

            push!(cellmatrices, matrix)
            celltomatrix[cellid] = length(cellmatrices)
        end
    end
    return InterfaceConstraints(cellmatrices, celltomatrix)
end

function coherent_interface_constraint(basis, interfacequads, cutmesh, penalty)
    cellsign = cell_sign(cutmesh)
    cellmap = cell_map(cutmesh, 1)
    return coherent_interface_constraint(
        basis,
        interfacequads,
        cellsign,
        cellmap,
        penalty,
    )
end
