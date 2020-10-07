struct CutMeshBilinearForms
    cellmatrices::Any
    celltomatrix::Any
    ncells::Any
    function CutMeshBilinearForms(cellmatrices, celltomatrix)
        nphase, ncells = size(celltomatrix)
        @assert nphase == 2
        @assert all(celltomatrix .>= 0)
        @assert all(celltomatrix .<= length(cellmatrices))
        new(cellmatrices, celltomatrix, ncells)
    end
end

function Base.getindex(cbf::CutMeshBilinearForms, s, cellid)
    flag = (s == -1 || s == +1) && (1 <= cellid <= cbf.ncells)
    flag || throw(BoundsError(cbf.celltomatrix, [s, cellid]))
    row = s == +1 ? 1 : 2
    return cbf.cellmatrices[cbf.celltomatrix[row, cellid]]
end

function Base.getindex(cbf::CutMeshBilinearForms, s)
    flag = (s == -1 || s == +1)
    flag || error("Expected s ∈ {-1,1}, got s = $s")
    idx = s == +1 ? 1 : 2
    return cbf.cellmatrices[idx]
end

function CutMeshBilinearForms(
    basis,
    cutmeshquads,
    stiffnesses,
    cellsign,
    cellmap,
)
    @assert length(stiffnesses) == 2
    ncells = length(cellsign)

    uniformquad = uniform_cell_quadrature(cutmeshquads)
    uniformbf1 = bilinear_form(basis, uniformquad, stiffnesses[1], cellmap)
    uniformbf2 = bilinear_form(basis, uniformquad, stiffnesses[2], cellmap)

    cellmatrices = [uniformbf1, uniformbf2]
    celltomatrix = zeros(Int, 2, ncells)

    for cellid = 1:ncells
        if cellsign[cellid] == +1
            celltomatrix[1, cellid] = 1
        elseif cellsign[cellid] == -1
            celltomatrix[2, cellid] = 2
        else
            pquad = cutmeshquads[+1, cellid]
            pbf = bilinear_form(basis, pquad, stiffnesses[1], cellmap)
            push!(cellmatrices, pbf)
            celltomatrix[1, cellid] = length(cellmatrices)

            nquad = cutmeshquads[-1, cellid]
            nbf = bilinear_form(basis, nquad, stiffnesses[2], cellmap)
            push!(cellmatrices, nbf)
            celltomatrix[2, cellid] = length(cellmatrices)
        end
    end
    return CutMeshBilinearForms(cellmatrices, celltomatrix)
end

function CutMeshBilinearForms(basis, cutmeshquads, stiffnesses, cutmesh)
    cellsign = cell_sign(cutmesh)
    cellmap = cell_map(cutmesh, 1)
    return CutMeshBilinearForms(
        basis,
        cutmeshquads,
        stiffnesses,
        cellsign,
        cellmap,
    )
end

function assemble_bilinear_form!(
    sysmatrix::SystemMatrix,
    cutmeshbfs::CutMeshBilinearForms,
    cutmesh::CutMesh,
    dofspernode,
)

    ncells = number_of_cells(cutmesh)
    cellsign = cell_sign(cutmesh)

    uniformvals1 = vec(cutmeshbfs[+1])
    uniformvals2 = vec(cutmeshbfs[-1])

    for cellid = 1:ncells
        s = cellsign[cellid]
        if s == +1
            nodeids = nodal_connectivity(cutmesh, +1, cellid)
            assemble_cell_bilinear_form!(
                sysmatrix,
                nodeids,
                dofspernode,
                uniformvals1,
            )
        elseif s == -1
            nodeids = nodal_connectivity(cutmesh, -1, cellid)
            assemble_cell_bilinear_form!(
                sysmatrix,
                nodeids,
                dofspernode,
                uniformvals2,
            )
        elseif s == 0
            nodeids1 = nodal_connectivity(cutmesh, +1, cellid)
            vals1 = vec(cutmeshbfs[+1, cellid])
            assemble_cell_bilinear_form!(
                sysmatrix,
                nodeids1,
                dofspernode,
                vals1,
            )

            nodeids2 = nodal_connectivity(cutmesh, -1, cellid)
            vals2 = vec(cutmeshbfs[-1, cellid])
            assemble_cell_bilinear_form!(
                sysmatrix,
                nodeids2,
                dofspernode,
                vals2,
            )
        else
            error("Expected s ∈ {+1,0,-1}, received s = $s")
        end
    end
end

struct CutMeshInterfaceConstraints
    cellmatrices::Any
    celltomatrix::Any
    ncells::Any
    function CutMeshInterfaceConstraints(cellmatrices, celltomatrix)
        ncells = length(celltomatrix)
        @assert all(celltomatrix .>= 0)
        @assert all(celltomatrix .<= length(cellmatrices))
        new(cellmatrices, celltomatrix, ncells)
    end
end

function Base.getindex(ic::CutMeshInterfaceConstraints, cellid)
    1 <= cellid <= ic.ncells || throw(BoundsError(ic.celltomatrix, [cellid]))
    idx = ic.celltomatrix[cellid]
    idx > 0 || throw(BoundsError(ic.cellmatrices, [idx]))
    return ic.cellmatrices[idx]
end

function coherent_constraint_on_cells(
    basis,
    cutmeshinterfacequads,
    cellsign,
    cellmap,
    penalty,
)

    ncells = length(cellsign)
    cellmatrices = []
    celltomatrix = zeros(Int, ncells)

    for cellid = 1:ncells
        if cellsign[cellid] == 0
            squad = cutmeshinterfacequads[cellid]
            normals = interface_normals(cutmeshinterfacequads, cellid)

            scale = scale_area(cellmap, normals)
            matrix = penalty * mass_matrix(basis, squad, scale, 2)

            push!(cellmatrices, matrix)
            celltomatrix[cellid] = length(cellmatrices)
        end
    end
    return CutMeshInterfaceConstraints(cellmatrices, celltomatrix)
end

function coherent_constraint_on_cells(basis, interfacequads, cutmesh, penalty)
    cellsign = cell_sign(cutmesh)
    cellmap = cell_map(cutmesh, 1)
    return coherent_constraint_on_cells(
        basis,
        interfacequads,
        cellsign,
        cellmap,
        penalty,
    )
end

function assemble_interface_constraints!(
    sysmatrix::SystemMatrix,
    interfaceconstraints,
    cutmesh,
    dofspernode,
)

    ncells = number_of_cells(cutmesh)
    cellsign = cell_sign(cutmesh)

    for cellid = 1:ncells
        if cellsign[cellid] == 0
            constraintmatrix = interfaceconstraints[cellid]

            nodeids1 = nodal_connectivity(cutmesh, +1, cellid)
            nodeids2 = nodal_connectivity(cutmesh, -1, cellid)

            assemble_coherent_interface!(
                sysmatrix,
                constraintmatrix,
                nodeids1,
                nodeids2,
                2,
            )
        end
    end
end
