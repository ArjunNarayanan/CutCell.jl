function assemble_bilinear_form!(
    sysmatrix::SystemMatrix,
    cutmeshbfs::BilinearForms,
    cutmesh::CutMesh,
)

    dofspernode = dimension(cutmesh)
    ncells = number_of_cells(cutmesh)
    cellsign = cell_sign(cutmesh)

    uniformvals1 = vec(cutmeshbfs[+1])
    uniformvals2 = vec(cutmeshbfs[-1])

    for cellid = 1:ncells
        s = cellsign[cellid]
        if s == +1
            nodeids = nodal_connectivity(cutmesh, +1, cellid)
            assemble_cell_matrix!(sysmatrix, nodeids, dofspernode, uniformvals1)
        elseif s == -1
            nodeids = nodal_connectivity(cutmesh, -1, cellid)
            assemble_cell_matrix!(sysmatrix, nodeids, dofspernode, uniformvals2)
        elseif s == 0
            nodeids1 = nodal_connectivity(cutmesh, +1, cellid)
            vals1 = vec(cutmeshbfs[+1, cellid])
            assemble_cell_matrix!(sysmatrix, nodeids1, dofspernode, vals1)

            nodeids2 = nodal_connectivity(cutmesh, -1, cellid)
            vals2 = vec(cutmeshbfs[-1, cellid])
            assemble_cell_matrix!(sysmatrix, nodeids2, dofspernode, vals2)
        else
            error("Expected cellsign ∈ {+1,0,-1}, got cellsign = $s")
        end
    end
end

function assemble_interface_condition!(
    sysmatrix::SystemMatrix,
    interfacecondition::InterfaceCondition,
    cutmesh::CutMesh,
)

    dofspernode = dimension(cutmesh)
    cellsign = cell_sign(cutmesh)

    for (cellid, s) in enumerate(cellsign)
        if s == 0
            negativenodeids = nodal_connectivity(cutmesh, -1, cellid)
            positivenodeids = nodal_connectivity(cutmesh, +1, cellid)

            negativetractionop = traction_operator(interfacecondition, -1, cellid)
            assemble_couple_cell_matrix!(
                sysmatrix,
                negativenodeids,
                positivenodeids,
                dofspernode,
                vec(negativetractionop),
            )
            positivetractionop = traction_operator(interfacecondition, +1, cellid)
            assemble_couple_cell_matrix!(
                sysmatrix,
                positivenodeids,
                negativenodeids,
                dofspernode,
                vec(positivetractionop),
            )

            mass = vec(mass_operator(interfacecondition, cellid))
            assemble_cell_matrix!(sysmatrix, negativenodeids, dofspernode, mass)
            assemble_cell_matrix!(sysmatrix, positivenodeids, dofspernode, mass)
            assemble_couple_cell_matrix!(
                sysmatrix,
                negativenodeids,
                positivenodeids,
                dofspernode,
                -mass,
            )
            assemble_couple_cell_matrix!(
                sysmatrix,
                positivenodeids,
                negativenodeids,
                dofspernode,
                -mass,
            )
        end
    end
end

function assemble_cut_mesh_body_force_linear_form!(
    systemrhs,
    rhsfunc,
    basis,
    cellquads,
    cutmesh,
)

    ncells = number_of_cells(cutmesh)
    cellsign = cell_sign(cutmesh)
    dofspernode = dimension(cutmesh)

    for cellid = 1:ncells
        s = cellsign[cellid]
        @assert s == -1 || s == 0 || s == 1
        cellmap = cell_map(cutmesh, cellid)
        if s == +1 || s == 0
            pquad = cellquads[+1, cellid]
            rhs = linear_form(rhsfunc, basis, pquad, cellmap)
            nodeids = nodal_connectivity(cutmesh, +1, cellid)
            edofs = element_dofs(nodeids, dofspernode)
            assemble!(systemrhs, edofs, rhs)
        end
        if s == -1 || s == 0
            nquad = cellquads[-1, cellid]
            rhs = linear_form(rhsfunc, basis, nquad, cellmap)
            nodeids = nodal_connectivity(cutmesh, -1, cellid)
            edofs = element_dofs(nodeids, dofspernode)
            assemble!(systemrhs, edofs, rhs)
        end
    end
end
