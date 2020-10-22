function assemble_bilinear_form!(
    sysmatrix::SystemMatrix,
    cutmeshbfs::BilinearForms,
    cutmesh,
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
    cutmesh::CutMesh;
    eta = 1,
)

    @assert eta == 1 || eta == 0 || eta == -1
    dofspernode = dimension(cutmesh)
    cellsign = cell_sign(cutmesh)

    cellids = findall(cellsign .== 0)
    for cellid in cellids
        negativenodeids = nodal_connectivity(cutmesh, -1, cellid)
        positivenodeids = nodal_connectivity(cutmesh, +1, cellid)

        negtop = -0.5 * vec(traction_operator(interfacecondition, -1, cellid))
        postop = -0.5 * vec(traction_operator(interfacecondition, +1, cellid))

        transnegtop =
            -0.5 *
            eta *
            vec(transpose(traction_operator(interfacecondition, -1, cellid)))
        transpostop =
            -0.5 *
            eta *
            vec(transpose(traction_operator(interfacecondition, +1, cellid)))

        assemble_cell_matrix!(sysmatrix, positivenodeids, dofspernode, -postop)
        assemble_cell_matrix!(sysmatrix, positivenodeids, dofspernode, -transpostop)

        assemble_couple_cell_matrix!(
            sysmatrix,
            positivenodeids,
            negativenodeids,
            dofspernode,
            -negtop,
        )
        assemble_couple_cell_matrix!(
            sysmatrix,
            positivenodeids,
            negativenodeids,
            dofspernode,
            transpostop,
        )

        assemble_cell_matrix!(sysmatrix, negativenodeids, dofspernode, negtop)
        assemble_cell_matrix!(sysmatrix, negativenodeids, dofspernode, transnegtop)

        assemble_couple_cell_matrix!(
            sysmatrix,
            negativenodeids,
            positivenodeids,
            dofspernode,
            postop,
        )
        assemble_couple_cell_matrix!(
            sysmatrix,
            negativenodeids,
            positivenodeids,
            dofspernode,
            -transnegtop,
        )

        posmass = vec(mass_operator(interfacecondition, +1, cellid))
        negmass = vec(mass_operator(interfacecondition, -1, cellid))

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

function assemble_body_force_linear_form!(
    systemrhs,
    rhsfunc,
    basis,
    cellquads::CellQuadratures,
    cutmesh::CutMesh,
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

function assemble_penalty_displacement_bc!(sysmatrix, sysrhs, dispcondition, cutmesh)

    ncells = number_of_cells(cutmesh)
    dim = dimension(cutmesh)

    for cellid = 1:ncells
        for faceid = 1:4
            for s in [-1, +1]
                if has_operator(dispcondition, s, faceid, cellid)
                    nodeids = nodal_connectivity(cutmesh, s, cellid)

                    mop = mass_operator(dispcondition, s, faceid, cellid)
                    top = traction_operator(dispcondition, s, faceid, cellid)
                    op = mop - top
                    assemble_cell_matrix!(sysmatrix, nodeids, dim, vec(op))

                    rhs = displacement_rhs(dispcondition, s, faceid, cellid)
                    assemble_cell_rhs!(sysrhs, nodeids, dim, rhs)
                end
            end
        end
    end
end

function assemble_penalty_displacement_component_bc!(
    sysmatrix,
    sysrhs,
    dispcondition,
    cutmesh;
    eta = 1.0,
)


end
