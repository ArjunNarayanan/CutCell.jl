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

function assemble_penalty_displacement_bc!(
    sysmatrix,
    sysrhs,
    dispcondition,
    cutmesh,
)

    ncells = number_of_cells(cutmesh)
    dim = dimension(cutmesh)

    for cellid in 1:ncells
        for faceid in 1:4
            for s in [-1,+1]
                if has_operator(dispcondition,s,faceid,cellid)
                    nodeids = nodal_connectivity(cutmesh,s,cellid)

                    mop = mass_operator(dispcondition,s,faceid,cellid)
                    top = traction_operator(dispcondition,s,faceid,cellid)
                    op = mop - top
                    assemble_cell_matrix!(sysmatrix,nodeids,dim,vec(op))

                    rhs = displacement_rhs(dispcondition,s,faceid,cellid)
                    assemble_cell_rhs!(sysrhs,nodeids,dim,rhs)
                end
            end
        end
    end
end

function assemble_penalty_displacement_component_bc!(
    sysmatrix,
    sysrhs,
    basis,
    facequads,
    stiffness,
    cutmesh,
    onboundary,
    component,
    bcval,
    penalty,
)

    cellids = findall(is_boundary_cell(cutmesh))
    cellconnectivity = cell_connectivity(cutmesh)
    facemidpoints = reference_face_midpoints()
    facenormals = reference_face_normals()
    nfaces = length(facemidpoints)
    cellsign = cell_sign(cutmesh)
    facedetjac = face_determinant_jacobian(cell_map(cutmesh, 1))
    dim = dimension(basis)

    for cellid in cellids
        cellmap = cell_map(cutmesh, cellid)
        s = cellsign[cellid]
        for faceid = 1:nfaces
            if cellconnectivity[faceid, cellid] == 0 &&
               onboundary(cellmap(facemidpoints[faceid]))

                if s == 0 || s == 1
                    pquad = facequads[+1, faceid, cellid]
                    tractionop = face_traction_component_operator(
                        basis,
                        pquad,
                        component,
                        facenormals[faceid],
                        stiffness[+1],
                        facedetjac[faceid],
                        cellmap,
                    )
                    displacementop =
                        penalty *
                        component_mass_matrix(basis, pquad, component, facedetjac[faceid])
                    rhs =
                        penalty * component_linear_form(
                            bcval,
                            basis,
                            pquad,
                            component,
                            facedetjac[faceid],
                        )

                    op = displacementop - tractionop
                    pnodeids = nodal_connectivity(cutmesh, +1, cellid)
                    assemble_cell_matrix!(sysmatrix, pnodeids, dim, vec(op))
                    assemble_cell_rhs!(sysrhs, pnodeids, dim, rhs)
                end

                if s == 0 || s == -1
                    nquad = facequads[-1, faceid, cellid]
                    tractionop = face_traction_component_operator(
                        basis,
                        nquad,
                        component,
                        facenormals[faceid],
                        stiffness[-1],
                        facedetjac[faceid],
                        cellmap,
                    )
                    displacementop =
                        penalty *
                        component_mass_matrix(basis, nquad, component, facedetjac[faceid])
                    rhs =
                        penalty * component_linear_form(
                            bcval,
                            basis,
                            nquad,
                            component,
                            facedetjac[faceid],
                        )

                    op = displacementop - tractionop
                    nnodeids = nodal_connectivity(cutmesh, -1, cellid)
                    assemble_cell_matrix!(sysmatrix, nnodeids, dim, vec(op))
                    assemble_cell_rhs!(sysrhs, nnodeids, dim, rhs)
                end

            end
        end
    end
end
