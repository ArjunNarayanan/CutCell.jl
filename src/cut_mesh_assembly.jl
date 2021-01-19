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
    cutmesh;
    eta = 1,
)

    @assert eta == 1 || eta == 0 || eta == -1
    dofspernode = dimension(cutmesh)
    cellsign = cell_sign(cutmesh)

    cellids = findall(cellsign .== 0)
    for cellid in cellids
        negativenodeids = nodal_connectivity(cutmesh, -1, cellid)
        positivenodeids = nodal_connectivity(cutmesh, +1, cellid)

        pptop =
            -0.5 * vec(traction_operator(interfacecondition, +1, +1, cellid))
        pntop =
            -0.5 * vec(traction_operator(interfacecondition, +1, -1, cellid))
        nptop =
            -0.5 * vec(traction_operator(interfacecondition, -1, +1, cellid))
        nntop =
            -0.5 * vec(traction_operator(interfacecondition, -1, -1, cellid))

        pptopT =
            -0.5 *
            eta *
            vec(
                transpose(
                    traction_operator(interfacecondition, +1, +1, cellid),
                ),
            )
        pntopT =
            -0.5 *
            eta *
            vec(
                transpose(
                    traction_operator(interfacecondition, +1, -1, cellid),
                ),
            )
        nptopT =
            -0.5 *
            eta *
            vec(
                transpose(
                    traction_operator(interfacecondition, -1, +1, cellid),
                ),
            )
        nntopT =
            -0.5 *
            eta *
            vec(
                transpose(
                    traction_operator(interfacecondition, -1, -1, cellid),
                ),
            )

        assemble_cell_matrix!(sysmatrix, positivenodeids, dofspernode, -pptop)
        assemble_cell_matrix!(sysmatrix, positivenodeids, dofspernode, -pptopT)

        assemble_couple_cell_matrix!(
            sysmatrix,
            positivenodeids,
            negativenodeids,
            dofspernode,
            -pntop,
        )
        assemble_couple_cell_matrix!(
            sysmatrix,
            positivenodeids,
            negativenodeids,
            dofspernode,
            nptopT,
        )

        assemble_cell_matrix!(sysmatrix, negativenodeids, dofspernode, nntop)
        assemble_cell_matrix!(sysmatrix, negativenodeids, dofspernode, nntopT)

        assemble_couple_cell_matrix!(
            sysmatrix,
            negativenodeids,
            positivenodeids,
            dofspernode,
            nptop,
        )
        assemble_couple_cell_matrix!(
            sysmatrix,
            negativenodeids,
            positivenodeids,
            dofspernode,
            -pntopT,
        )

        ppmass = vec(mass_operator(interfacecondition, +1, +1, cellid))
        pnmass = vec(mass_operator(interfacecondition, +1, -1, cellid))
        npmass = vec(mass_operator(interfacecondition, -1, +1, cellid))
        nnmass = vec(mass_operator(interfacecondition, -1, -1, cellid))

        assemble_cell_matrix!(sysmatrix, negativenodeids, dofspernode, nnmass)
        assemble_cell_matrix!(sysmatrix, positivenodeids, dofspernode, ppmass)
        assemble_couple_cell_matrix!(
            sysmatrix,
            negativenodeids,
            positivenodeids,
            dofspernode,
            -npmass,
        )
        assemble_couple_cell_matrix!(
            sysmatrix,
            positivenodeids,
            negativenodeids,
            dofspernode,
            -pnmass,
        )
    end
end

function assemble_face_interelement_condition!(
    sysmatrix,
    nodeids1,
    nodeids2,
    dofspernode,
    tractionop,
    massop,
)
    assemble_cell_matrix!(sysmatrix, nodeids1, dofspernode, +tractionop.nn)
    assemble_cell_matrix!(sysmatrix, nodeids1, dofspernode, +tractionop.nnT)
    assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids1,
        nodeids2,
        dofspernode,
        +tractionop.np,
    )
    assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids1,
        nodeids2,
        dofspernode,
        -tractionop.pnT,
    )

    assemble_cell_matrix!(sysmatrix, nodeids2, dofspernode, -tractionop.pp)
    assemble_cell_matrix!(sysmatrix, nodeids2, dofspernode, -tractionop.ppT)
    assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids2,
        nodeids1,
        dofspernode,
        -tractionop.pn,
    )
    assemble_couple_cell_matrix!(
        sysmatrix,
        nodeids2,
        nodeids1,
        dofspernode,
        +tractionop.npT,
    )
end

function assemble_interelement_condition!(
    sysmatrix,
    basis,
    facequads,
    stiffness,
    cutmesh,
    penalty;
    eta = 1,
)

    @assert eta == 1 || eta == 0 || eta == -1
    uniformquads = uniform_face_quadratures(facequads)
    normals = reference_face_normals()
    facedetjac = face_determinant_jacobian(cutmesh)
    jac = jacobian(cutmesh)
    nfaces = length(normals)
    dim = dimension(cutmesh)

    faceids = 1:nfaces
    nbrfaceids = [opposite_face(faceid) for faceid in faceids]

    uniformtop1 = [
        interelement_traction_operators(
            basis,
            uniformquads[faceid],
            uniformquads[nbrfaceids[faceid]],
            normals[faceid],
            stiffness[+1],
            facedetjac[faceid],
            jac,
            eta,
        ) for faceid in faceids
    ]
    uniformtop2 = [
        interelement_traction_operators(
            basis,
            uniformquads[faceid],
            uniformquads[nbrfaceids[faceid]],
            normals[faceid],
            stiffness[-1],
            facedetjac[faceid],
            jac,
            eta,
        ) for faceid in faceids
    ]

    uniformtop = [uniformtop1, uniformtop2]
    uniformmassop = [
        interelement_mass_operators(
            basis,
            uniformquads[faceid],
            uniformquads[nbrfaceids[faceid]],
            penalty * facedetjac[faceid],
        ) for faceid in faceids
    ]

    ncells = number_of_cells(cutmesh)

    for cellid = 1:ncells
        s = cell_sign(cutmesh, cellid)
        @assert s == +1 || s == -1 || s == 0
        if s == +1 || s == -1
            nodeids1 = nodal_connectivity(cutmesh, s, cellid)
            row = cell_sign_to_row(s)

            for faceid in faceids
                nbrcellid = cell_connectivity(cutmesh, faceid, cellid)
                if cellid < nbrcellid
                    nodeids2 = nodal_connectivity(cutmesh, s, nbrcellid)

                    assemble_face_interelement_condition!(
                        sysmatrix,
                        nodeids1,
                        nodeids2,
                        dim,
                        uniformtop[row][faceid],
                        uniformmassop[faceid],
                    )
                end
            end

        else
            error("Cut Cell Assembly not implemented yet")
        end
    end
end

function assemble_body_force_linear_form!(
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
        if s == +1 || s == 0
            cellmap = cell_map(cutmesh, +1, cellid)
            pquad = cellquads[+1, cellid]
            rhs = linear_form(rhsfunc, basis, pquad, cellmap)
            nodeids = nodal_connectivity(cutmesh, +1, cellid)
            edofs = element_dofs(nodeids, dofspernode)
            assemble!(systemrhs, edofs, rhs)
        end
        if s == -1 || s == 0
            cellmap = cell_map(cutmesh, -1, cellid)
            nquad = cellquads[-1, cellid]
            rhs = linear_form(rhsfunc, basis, nquad, cellmap)
            nodeids = nodal_connectivity(cutmesh, -1, cellid)
            edofs = element_dofs(nodeids, dofspernode)
            assemble!(systemrhs, edofs, rhs)
        end
    end
end

function assemble_traction_force_linear_form!(
    systemrhs,
    tractionfunc,
    basis,
    facequads,
    cutmesh,
    onboundary,
)

    dim = dimension(cutmesh)
    facemidpoints = reference_face_midpoints()
    cellconnectivity = cell_connectivity(cutmesh)
    cellids = findall(is_boundary_cell(cutmesh))
    nfaces = length(facemidpoints)
    facedetjac = face_determinant_jacobian(cutmesh)

    refnormals = reference_face_normals()

    for cellid in cellids
        cellmap = cell_map(cutmesh, cellid)
        s = cell_sign(cutmesh, cellid)
        for faceid = 1:nfaces
            if cellconnectivity[faceid, cellid] == 0
                if onboundary(cellmap(facemidpoints[faceid]))
                    if s == 0 || s == +1
                        rhs = linear_form(
                            tractionfunc,
                            basis,
                            facequads[+1, faceid, cellid],
                            cellmap,
                            facedetjac[faceid],
                        )
                        nodeids = nodal_connectivity(cutmesh, +1, cellid)
                        edofs = element_dofs(nodeids, dim)
                        assemble!(systemrhs, edofs, rhs)
                    end
                    if s == 0 || s == -1
                        rhs = linear_form(
                            tractionfunc,
                            basis,
                            facequads[-1, faceid, cellid],
                            cellmap,
                            facedetjac[faceid],
                        )
                        nodeids = nodal_connectivity(cutmesh, -1, cellid)
                        edofs = element_dofs(nodeids, dim)
                        assemble!(systemrhs, edofs, rhs)
                    end
                end
            end
        end
    end
end

function assemble_traction_force_component_linear_form!(
    systemrhs,
    tractionfunc,
    basis,
    facequads,
    cutmesh,
    onboundary,
    component,
)

    dim = dimension(cutmesh)
    facemidpoints = reference_face_midpoints()
    cellconnectivity = cell_connectivity(cutmesh)
    cellids = findall(is_boundary_cell(cutmesh))
    nfaces = length(facemidpoints)
    facedetjac = face_determinant_jacobian(cutmesh)

    refnormals = reference_face_normals()

    for cellid in cellids
        cellmap = cell_map(cutmesh, cellid)
        s = cell_sign(cutmesh, cellid)
        for faceid = 1:nfaces
            if cellconnectivity[faceid, cellid] == 0
                if onboundary(cellmap(facemidpoints[faceid]))
                    if s == 0 || s == +1
                        rhs = component_linear_form(
                            tractionfunc,
                            basis,
                            facequads[+1, faceid, cellid],
                            component,
                            cellmap,
                            facedetjac[faceid],
                        )
                        nodeids = nodal_connectivity(cutmesh, +1, cellid)
                        edofs = element_dofs(nodeids, dim)
                        assemble!(systemrhs, edofs, rhs)
                    end
                    if s == 0 || s == -1
                        rhs = component_linear_form(
                            tractionfunc,
                            basis,
                            facequads[-1, faceid, cellid],
                            component,
                            cellmap,
                            facedetjac[faceid],
                        )
                        nodeids = nodal_connectivity(cutmesh, -1, cellid)
                        edofs = element_dofs(nodeids, dim)
                        assemble!(systemrhs, edofs, rhs)
                    end
                end
            end
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

function assemble_bulk_transformation_linear_form!(
    systemrhs,
    transfstress,
    basis,
    cellquads,
    cutmesh,
)

    ncells = number_of_cells(cutmesh)
    cellsign = cell_sign(cutmesh)
    dofspernode = dimension(cutmesh)
    jac = jacobian(cutmesh)

    for cellid = 1:ncells
        s = cellsign[cellid]
        @assert s == -1 || s == 0 || s == +1
        if s == +1 || s == 0
            pquad = cellquads[+1, cellid]
            rhs = bulk_transformation_rhs(basis, pquad, transfstress, jac)
            nodeids = nodal_connectivity(cutmesh, +1, cellid)
            edofs = element_dofs(nodeids, dofspernode)
            assemble!(systemrhs, edofs, rhs)
        end
    end
end

function assemble_coherent_interface_transformation_rhs!(
    systemrhs,
    transfstress,
    basis,
    interfacequads,
    cutmesh,
)

    cellsign = cell_sign(cutmesh)
    dofspernode = dimension(cutmesh)

    cellids = findall(cellsign .== 0)
    for cellid in cellids
        cellmap = cell_map(cutmesh, cellid)
        normals = interface_normals(interfacequads, cellid)

        for s in [+1, -1]
            quad = interfacequads[s, cellid]
            rhs =
                s *
                0.5 *
                interface_transformation_rhs(
                    basis,
                    quad,
                    normals,
                    transfstress,
                    cellmap,
                )
            nodeids = nodal_connectivity(cutmesh, s, cellid)
            assemble_cell_rhs!(systemrhs, nodeids, dofspernode, rhs)
        end

    end
end

function assemble_incoherent_interface_transformation_rhs!(
    systemrhs,
    transfstress,
    basis,
    interfacequads,
    cutmesh,
)

    cellsign = cell_sign(cutmesh)
    dofspernode = dimension(cutmesh)

    cellids = findall(cellsign .== 0)
    for cellid in cellids
        cellmap = cell_map(cutmesh, cellid)
        normals = interface_normals(interfacequads, cellid)
        components = normals

        for s in [+1, -1]
            quad = interfacequads[s, cellid]
            rhs =
                s *
                0.5 *
                interface_transformation_component_rhs(
                    basis,
                    quad,
                    components,
                    normals,
                    transfstress,
                    cellmap,
                )
            nodeids = nodal_connectivity(cutmesh, s, cellid)
            assemble_cell_rhs!(systemrhs, nodeids, dofspernode, rhs)
        end

    end
end

function assemble_stress_mass_matrix!(sysmatrix, basis, cellquads, mesh)
    ncells = number_of_cells(mesh)
    detjac = determinant_jacobian(mesh)
    dim = dimension(basis)
    sdim = number_of_symmetric_degrees_of_freedom(dim)

    uniformquad = uniform_cell_quadrature(cellquads)
    uniformcellvals = vec(stress_cell_mass_matrix(basis, uniformquad, detjac))

    for cellid = 1:ncells
        s = cell_sign(mesh, cellid)
        @assert s == -1 || s == 0 || s == +1
        if s == -1 || s == +1
            nodeids = nodal_connectivity(mesh, s, cellid)
            assemble_cell_matrix!(sysmatrix, nodeids, sdim, uniformcellvals)
        else
            pquad = cellquads[+1, cellid]
            pvals = vec(stress_cell_mass_matrix(basis, pquad, detjac))
            pnodeids = nodal_connectivity(mesh, +1, cellid)
            assemble_cell_matrix!(sysmatrix, pnodeids, sdim, pvals)

            nquad = cellquads[-1, cellid]
            nvals = vec(stress_cell_mass_matrix(basis, nquad, detjac))
            nnodeids = nodal_connectivity(mesh, -1, cellid)
            assemble_cell_matrix!(sysmatrix, nnodeids, sdim, nvals)
        end
    end
end

function assemble_stress_linear_form!(
    sysrhs,
    basis,
    cellquads::CellQuadratures,
    stiffness,
    nodaldisplacement,
    mesh,
)

    dim = dimension(basis)
    sdim = number_of_symmetric_degrees_of_freedom(dim)
    ncells = number_of_cells(mesh)
    jac = jacobian(mesh)
    uniformquad = uniform_cell_quadrature(cellquads)

    uniformop = [
        stress_cell_rhs_operator(basis, uniformquad, stiffness[s], jac) for
        s in [+1, -1]
    ]

    for cellid = 1:ncells
        s = cell_sign(mesh, cellid)
        @assert s == -1 || s == 0 || s == +1
        if s == -1 || s == +1
            nodeids = nodal_connectivity(mesh, s, cellid)
            dispdofs = element_dofs(nodeids, dim)

            idx = cell_sign_to_row(s)
            op = uniformop[idx]
            vals = op * nodaldisplacement[dispdofs]

            stressdofs = element_dofs(nodeids, sdim)
            assemble!(sysrhs, stressdofs, vals)
        else
            pnodeids = nodal_connectivity(mesh, +1, cellid)
            pdispdofs = element_dofs(pnodeids, dim)
            pdisp = nodaldisplacement[pdispdofs]
            pquad = cellquads[+1, cellid]
            pvals = stress_cell_rhs(basis, pquad, stiffness[+1], pdisp, jac)
            pstressdofs = element_dofs(pnodeids, sdim)
            assemble!(sysrhs, pstressdofs, pvals)

            nnodeids = nodal_connectivity(mesh, -1, cellid)
            ndispdofs = element_dofs(nnodeids, dim)
            ndisp = nodaldisplacement[ndispdofs]
            nquad = cellquads[-1, cellid]
            nvals = stress_cell_rhs(basis, nquad, stiffness[-1], ndisp, jac)
            nstressdofs = element_dofs(nnodeids, sdim)
            assemble!(sysrhs, nstressdofs, nvals)
        end
    end
end

function assemble_transformation_stress_linear_form!(
    sysrhs,
    transfstress,
    basis,
    cellquads::CellQuadratures,
    mesh,
)

    dim = dimension(basis)
    sdim = number_of_symmetric_degrees_of_freedom(dim)
    ncells = number_of_cells(mesh)
    detjac = determinant_jacobian(mesh)

    uniformquad = uniform_cell_quadrature(cellquads)
    uniformtransfrhs =
        constant_linear_form(transfstress, basis, uniformquad, detjac)

    for cellid = 1:ncells
        s = cell_sign(mesh, cellid)
        @assert s == -1 || s == 0 || s == +1
        if s == +1
            nodeids = nodal_connectivity(mesh, +1, cellid)
            stressdofs = element_dofs(nodeids, sdim)

            assemble!(sysrhs, stressdofs, -uniformtransfrhs)
        elseif s == 0
            pquad = cellquads[+1, cellid]
            transfrhs = constant_linear_form(transfstress, basis, pquad, detjac)
            nodeids = nodal_connectivity(mesh, +1, cellid)
            stressdofs = element_dofs(nodeids, sdim)

            assemble!(sysrhs, stressdofs, -transfrhs)
        end
    end
end

function assemble_penalty_displacement_transformation_rhs!(
    sysrhs,
    transfstress,
    basis,
    facequads,
    cutmesh,
    onboundary,
)

    cellids = findall(is_boundary_cell(cutmesh))
    cellconnectivity = cell_connectivity(cutmesh)
    facemidpoints = reference_face_midpoints()
    nfaces = length(facemidpoints)
    facedetjac = face_determinant_jacobian(cutmesh)

    dim = dimension(cutmesh)
    refnormals = reference_face_normals()
    refquads = uniform_face_quadratures(facequads)

    uniformrhs = [
        face_traction_transformation_rhs(basis, q, n, transfstress, s) for
        (q, n, s) in zip(refquads, refnormals, facedetjac)
    ]

    for cellid in cellids
        cellmap = cell_map(cutmesh, cellid)
        s = cell_sign(cutmesh, cellid)
        for faceid = 1:nfaces
            if cellconnectivity[faceid, cellid] == 0
                if onboundary(cellmap(facemidpoints[faceid]))
                    if s == 1
                        rhs = uniformrhs[faceid]
                        nodeids = nodal_connectivity(cutmesh, +1, cellid)
                        assemble_cell_rhs!(sysrhs, nodeids, dim, -rhs)
                    elseif s == 0
                        pquad = facequads[+1, faceid, cellid]
                        rhs = face_traction_transformation_rhs(
                            basis,
                            pquad,
                            refnormals[faceid],
                            transfstress,
                            facedetjac[faceid],
                        )
                        nodeids = nodal_connectivity(cutmesh, +1, cellid)
                        assemble_cell_rhs!(sysrhs, nodeids, dim, -rhs)
                    end
                end
            end
        end
    end
end

function assemble_penalty_displacement_component_transformation_rhs!(
    sysrhs,
    transfstress,
    basis,
    facequads,
    cutmesh,
    onboundary,
    component,
)

    cellids = findall(is_boundary_cell(cutmesh))
    cellconnectivity = cell_connectivity(cutmesh)
    facemidpoints = reference_face_midpoints()
    nfaces = length(facemidpoints)
    facedetjac = face_determinant_jacobian(cutmesh)

    dim = dimension(cutmesh)
    refnormals = reference_face_normals()
    refquads = uniform_face_quadratures(facequads)

    uniformrhs = [
        face_traction_component_transformation_rhs(
            basis,
            q,
            component,
            n,
            transfstress,
            s,
        ) for (q, n, s) in zip(refquads, refnormals, facedetjac)
    ]

    for cellid in cellids
        cellmap = cell_map(cutmesh, cellid)
        s = cell_sign(cutmesh, cellid)
        for faceid = 1:nfaces
            if cellconnectivity[faceid, cellid] == 0
                if onboundary(cellmap(facemidpoints[faceid]))
                    if s == 1
                        rhs = uniformrhs[faceid]
                        nodeids = nodal_connectivity(cutmesh, +1, cellid)
                        assemble_cell_rhs!(sysrhs, nodeids, dim, -rhs)
                    elseif s == 0
                        pquad = facequads[+1, faceid, cellid]
                        rhs = face_traction_component_transformation_rhs(
                            basis,
                            pquad,
                            component,
                            refnormals[faceid],
                            transfstress,
                            facedetjac[faceid],
                        )
                        nodeids = nodal_connectivity(cutmesh, +1, cellid)
                        assemble_cell_rhs!(sysrhs, nodeids, dim, -rhs)
                    end
                end
            end
        end
    end
end
