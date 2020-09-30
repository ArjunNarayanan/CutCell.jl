function tangents(normals)
    rot = [
        0.0 1.0
        -1.0 0.0
    ]
    return rot * normals
end

function scale_area(cellmap, normals)
    t = tangents(normals)
    invjac = inverse_jacobian(cellmap)
    den = sqrt.((t .^ 2)' * (invjac .^ 2))
    return 1.0 ./ den
end

function assemble_coherent_interface!(
    sysmatrix::SystemMatrix,
    massmatrix,
    nodeids1,
    nodeids2,
    dofspernode,
)

    edofs1 = element_dofs(nodeids1,dofspernode)
    edofs2 = element_dofs(nodeids2,dofspernode)
    vals = vec(massmatrix)

    r1,c1 = element_dofs_to_operator_dofs(edofs1,edofs1)
    assemble!(sysmatrix,r1,c1,vals)
    r2,c2 = element_dofs_to_operator_dofs(edofs1,edofs2)
    assemble!(sysmatrix,r2,c2,-vals)
    r3,c3 = element_dofs_to_operator_dofs(edofs2,edofs1)
    assemble!(sysmatrix,r3,c3,-vals)
    r4,c4 = element_dofs_to_operator_dofs(edofs2,edofs2)
    assemble!(sysmatrix,r4,c4,vals)
end
