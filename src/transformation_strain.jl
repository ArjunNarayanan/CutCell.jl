function plane_transformation_strain(theta)
    return [theta / 3.0, theta / 3.0, 0.0]
end

function plane_strain_transformation_stress(lambda, mu, theta)
    pt = (lambda + 2mu / 3) * theta
    transfstress = [pt, pt, 0.0]
    return transfstress
end

function bulk_transformation_rhs(basis, quad, transfstress, jac)
    dim = dimension(basis)
    nf = number_of_basis_functions(basis)
    ndofs = dim * nf
    rhs = zeros(ndofs)
    vectosymmconverter = vector_to_symmetric_matrix_converter()
    detjac = prod(jac)

    for (p, w) in quad
        grad = transform_gradient(gradient(basis, p), jac)
        NK = sum([make_row_matrix(vectosymmconverter[k], grad[:, k]) for k = 1:dim])
        rhs .+= NK' * transfstress * detjac * w
    end
    return rhs
end

function interface_transformation_rhs(basis, quad, normals, transfstress, cellmap)
    numqp = length(quad)
    @assert size(normals)[2] == numqp
    dim = dimension(basis)
    nf = number_of_basis_functions(basis)
    ndofs = dim * nf
    rhs = zeros(ndofs)

    vectosymmconverter = vector_to_symmetric_matrix_converter()
    scalearea = scale_area(cellmap, normals)

    for qpidx = 1:numqp
        p, w = quad[qpidx]

        vals = basis(p)
        NI = interpolation_matrix(vals, dim)

        normal = normals[:, qpidx]
        NK = sum([normal[k] * vectosymmconverter[k]' for k = 1:dim])

        rhs .+= NI' * NK * transfstress * scalearea[qpidx] * w
    end
    return rhs
end

function interface_transformation_component_rhs(
    basis,
    quad,
    components,
    normals,
    transfstress,
    cellmap,
)

    numqp = length(quad)
    @assert size(normals)[2] == size(components)[2] == numqp
    dim = dimension(basis)
    nf = number_of_basis_functions(basis)
    ndofs = dim * nf
    rhs = zeros(ndofs)

    vectosymmconverter = vector_to_symmetric_matrix_converter()
    scalearea = scale_area(cellmap, normals)

    for qpidx = 1:numqp
        p, w = quad[qpidx]

        vals = basis(p)
        NI = interpolation_matrix(vals, dim)

        normal = normals[:, qpidx]
        NK = sum([normal[k] * vectosymmconverter[k]' for k = 1:dim])

        component = components[:,qpidx]
        projector = component*component'

        rhs .+= NI' * projector * NK * transfstress * scalearea[qpidx] * w
    end
    return rhs
end

function face_traction_transformation_rhs(basis, quad, normal, transfstress, facescale)
    dim = dimension(basis)
    @assert length(normal) == dim
    nf = number_of_basis_functions(basis)
    ndofs = dim * nf
    rhs = zeros(ndofs)

    vectosymmconverter = vector_to_symmetric_matrix_converter()
    NK = sum([normal[k] * vectosymmconverter[k]' for k = 1:dim])

    for (p, w) in quad
        vals = basis(p)
        NI = interpolation_matrix(vals, dim)

        rhs .+= NI' * NK * transfstress * facescale * w
    end
    return rhs
end

function face_traction_component_transformation_rhs(
    basis,
    quad,
    component,
    normal,
    transfstress,
    facescale,
)

    dim = dimension(basis)
    @assert length(normal) == dim
    nf = number_of_basis_functions(basis)
    ndofs = dim * nf
    rhs = zeros(ndofs)

    projector = component * component'
    vectosymmconverter = vector_to_symmetric_matrix_converter()
    NK = sum([normal[k] * vectosymmconverter[k]' for k = 1:dim])

    for (p, w) in quad
        vals = basis(p)
        NI = interpolation_matrix(vals, dim)

        rhs .+= NI' * projector * NK * transfstress * facescale * w
    end
    return rhs
end
