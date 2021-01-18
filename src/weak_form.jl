function vector_to_symmetric_matrix_converter()
    E1 = [
        1.0 0.0
        0.0 0.0
        0.0 1.0
    ]
    E2 = [
        0.0 0.0
        0.0 1.0
        1.0 0.0
    ]
    return [E1, E2]
end

function make_row_matrix(matrix, vals)
    return hcat([v * matrix for v in vals]...)
end

function interpolation_matrix(vals, ndofs)
    return make_row_matrix(diagm(ones(ndofs)), vals)
end

function transform_gradient(gradf, jacobian)
    return gradf / Diagonal(jacobian)
end

function bilinear_form(basis, quad, stiffness, jacobian)
    dim = dimension(basis)
    nf = number_of_basis_functions(basis)
    ndofs = dim * nf
    matrix = zeros(ndofs, ndofs)
    vectosymmconverter = vector_to_symmetric_matrix_converter()
    detjac = prod(jacobian)
    for (p, w) in quad
        grad = transform_gradient(gradient(basis, p), jacobian)
        NK = zeros(3, 2nf)
        for k = 1:dim
            NK .+= make_row_matrix(vectosymmconverter[k], grad[:, k])
        end
        matrix .+= NK' * stiffness * NK * detjac * w
    end
    return matrix
end

function bilinear_form(basis, quad, stiffness, cellmap::CellMap)
    return bilinear_form(basis, quad, stiffness, jacobian(cellmap))
end

function mass_matrix(basis, quad, detjac::R, ndofs) where {R<:Real}
    nf = number_of_basis_functions(basis)
    totaldofs = ndofs * nf
    matrix = zeros(totaldofs, totaldofs)
    for (p, w) in quad
        vals = basis(p)
        N = interpolation_matrix(vals, ndofs)
        matrix .+= N' * N * detjac * w
    end
    return matrix
end

function mass_matrix(basis, quad, cellmap::CellMap, ndofs)
    detjac = determinant_jacobian(cellmap)
    return mass_matrix(basis, quad, detjac, ndofs)
end

function mass_matrix(basis, quad, scale::V, ndofs) where {V<:AbstractVector}
    nf = number_of_basis_functions(basis)
    totaldofs = ndofs * nf
    matrix = zeros(totaldofs, totaldofs)
    @assert length(scale) == length(quad)
    for (idx, (p, w)) in enumerate(quad)
        vals = basis(p)
        N = interpolation_matrix(vals, ndofs)
        matrix .+= N' * N * scale[idx] * w
    end
    return matrix
end

function component_mass_matrix(basis, quad, component, scale)
    nf = number_of_basis_functions(basis)
    dim = dimension(basis)
    @assert dim == length(component)
    totaldofs = dim * nf
    matrix = zeros(totaldofs, totaldofs)
    projector = component * component'
    for (p, w) in quad
        vals = basis(p)
        NI = interpolation_matrix(vals, dim)
        NP = make_row_matrix(projector, vals)
        matrix .+= NI' * NP * scale * w
    end
    return matrix
end

function face_traction_operator(
    basis,
    quad1,
    quad2,
    normal,
    stiffness,
    facescale,
    cellmap,
)

    numqp = length(quad1)
    @assert length(quad2) == numqp

    dim = dimension(basis)
    @assert length(normal) == dim
    nf = number_of_basis_functions(basis)
    ndofs = dim * nf
    matrix = zeros(ndofs, ndofs)

    vectosymmconverter = vector_to_symmetric_matrix_converter()
    N = sum([normal[k] * vectosymmconverter[k]' for k = 1:dim])
    jac = jacobian(cellmap)

    for qpidx = 1:numqp
        p1, w1 = quad1[qpidx]
        p2, w2 = quad2[qpidx]

        @assert w1 ≈ w2

        vals = basis(p1)
        grad = transform_gradient(gradient(basis, p2), jac)

        NK = sum([
            make_row_matrix(vectosymmconverter[k], grad[:, k]) for k = 1:dim
        ])
        NI = interpolation_matrix(vals, dim)

        matrix .+= NI' * N * stiffness * NK * facescale * w1
    end
    return matrix
end

function face_traction_operator(
    basis,
    quad,
    normal,
    stiffness,
    facescale,
    cellmap,
)

    return face_traction_operator(
        basis,
        quad,
        quad,
        normal,
        stiffness,
        facescale,
        cellmap,
    )
    # dim = dimension(basis)
    # @assert length(normal) == dim
    # nf = number_of_basis_functions(basis)
    # ndofs = dim * nf
    # matrix = zeros(ndofs, ndofs)
    #
    # vectosymmconverter = vector_to_symmetric_matrix_converter()
    # N = sum([normal[k] * vectosymmconverter[k]' for k = 1:dim])
    # jac = jacobian(cellmap)
    #
    # for (p, w) in quad
    #     grad = transform_gradient(gradient(basis, p), jac)
    #     vals = basis(p)
    #
    #     NK = zeros(3, 2nf)
    #     for k = 1:dim
    #         NK .+= make_row_matrix(vectosymmconverter[k], grad[:, k])
    #     end
    #     NI = interpolation_matrix(vals, dim)
    #     matrix .+= NI' * N * stiffness * NK * facescale * w
    # end
    # return matrix
end

function face_traction_component_operator(
    basis,
    quad,
    component,
    normal,
    stiffness,
    facescale,
    cellmap,
)
    dim = dimension(basis)
    @assert length(normal) == dim
    @assert length(component) == dim

    nf = number_of_basis_functions(basis)
    ndofs = dim * nf
    matrix = zeros(ndofs, ndofs)

    vectosymmconverter = vector_to_symmetric_matrix_converter()
    N = sum([normal[k] * vectosymmconverter[k]' for k = 1:dim])
    projector = component * component'
    jac = jacobian(cellmap)

    for (p, w) in quad
        grad = transform_gradient(gradient(basis, p), jac)
        vals = basis(p)

        NK = sum([
            make_row_matrix(vectosymmconverter[k], grad[:, k]) for k = 1:dim
        ])
        NI = interpolation_matrix(vals, dim)

        matrix .+= NI' * projector * N * stiffness * NK * facescale * w
    end
    return matrix
end

function component_linear_form(rhsfunc, basis, quad, component, cellmap, detjac)
    dim = dimension(basis)
    @assert length(component) == dim
    nf = number_of_basis_functions(basis)
    rhs = zeros(dim * nf)
    for (p, w) in quad
        rhsval = rhsfunc(cellmap(p))
        NI = interpolation_matrix(basis(p), dim)
        rhs .+= rhsval * NI' * component * detjac * w
    end
    return rhs
end

function component_linear_form(rhsval, basis, quad, component, detjac)
    dim = dimension(basis)
    @assert length(component) == dim
    nf = number_of_basis_functions(basis)
    rhs = zeros(dim * nf)
    for (p, w) in quad
        NI = interpolation_matrix(basis(p), dim)
        rhs .+= rhsval * NI' * component * detjac * w
    end
    return rhs
end

function linear_form(rhsfunc, basis, quad, cellmap)
    dim = dimension(basis)
    nf = number_of_basis_functions(basis)
    rhs = zeros(dim * nf)
    detjac = determinant_jacobian(cellmap)
    for (p, w) in quad
        vals = rhsfunc(cellmap(p))
        N = interpolation_matrix(basis(p), dim)
        rhs .+= N' * vals * detjac * w
    end
    return rhs
end

function linear_form(rhsfunc, basis, quad, cellmap, detjac)
    dim = dimension(basis)
    nf = number_of_basis_functions(basis)
    rhs = zeros(dim * nf)
    for (p, w) in quad
        vals = rhsfunc(cellmap(p))
        N = interpolation_matrix(basis(p), dim)
        rhs .+= N' * vals * detjac * w
    end
    return rhs
end

function constant_linear_form(rhsval, basis, quad, detjac)
    ndofs = length(rhsval)
    nf = number_of_basis_functions(basis)
    rhs = zeros(ndofs * nf)
    for (p, w) in quad
        vals = basis(p)
        NI = interpolation_matrix(basis(p), ndofs)
        rhs .+= NI' * rhsval * detjac * w
    end
    return rhs
end
