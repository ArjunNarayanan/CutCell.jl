function stress_cell_mass_matrix(basis, quad, detjac)
    dim = dimension(basis)
    sdim = number_of_symmetric_degrees_of_freedom(dim)
    return mass_matrix(basis, quad, detjac, sdim)
end

function stress_cell_rhs_operator(basis, quad, stiffness, jacobian)
    dim = dimension(basis)
    sdim = number_of_symmetric_degrees_of_freedom(dim)
    nf = number_of_basis_functions(basis)
    op = zeros(sdim * nf, dim * nf)
    vectosymmconverter = vector_to_symmetric_matrix_converter()
    detjac = prod(jacobian)

    for (p, w) in quad
        grad = transform_gradient(gradient(basis, p), jacobian)
        NK = sum([make_row_matrix(vectosymmconverter[k], grad[:, k]) for k = 1:dim])
        vals = basis(p)
        NI = interpolation_matrix(vals, sdim)

        op .+= NI' * stiffness * NK * detjac * w
    end
    return op
end

function stress_cell_rhs(basis, quad, stiffness, celldisp, jacobian)

    op = stress_cell_rhs_operator(basis, quad, stiffness, jacobian)
    return op * celldisp
end
