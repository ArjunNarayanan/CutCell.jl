function stress_cell_mass_matrix(basis, quad, detjac)
    dim = dimension(basis)
    sdim = number_of_symmetric_degrees_of_freedom(dim)
    return mass_matrix(basis, quad, detjac, sdim)
end

function stress_cell_rhs(basis, quad, stiffness, celldisp, jacobian)
    dim = dimension(basis)
    sdim = number_of_symmetric_degrees_of_freedom(dim)
    nf = number_of_basis_functions(basis)
    ndofs = sdim * nf
    rhs = zeros(ndofs)
    vectosymmconverter = vector_to_symmetric_matrix_converter()
    detjac = prod(jacobian)

    for (p, w) in quad
        grad = transform_gradient(gradient(basis, p), jacobian)
        NK = sum([make_row_matrix(vectosymmconverter[k], grad[:, k]) for k = 1:dim])
        vals = basis(p)
        NI = interpolation_matrix(vals, sdim)

        rhs .+= NI' * stiffness * NK * celldisp * detjac * w
    end
end
