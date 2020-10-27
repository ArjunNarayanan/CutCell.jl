function plane_strain_transformation_stress(lambda,mu,theta)
    pt = (lambda + 2mu/3)*theta
    transfstress = [pt,pt,0.0]
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
