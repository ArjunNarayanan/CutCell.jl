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

function transform_gradient(gradf, cm::CellMap)
      return gradf / Diagonal(jacobian(cm))
end

function dimension(basis::TensorProductBasis{dim}) where {dim}
      return dim
end

function number_of_basis_functions(
      basis::TensorProductBasis{dim,T,NF},
) where {dim,T,NF}

      return NF
end

function plane_strain_voigt_hooke_matrix(lambda, mu)
      l2mu = lambda + 2mu
      return [
            l2mu lambda 0.0
            lambda l2mu 0.0
            0.0 0.0 mu
      ]
end

function bilinear_form(basis, quad, stiffness, cellmap)
      dim = dimension(basis)
      nf = number_of_basis_functions(basis)
      ndofs = dim * nf
      matrix = zeros(ndofs, ndofs)
      vectosymmconverter = vector_to_symmetric_matrix_converter()
      detjac = determinant_jacobian(cellmap)
      for (p, w) in quad
            grad = transform_gradient(gradient(basis, p), cellmap)
            for k = 1:dim
                  NK = make_row_matrix(vectosymmconverter[k], grad[:, k])
                  matrix .+= NK' * stiffness * NK * detjac * w
            end
      end
      return matrix
end
