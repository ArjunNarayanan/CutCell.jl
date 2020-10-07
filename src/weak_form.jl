function Base.length(quad::QuadratureRule{D,NQ}) where {D,NQ}
      return NQ
end

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
            NK = zeros(3, 2nf)
            for k = 1:dim
                  NK .+= make_row_matrix(vectosymmconverter[k], grad[:, k])
            end
            matrix .+= NK' * stiffness * NK * detjac * w
      end
      return matrix
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
      return mass_matrix(basis,quad,detjac,ndofs)
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
