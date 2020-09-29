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

function interpolation_matrix(vals,ndofs)
      return make_row_matrix(diagm(ones(ndofs)),vals)
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
            l2mu   lambda 0.0
            lambda l2mu   0.0
            0.0    0.0    mu
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
            NK = zeros(3,2nf)
            for k = 1:dim
                  NK .+= make_row_matrix(vectosymmconverter[k], grad[:, k])
            end
            matrix .+= NK' * stiffness * NK * detjac * w
      end
      return matrix
end

function mass_matrix(basis,quad,cellmap,ndofs)
      nf = number_of_basis_functions(basis)
      totaldofs = ndofs*nf
      matrix = zeros(totaldofs,totaldofs)
      detjac = determinant_jacobian(cellmap)
      for (p,w) in quad
            vals = basis(p)
            N = interpolation_matrix(vals,ndofs)
            matrix .+= N'*N*detjac*w
      end
      return matrix
end

function linear_form(rhsfunc, basis, quad, cellmap)
      dim = dimension(basis)
      nf = number_of_basis_functions(basis)
      rhs = zeros(dim*nf)
      detjac = determinant_jacobian(cellmap)
      for (p,w) in quad
            vals = rhsfunc(cellmap(p))
            N = interpolation_matrix(basis(p),dim)
            rhs .+= N'*vals*detjac*w
      end
      return rhs
end

function linear_form(rhsfunc, basis, quad, cellmap, detjac)
      dim = dimension(basis)
      nf = number_of_basis_functions(basis)
      rhs = zeros(dim*nf)
      for (p,w) in quad
            vals = rhsfunc(cellmap(p))
            N = interpolation_matrix(basis(p),dim)
            rhs .+= N'*vals*detjac*w
      end
      return rhs
end

function extend_to_bottom_face(points)
      @assert size(points)[1] == 1
      return extend([-1.0],1,points)
end

function extend_to_right_face(points)
      @assert size(points)[1] == 1
      return extend([1.0],2,points)
end

function extend_to_top_face(points)
      @assert size(points)[1] == 1
      return extend([1.0],1,points)
end

function extend_to_left_face(points)
      @assert size(points)[1] == 1
      return extend([-1.0],2,points)
end

function extend_to_face(points,faceid)
      if faceid == 1
            extend_to_bottom_face(points)
      elseif faceid == 2
            extend_to_right_face(points)
      elseif faceid == 3
            extend_to_top_face(points)
      elseif faceid == 4
            extend_to_left_face(points)
      else
            error("Expected faceid ∈ {1,2,3,4}, got faceid = $faceid")
      end
end

function extend_to_face(quad::QuadratureRule,faceid)
      extp = extend_to_face(quad.points,faceid)
      return QuadratureRule(extp,quad.weights)
end

function face_quadrature_rules(numqp)
      quad1d = tensor_product_quadrature(1,numqp)
      facequads = [extend_to_face(quad1d,faceid) for faceid = 1:4]
      return facequads
end
