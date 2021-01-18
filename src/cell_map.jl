struct CellMap
    yL::Any
    yR::Any
    jacobian::Any
    dim::Any
    function CellMap(yL, yR)
        dim = length(yL)
        @assert length(yR) == dim
        @assert all(yR .>= yL)
        jacobian = 0.5 * (yR - yL)
        new(yL, yR, jacobian, dim)
    end
end

function Base.show(io::IO, cellmap::CellMap)
    dim = cellmap.dim
    yL = cellmap.yL
    yR = cellmap.yR
    str = "CellMap\n\tLeft : $yL\n\tRight: $yR"
    print(io, str)
end

function dimension(C::CellMap)
    return C.dim
end

function jacobian(C::CellMap)
    return C.jacobian
end

function inverse_jacobian(C::CellMap)
    return 1.0 ./ jacobian(C)
end

function determinant_jacobian(C)
    return prod(jacobian(C))
end

function face_determinant_jacobian(C::CellMap)
    jac = jacobian(C)
    return [jac[1], jac[2], jac[1], jac[2]]
end

function (C::CellMap)(x)
    dim = dimension(C)
    return C.yL .+ (jacobian(C) .* (x .+ ones(dim)))
end

function inverse(c::CellMap,x)
    jac = jacobian(c)
    return ((x .- c.yL) ./ jac) .- 1.0
end
