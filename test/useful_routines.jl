import Base.==, Base.≈

function allapprox(v1, v2)
    return all(v1 .≈ v2)
end

function allapprox(v1, v2, atol)
    return length(v1) == length(v2) &&
           all([isapprox(v1[i], v2[i], atol = atol) for i = 1:length(v1)])
end

function allequal(v1, v2)
    return all(v1 .== v2)
end

function Base.isequal(c1::CutCell.CellMap, c2::CutCell.CellMap)
    return allequal(c1.yL, c2.yL) && allequal(c1.yR, c2.yR)
end

function ==(c1::CutCell.CellMap, c2::CutCell.CellMap)
    return isequal(c1, c2)
end

function ≈(q1::QuadratureRule,q2::QuadratureRule)
    flag = allapprox(q1.points,q2.points) && allapprox(q1.weights,q2.weights)
end

function plane_distance_function(coords, normal, x0)
    return (coords .- x0)' * normal
end

function plane_distance_function(coords, normal, x0)
    return (coords .- x0)' * normal
end

function circle_distance_function(coords, center, radius)
    difference = coords .- center
    distance = [radius - norm(difference[:,i]) for i = 1:size(difference)[2]]
    return distance
end
