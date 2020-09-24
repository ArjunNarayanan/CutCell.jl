import Base.==

function allapprox(v1,v2)
    return all(v1 .≈ v2)
end

function allequal(v1,v2)
    return all(v1 .== v2)
end

function Base.isequal(c1::CutCell.CellMap,c2::CutCell.CellMap)
    return allequal(c1.yL,c2.yL) && allequal(c1.yR,c2.yR)
end

function ==(c1::CutCell.CellMap,c2::CutCell.CellMap)
    return isequal(c1,c2)
end
