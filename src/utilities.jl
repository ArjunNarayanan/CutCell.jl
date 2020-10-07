function levelset_normal(levelset, p::V, invjac) where {V<:AbstractVector}
    g = vec(gradient(levelset, p))
    n = invjac .* g
    return n / norm(n)
end

function levelset_normal(levelset, points::M, invjac) where {M<:AbstractMatrix}
    npts = size(points)[2]
    g = hcat([gradient(levelset, points[:, i])' for i = 1:npts]...)
    normals = diagm(invjac) * g
    for i = 1:npts
        n = normals[:, i]
        normals[:, i] = n / norm(n)
    end
    return normals
end

function levelset_coefficients(distancefunc,mesh)
    nodalcoordinates = nodal_coordinates(mesh)
    return distancefunc(nodalcoordinates)
end

function reference_bottom_face_midpoint()
    [0.0, -1.0]
end

function reference_right_face_midpoint()
    [1.0, 0.0]
end

function reference_top_face_midpoint()
    [0.0, 1.0]
end

function reference_left_face_midpoint()
    [-1.0, 0.0]
end

function reference_face(faceid)
    if faceid == 1
        return (2, -1.0)
    elseif faceid == 2
        return (1, +1.0)
    elseif faceid == 3
        return (2, +1.0)
    elseif faceid == 4
        return (1, -1.0)
    else
        error("Expected faceid ∈ {1,2,3,4}, got faceid = $faceid")
    end
end

function reference_face_midpoints()
    [
        reference_bottom_face_midpoint(),
        reference_right_face_midpoint(),
        reference_top_face_midpoint(),
        reference_left_face_midpoint(),
    ]
end

function extend_to_face(points, faceid)
    dir, coordval = reference_face(faceid)
    @assert dir == 1 || dir == 2
    flipdir = dir == 1 ? 2 : 1
    return extend([coordval], flipdir, points)
end

function tangents(normals)
    rot = [
        0.0 1.0
        -1.0 0.0
    ]
    return rot * normals
end

function scale_area(cellmap, normals)
    t = tangents(normals)
    invjac = inverse_jacobian(cellmap)
    den = sqrt.((t .^ 2)' * (invjac .^ 2))
    return 1.0 ./ den
end
