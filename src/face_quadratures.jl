function face_quadrature(faceid, levelset, signcondition, quad1d)
    dir, coordval = reference_face(faceid)
    quad = QuadratureRule(ImplicitDomainQuadrature.one_dimensional_quadrature(
        [x -> levelset(extend(x, dir, coordval))],
        [signcondition],
        -1.0,
        +1.0,
        quad1d,
    ))
    return extend_to_face(quad, faceid)
end


function face_quadratures(levelset, signcondition, quad1d)
    quads = [
        face_quadrature(faceid, levelset, signcondition, quad1d)
        for faceid = 1:4
    ]
    return quads
end
