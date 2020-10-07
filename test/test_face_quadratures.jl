quad1d = ImplicitDomainQuadrature.ReferenceQuadratureRule(numqp)
update!(levelset, levelsetcoeffs[nodalconnectivity[:, 1]])
CutCell.face_quadrature_rules(levelset, +1, quad1d)
