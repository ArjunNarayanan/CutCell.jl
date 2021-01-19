struct InterElementTractionOperatorValues
    nn
    np
    pn
    pp
    nnT
    npT
    pnT
    ppT
    function InterElementTractionOperatorValues(nntop,nptop,pntop,pptop,eta)
        nn = -0.5*vec(nntop)
        np = -0.5*vec(nptop)
        pn = -0.5*vec(pntop)
        pp = -0.5*vec(pptop)

        nnT = -0.5*eta*vec(transpose(nntop))
        npT = -0.5*eta*vec(transpose(nptop))
        pnT = -0.5*eta*vec(transpose(pntop))
        ppT = -0.5*eta*vec(transpose(pptop))

        new(nn,np,pn,pp,nnT,npT,pnT,ppT)
    end
end

function interelement_traction_operators(basis,quad1,quad2,normal,stiffness,facedetjac,jac,eta)
    nntop = face_traction_operator(basis,quad1,quad1,normal,stiffness,facedetjac,jac)
    nptop = face_traction_operator(basis,quad1,quad2,normal,stiffness,facedetjac,jac)
    pntop = face_traction_operator(basis,quad2,quad1,normal,stiffness,facedetjac,jac)
    pptop = face_traction_operator(basis,quad2,quad2,normal,stiffness,facedetjac,jac)

    return InterElementTractionOperatorValues(nntop,nptop,pntop,pptop,eta)
end

struct InterElementMassOperatorValues
    nn
    np
    pn
    pp
    function InterElementMassOperatorValues(nnmop,npmop,pnmop,ppmop)
        nn = vec(nnmop)
        np = vec(npmop)
        pn = vec(pnmop)
        pp = vec(ppmop)

        new(nn,np,pn,pp)
    end
end

function interelement_mass_operators(basis,quad1,quad2,facescale)
    dim = dimension(basis)
    nn = mass_matrix(basis,quad1,quad1,facescale,dim)
    np = mass_matrix(basis,quad1,quad2,facescale,dim)
    pn = mass_matrix(basis,quad2,quad1,facescale,dim)
    pp = mass_matrix(basis,quad2,quad2,facescale,dim)

    return InterElementMassOperatorValues(nn,np,pn,pp)
end
