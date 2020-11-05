struct HookeStiffness
    stiffness::Any
    lamecoeffs
    function HookeStiffness(stiffness,lamecoeffs)
        @assert length(stiffness) == length(lamecoeffs) == 2
        new(stiffness,lamecoeffs)
    end
end

function HookeStiffness(lambda1, mu1, lambda2, mu2)
    stiffness1 = plane_strain_voigt_hooke_matrix(lambda1, mu1)
    stiffness2 = plane_strain_voigt_hooke_matrix(lambda2, mu2)
    lamecoeffs = [(lambda1,mu1),(lambda2,mu2)]
    return HookeStiffness([stiffness1, stiffness2],lamecoeffs)
end

function Base.getindex(hs::HookeStiffness, s)
    row = cell_sign_to_row(s)
    return hs.stiffness[row]
end

function lame_coefficients(hs::HookeStiffness,s)
    row = cell_sign_to_row(s)
    return hs.lamecoeffs[row]
end

function plane_strain_voigt_hooke_matrix(lambda, mu)
    l2mu = lambda + 2mu
    return [
        l2mu lambda 0.0
        lambda l2mu 0.0
        0.0 0.0 mu
    ]
end
