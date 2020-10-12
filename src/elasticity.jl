struct HookeStiffness
    stiffness::Any
    function HookeStiffness(stiffness)
        @assert length(stiffness) == 2
        new(stiffness)
    end
end

function HookeStiffness(lambda1, mu1, lambda2, mu2)
    stiffness1 = plane_strain_voigt_hooke_matrix(lambda1, mu1)
    stiffness2 = plane_strain_voigt_hooke_matrix(lambda2, mu2)
    return HookeStiffness([stiffness1, stiffness2])
end

function Base.getindex(hs::HookeStiffness, s)
    (s == -1 || s == +1) ||
        error("Use ± to index into 1st dimension of HookeStiffness, got index = $s")
    row = s == +1 ? 1 : 2
    return hs.stiffness[row]
end

function plane_strain_voigt_hooke_matrix(lambda, mu)
    l2mu = lambda + 2mu
    return [
        l2mu lambda 0.0
        lambda l2mu 0.0
        0.0 0.0 mu
    ]
end
