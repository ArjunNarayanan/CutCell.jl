module CutCell

using LinearAlgebra
using SparseArrays
using CartesianMesh
using PolynomialBasis
using ImplicitDomainQuadrature

include("mesh_routines.jl")
include("weak_form.jl")
include("assembly.jl")

export plane_strain_voigt_hooke_matrix

end # module
