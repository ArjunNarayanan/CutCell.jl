module CutCell

using LinearAlgebra
using SparseArrays
using IntervalArithmetic
using CartesianMesh
using PolynomialBasis
using ImplicitDomainQuadrature
import ImplicitDomainQuadrature: extend

include("mesh_routines.jl")
include("weak_form.jl")
include("cut_mesh.jl")
include("assembly.jl")
include("interface_conditions.jl")

export plane_strain_voigt_hooke_matrix

end # module
