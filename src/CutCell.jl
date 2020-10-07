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
include("assembly.jl")
include("interface_conditions.jl")
include("cut_mesh.jl")
include("cell_quadratures.jl")
include("interface_quadratures.jl")

export plane_strain_voigt_hooke_matrix

end # module
