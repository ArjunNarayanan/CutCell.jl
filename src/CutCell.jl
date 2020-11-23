module CutCell

using LinearAlgebra
using SparseArrays
using NearestNeighbors
using CartesianMesh
using PolynomialBasis
using ImplicitDomainQuadrature
import ImplicitDomainQuadrature: extend

include("cell_map.jl")
include("elasticity.jl")
include("mesh_routines.jl")
include("dg_mesh.jl")
include("weak_form.jl")
include("assembly.jl")
include("dirichlet_bc.jl")
include("cut_mesh.jl")
include("cell_quadratures.jl")
include("face_quadratures.jl")
include("interface_quadratures.jl")
include("cell_merging.jl")
include("cut_mesh_bilinear_forms.jl")
include("transformation_strain.jl")
include("interface_conditions.jl")
include("penalty_displacement_bc.jl")
include("stress.jl")
include("cut_mesh_assembly.jl")
include("postprocess.jl")
include("levelset_reinitialize.jl")
include("utilities.jl")

export plane_strain_voigt_hooke_matrix

end # module
