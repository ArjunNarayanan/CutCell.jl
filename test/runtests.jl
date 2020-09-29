using SafeTestsets

@safetestset "Test Mesh Routines" begin
    include("test_mesh_routines.jl")
end

@safetestset "Test Weak Form" begin
    include("test_weak_form.jl")
end

@safetestset "Test Assembly" begin
    include("test_assembly.jl")
end

@safetestset "Test Full Displacement BC Convergence" begin
    include("test_full_displacement_convergence.jl")
end

@safetestset "Test Uniform Tension" begin
    include("test_traction_bc.jl")
end

@safetestset "Test Displacement + Traction BC Convergence" begin
    include("test_displacement_traction_convergence.jl")
end
