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

@safetestset "Test Cut Mesh Algorithms" begin
    include("test_cut_mesh.jl")
end

@safetestset "Test Cell Quadratures" begin
    include("test_cell_quadratures.jl")
end

@safetestset "Test Interface Quadratures" begin
    include("test_interface_quadratures.jl")
end
