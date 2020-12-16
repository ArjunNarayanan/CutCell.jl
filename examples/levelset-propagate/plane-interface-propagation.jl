using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("../../test/useful_routines.jl")

function step_interface(levelset, levelsetcoeffs, mesh, levelsetspeed, dt)
    cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
    paddedmesh = CutCell.BoundaryPaddedMesh(cutmesh, 1)
    refseedpoints, spatialseedpoints, seedcellids =
        CutCell.seed_zero_levelset(2, levelset, levelsetcoeffs, cutmesh)
    paddedlevelset = CutCell.BoundaryPaddedLevelset(
        paddedmesh,
        refseedpoints,
        spatialseedpoints,
        seedcellids,
        levelset,
        levelsetcoeffs,
        cutmesh,
        1e-10,
    )
    return CutCell.step_first_order_levelset(paddedlevelset,levelsetspeed,dt)
end

function grid_size(mesh)
    w = CutCell.widths(mesh)
    nn = CutCell.nodes_per_mesh_side(mesh)

    return w ./ (nn .- 1)
end

function time_step_size(levelsetspeed,mesh;CFL=0.9)
    dx = minimum(grid_size(mesh))
    s = maximum(abs.(levelsetspeed))
    return CFL*dx/s
end

function run_time_steps(levelset,initialcondition,mesh,levelsetspeed,dt,nsteps)
    levelsetcoeffs = [copy(initialcondition) for i = 1:nsteps+1]
    for i = 1:nsteps
        levelsetcoeffs[i+1] = step_interface(levelset,levelsetcoeffs[i],mesh,levelsetspeed,dt)
    end
    return levelsetcoeffs
end


x0 = [0.0, 0.0]
L, W = 1.0, 1.0
nelmts = 10
numghostlayers = 1
polyorder = 2

xI = [0.05, 0.0]
normal = [1.0, 0.0]
tol = 1e-8
speed = 1.0
stoptime = 0.8

basis = TensorProductBasis(2, polyorder)
mesh = CutCell.Mesh(x0, [L, W], [nelmts, nelmts], basis)

levelset = InterpolatingPolynomial(1, basis)
initiallevelset =
    CutCell.levelset_coefficients(x -> plane_distance_function(x, normal, xI), mesh)
levelsetspeed = speed*ones(length(initiallevelset))

dt = time_step_size(levelsetspeed,mesh)
nsteps = ceil(Int,stoptime/dt)
levelsetcoeffs = run_time_steps(levelset,initiallevelset,mesh,levelsetspeed,stoptime)

actualstoptime = dt*nsteps
xT = xI + [actualstoptime*speed,0.0]

quad = tensor_product_quadrature(2,3)
err = uniform_mesh_L2_error(levelsetcoeffs[end]',x->plane_distance_function(x,normal,xT),basis,quad,mesh)
@test allapprox(err,[0.0],1e2eps())


function grid_range(mesh)
    x0 = CutCell.reference_corner(mesh)
    w = CutCell.widths(mesh)
    nfmside = CutCell.nodes_per_mesh_side(mesh)

    x = range(x0[1],stop=x0[1]+w[1],length=nfmside[1])
    y = range(x0[2],stop=x0[2]+w[2],length=nfmside[2])

    return x,y
end


using Plots
x,y = grid_range(mesh)
Z = reshape(levelsetcoeffs[1],length(y),:)
Plots.contour(x,y,Z,levels=-1:0.25:1,color="black",linewidth=2,legend=false)

anim = @animate for i = 1:length(levelsetcoeffs)
    Z = reshape(levelsetcoeffs[i],length(y),:)
    contour(x, y, Z, levels = -1:0.25:1,color="black",linewidth=2,legend=false)
end

gif(anim,fps=5)
