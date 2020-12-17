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

function time_step_size(levelsetspeed,mesh;CFL=0.5)
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

function grid_range(mesh)
    x0 = CutCell.reference_corner(mesh)
    w = CutCell.widths(mesh)
    nfmside = CutCell.nodes_per_mesh_side(mesh)

    x = range(x0[1],stop=x0[1]+w[1],length=nfmside[1])
    y = range(x0[2],stop=x0[2]+w[2],length=nfmside[2])

    return x,y
end

rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])


x0 = [0.0, 0.0]
L, W = 1.0, 1.0
nelmts = 40
numghostlayers = 1
polyorder = 2

xc = [0.87,0.87]
speed = 1.0
stoptime = 0.4

basis = TensorProductBasis(2,polyorder)
mesh = CutCell.Mesh(x0,[L,W],[nelmts,nelmts],basis)

levelset = InterpolatingPolynomial(1, basis)
initiallevelset =
    CutCell.levelset_coefficients(x -> corner_distance_function(x,xc), mesh)

levelsetspeed = speed*ones(length(initiallevelset))
dt = time_step_size(levelsetspeed,mesh)
@assert isinteger(stoptime/dt)
nsteps = round(Int,stoptime/dt)
# nsteps = 107

levelsetcoeffs = run_time_steps(levelset,initiallevelset,mesh,levelsetspeed,dt,nsteps)



using Plots
x,y = grid_range(mesh)

# Z1 = reshape(levelsetcoeffs[1],length(y),:)
# Z2 = reshape(levelsetcoeffs[end],length(y),:)
# fig = plot(legend=false,aspect_ratio=:equal)
# plot!(fig,rectangle(L,W,x0[1],x0[2]),opacity=0.2,linewidth=2,strokecolor="black")
# contour!(fig,x,y,Z1,levels=[0.0],color="black",linewidth=2)
# contour!(fig,x,y,Z2,levels=[0.0],color="red",linewidth=2)
#

anim = @animate for i = 1:length(levelsetcoeffs)
    Z = reshape(levelsetcoeffs[i],length(y),:)
    fig = plot(legend=false,aspect_ratio=:equal)
    plot!(fig,rectangle(L,W,x0[1],x0[2]),opacity=0.2,linewidth=2,fillcolor="blue")
    contour!(fig,x,y,Z,levels=[0.0],linewidth=2,color="red")
end

gif(anim,fps=10)
gif(anim,"examples/levelset-propagate/corner.gif",fps=10)
