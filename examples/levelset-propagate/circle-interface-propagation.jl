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

xc = [0.5,0.5]
radius = 0.5
speed = 1.0
stoptime = 0.3

basis = TensorProductBasis(2,polyorder)
mesh = CutCell.Mesh(x0,[L,W],[nelmts,nelmts],basis)

levelset = InterpolatingPolynomial(1, basis)
initiallevelset =
    CutCell.levelset_coefficients(x -> circle_distance_function(x,xc,radius), mesh)
levelsetspeed = speed*ones(length(initiallevelset))

dt = time_step_size(levelsetspeed,mesh)
nsteps = ceil(Int,stoptime/dt)

levelsetcoeffs = run_time_steps(levelset,initiallevelset,mesh,levelsetspeed,dt,nsteps)

actualstoptime = dt*nsteps
finalradius = radius - speed*actualstoptime

# quad = tensor_product_quadrature(2,3)
# initialerror = uniform_mesh_L2_error(levelsetcoeffs[1]',x->circle_distance_function(x,xc,radius)[1],basis,quad,mesh)
# finalerror = uniform_mesh_L2_error(levelsetcoeffs[end]',x->circle_distance_function(x,xc,finalradius)[1],basis,quad,mesh)

function grid_range(mesh)
    x0 = CutCell.reference_corner(mesh)
    w = CutCell.widths(mesh)
    nfmside = CutCell.nodes_per_mesh_side(mesh)

    x = range(x0[1],stop=x0[1]+w[1],length=nfmside[1])
    y = range(x0[2],stop=x0[2]+w[2],length=nfmside[2])

    return x,y
end

rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])

# using Plots
# x,y = grid_range(mesh)
# Z1 = reshape(levelsetcoeffs[1],length(y),:)
# Z2 = reshape(levelsetcoeffs[end],length(y),:)
# fig = plot(legend=false,aspect_ratio=:equal)
# plot!(fig,rectangle(L,W,x0[1],x0[2]),opacity=0.2,linewidth=2,strokecolor="black")
# contour!(fig,x,y,Z1,color="black",linewidth=2)
# contour!(fig,x,y,Z2,color="red",linewidth=2)

anim = @animate for i = 1:length(levelsetcoeffs)
    Z = reshape(levelsetcoeffs[i],length(y),:)
    fig = plot(legend=false,aspect_ratio=:equal)
    plot!(fig,rectangle(L,W,x0[1],x0[2]),opacity=0.2,linewidth=2,fillcolor="blue")
    contour!(fig,x,y,Z,levels=[0.0],color="red",linewidth=2)
end

gif(anim,"examples/levelset-propagate/shrinking-circle.gif",fps=5)
