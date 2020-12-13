using Test
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("useful_routines.jl")

function reinitialization_error(distancefunction, nelmts, polyorder)
    L, W = 1.0, 1.0
    basis = TensorProductBasis(2, polyorder)
    numqp = required_quadrature_order(polyorder) + 2
    quad = tensor_product_quadrature(2, numqp)
    levelset = InterpolatingPolynomial(1, basis)

    mesh = CutCell.Mesh([0.0, 0.0], [L, W], [nelmts, nelmts], basis)
    levelsetcoeffs = CutCell.levelset_coefficients(distancefunction, mesh)
    cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)

    refseedpoints, spatialseedpoints, seedcellids =
        CutCell.seed_zero_levelset(2, levelset, levelsetcoeffs, cutmesh)

    nodalcoordinates = CutCell.nodal_coordinates(cutmesh)
    signeddistance = CutCell.distance_to_zero_levelset(
        nodalcoordinates,
        refseedpoints,
        spatialseedpoints,
        seedcellids,
        levelset,
        levelsetcoeffs,
        cutmesh,
        1e-8,
        boundingradius = 4.5,
    )

    err = uniform_mesh_L2_error(
        signeddistance',
        x -> distancefunction(x)[1],
        basis,
        quad,
        cutmesh.mesh,
    )
    cellmaps = CutCell.cell_maps(cutmesh.mesh)
    den = integral_norm_on_uniform_mesh(distancefunction, quad, cellmaps, 1)

    return err[1] / den[1]
end

function convergence_rate(v,dx)
    return diff(log.(v)) ./ diff(log.(dx))
end

L, W = 1.0, 1.0
xc = [0.5, 0.5]
rad = 0.25
polyorder = 2

nelmts = [2^i + 1 for i = 1:7]

dx = 1.0 ./nelmts
err = [
    reinitialization_error(x -> circle_distance_function(x, xc, rad), ne, polyorder)
    for ne in nelmts
]
rate = convergence_rate(err,dx)

@test allapprox(rate,2*ones(length(rate)),0.05)
