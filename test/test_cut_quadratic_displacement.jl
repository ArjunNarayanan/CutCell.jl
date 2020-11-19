using Test
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCell
include("useful_routines.jl")

function displacement(x)
    u1 = x[1]^2 + 2x[1] * x[2]
    u2 = x[2]^2 + 3x[1]
    return [u1, u2]
end

function body_force(lambda, mu, x)
    b1 = -2 * (lambda + 2mu)
    b2 = -(4lambda + 6mu)
    return [b1, b2]
end

function onboundary(x, L, W)
    return x[2] ≈ 0.0 || x[1] ≈ L || x[2] ≈ W || x[1] ≈ 0.0
end

L = 1.0
W = 1.0
lambda, mu = 1.0, 2.0
xc = 1.5
radius = 1.0
penaltyfactor = 1e2
nelmts = 1
polyorder = 2
numqp = required_quadrature_order(polyorder)+2

dx = 1.0 / nelmts
penalty = penaltyfactor / dx * (lambda + mu)
stiffness = CutCell.HookeStiffness(lambda, mu, lambda, mu)

basis = TensorProductBasis(2, polyorder)
mesh = CutCell.Mesh([0.0, 0.0], [L, W], [nelmts, nelmts], basis)
levelset = InterpolatingPolynomial(1, basis)
levelsetcoeffs =
    CutCell.levelset_coefficients(x -> circle_distance_function(x, xc, radius), mesh)

cutmesh = CutCell.CutMesh(levelset, levelsetcoeffs, mesh)
cellquads = CutCell.CellQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
interfacequads = CutCell.InterfaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
facequads = CutCell.FaceQuadratures(levelset, levelsetcoeffs, cutmesh, numqp)
