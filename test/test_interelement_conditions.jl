using Test
using LinearAlgebra
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("useful_routines.jl")



polyorder = 1
numqp = 2
basis = TensorProductBasis(2, polyorder)
facequads = CutCell.face_quadratures(numqp)
lambda1, mu1 = 1.0, 2.0
lambda2, mu2 = 3.0, 4.0
stiffness = CutCell.HookeStiffness(lambda1, mu1, lambda2, mu2)
eta = 1.0
penalty = 1.0

faceids = 1:4
nbrfaceid = [CutCell.opposite_face(faceid) for faceid in faceids]

cellmap = CutCell.CellMap([0.0, 0.0], [2.0, 1.0])
jac = CutCell.jacobian(cellmap)
facedetjac = CutCell.face_determinant_jacobian(cellmap)
normals = CutCell.reference_face_normals()

uniformtop1 = [
    CutCell.interelement_traction_operators(
        basis,
        facequads[faceid],
        facequads[nbrfaceid[faceid]],
        normals[faceid],
        stiffness[+1],
        facedetjac[faceid],
        jac,
        eta,
    ) for faceid in faceids
]
uniformmassop = [
    CutCell.interelement_mass_operators(
        basis,
        facequads[faceid],
        facequads[nbrfaceid[faceid]],
        penalty * facedetjac[faceid],
    ) for faceid in faceids
]
