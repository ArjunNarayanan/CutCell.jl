using Test
using LinearAlgebra
using IntervalArithmetic
using PolynomialBasis
using ImplicitDomainQuadrature
using Revise
using CutCell
include("useful_routines.jl")


function test_two_cell_coherent_interface_on_linear_element_boundary()
    cellmap = CutCell.CellMap([0.0, 0.0], [1.0, 1.0])
    facedetjac = CutCell.face_determinant_jacobian(cellmap)

    numqp = 2
    basis = TensorProductBasis(2, 1)
    vquad = tensor_product_quadrature(2, 2)
    rfquad = CutCell.extend_to_face(tensor_product_quadrature(1, numqp), 2)
    lfquad = CutCell.extend_to_face(tensor_product_quadrature(1, numqp), 4)
    normal1 = repeat([1.0, 0.0], 1, numqp)
    normal2 = -normal1

    lambda, mu = (1.0, 2.0)
    penalty = 1.
    dx = 0.1
    e11 = dx / 2.0
    e22 = -lambda / (lambda + 2mu) * e11
    dy = e22
    s11 = (lambda + 2mu) * e11 + lambda * e22
    stiffness = plane_strain_voigt_hooke_matrix(lambda, mu)

    bf = CutCell.bilinear_form(basis, vquad, stiffness, cellmap)


    top12 = CutCell.coherent_traction_operator(basis,rfquad,lfquad,normal2,stiffness,cellmap)
    top21 = CutCell.coherent_traction_operator(basis,lfquad,rfquad,normal1,stiffness,cellmap)

    mass11 = penalty*CutCell.mass_matrix(basis,rfquad,rfquad,0.5,2)
    mass12 = penalty*CutCell.mass_matrix(basis,rfquad,lfquad,0.5,2)
    mass21 = penalty*CutCell.mass_matrix(basis,lfquad,rfquad,0.5,2)
    mass22 = penalty*CutCell.mass_matrix(basis,lfquad,lfquad,0.5,2)

    K = [
        (bf+mass11)        (top12-mass12)
        (top21-mass21)     (bf+mass22)
    ]

    rhs = zeros(16)

    CutCell.apply_dirichlet_bc!(K, rhs, 1, 0.0)
    CutCell.apply_dirichlet_bc!(K, rhs, 2, 0.0)
    CutCell.apply_dirichlet_bc!(K, rhs, 3, 0.0)

    CutCell.apply_dirichlet_bc!(K, rhs, 13, dx)
    CutCell.apply_dirichlet_bc!(K, rhs, 15, dx)

    sol = K\rhs
    disp = reshape(sol,2,:)

    testdisp = [0. 0. dx/2 dx/2 dx/2 dx/2 dx dx
                0. dy 0.   dy   0.   dy   0. dy]
    @test allapprox(disp,testdisp,1e-15)
end



function test_linear_cut_cell_coherent_interface()
    cellmap = CutCell.CellMap([0.,0.],[1.,1.])
    invjac = CutCell.inverse_jacobian(cellmap)
    numqp = 4

    basis = TensorProductBasis(2,1)
    quad1d = ReferenceQuadratureRule(numqp)
    levelset = InterpolatingPolynomial(1,basis)
    coords = cellmap(basis.points)

    x0 = [0.5,0.]
    theta = 20.
    normal = normal_from_angle(theta)
    levelsetcoeffs = plane_distance_function(coords,normal,x0)
    update!(levelset,levelsetcoeffs)

    box = IntervalBox(-1..1,2)
    quad1 = area_quadrature(levelset,-1,box,quad1d)
    quad2 = area_quadrature(levelset,+1,box,quad1d)
    fquad = surface_quadrature(levelset,box,quad1d)

    normal1 = CutCell.levelset_normal(levelset,fquad.points,invjac)
    normal2 = -normal1

    lambda, mu = (1.0, 2.0)
    penalty = 1.
    dx = 0.1
    e11 = dx / 1.0
    e22 = -lambda / (lambda + 2mu) * e11
    dy = e22
    s11 = (lambda + 2mu) * e11 + lambda * e22
    stiffness = plane_strain_voigt_hooke_matrix(lambda, mu)

    bf1 = CutCell.bilinear_form(basis,quad1,stiffness,cellmap)
    bf2 = CutCell.bilinear_form(basis,quad2,stiffness,cellmap)

    top12 = CutCell.coherent_traction_operator(basis,fquad,normal2,stiffness,cellmap)
    top21 = CutCell.coherent_traction_operator(basis,fquad,normal1,stiffness,cellmap)

    facescale = CutCell.scale_area(cellmap,normal1)
    mass = penalty*CutCell.mass_matrix(basis,fquad,facescale,2)

    K = [(bf1+mass)    (top12-mass)
         (top21-mass)  (bf2+mass)]

    rhs = zeros(16)

    CutCell.apply_dirichlet_bc!(K,rhs,1,0.0)
    CutCell.apply_dirichlet_bc!(K,rhs,2,0.0)
    CutCell.apply_dirichlet_bc!(K,rhs,3,0.0)

    CutCell.apply_dirichlet_bc!(K,rhs,13,dx)
    CutCell.apply_dirichlet_bc!(K,rhs,15,dx)

    sol = K\rhs
    disp = reshape(sol,2,:)

    testdisp = [0. 0. dx dx 0. 0. dx dx
                0. dy 0. dy 0. dy 0. dy]

    @test allapprox(disp,testdisp,1e-15)
end


function test_curved_interface_linear_displacement()
    cellmap = CutCell.CellMap([0.,0.],[1.,1.])
    invjac = CutCell.inverse_jacobian(cellmap)
    numqp = 4

    basis = TensorProductBasis(2,2)
    quad1d = ReferenceQuadratureRule(numqp)
    levelset = InterpolatingPolynomial(1,basis)
    coords = cellmap(basis.points)

    rad = 1.0
    center = [1.5,0.5]
    levelsetcoeffs = circle_distance_function(coords,center,rad)
    update!(levelset,levelsetcoeffs)

    box = IntervalBox(-1..1,2)
    quad1 = area_quadrature(levelset,-1,box,quad1d)
    quad2 = area_quadrature(levelset,+1,box,quad1d)
    fquad = surface_quadrature(levelset,box,quad1d)

    normal1 = CutCell.levelset_normal(levelset,fquad.points,invjac)
    normal2 = -normal1

    lambda, mu = (1.0, 2.0)
    penalty = 1.
    dx = 0.1
    e11 = dx / 1.0
    e22 = -lambda / (lambda + 2mu) * e11
    dy = e22
    s11 = (lambda + 2mu) * e11 + lambda * e22
    stiffness = plane_strain_voigt_hooke_matrix(lambda, mu)

    bf1 = CutCell.bilinear_form(basis,quad1,stiffness,cellmap)
    bf2 = CutCell.bilinear_form(basis,quad2,stiffness,cellmap)

    top12 = CutCell.coherent_traction_operator(basis,fquad,normal2,stiffness,cellmap)
    top21 = CutCell.coherent_traction_operator(basis,fquad,normal1,stiffness,cellmap)

    facescale = CutCell.scale_area(cellmap,normal1)
    mass = penalty*CutCell.mass_matrix(basis,fquad,facescale,2)

    K = [(bf1+mass)    (top12-mass)
         (top21-mass)  (bf2+mass)]

    rhs = zeros(36)

    CutCell.apply_dirichlet_bc!(K,rhs,1,0.0)
    CutCell.apply_dirichlet_bc!(K,rhs,2,0.0)
    CutCell.apply_dirichlet_bc!(K,rhs,3,0.0)
    CutCell.apply_dirichlet_bc!(K,rhs,5,0.0)

    CutCell.apply_dirichlet_bc!(K,rhs,31,dx)
    CutCell.apply_dirichlet_bc!(K,rhs,33,dx)
    CutCell.apply_dirichlet_bc!(K,rhs,35,dx)

    sol = K\rhs
    disp = reshape(sol,2,:)

    testdisp = [0. 0.   0.   dx/2  dx/2  dx/2  dx  dx  dx
                0. dy/2 dy   0.    dy/2  dy    0. dy/2 dy]
    testdisp = repeat(testdisp,outer=(1,2))

    @test allapprox(disp,testdisp,1e-14)
end
