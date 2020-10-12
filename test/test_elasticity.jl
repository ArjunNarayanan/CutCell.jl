using Test
# using Revise
using CutCell
include("useful_routines.jl")

l,m = 1.,2.
stiffness = CutCell.plane_strain_voigt_hooke_matrix(l,m)
teststiffness = [5. 1. 0.
                 1. 5. 0.
                 0. 0. 2.]
@test allapprox(teststiffness,stiffness)

stiffness = CutCell.HookeStiffness(l,m,l,m)
@test allapprox(teststiffness,stiffness[-1])
@test allapprox(teststiffness,stiffness[+1])
