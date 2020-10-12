using Test
using SparseArrays
using PolynomialBasis
using ImplicitDomainQuadrature
# using Revise
using CutCell
include("useful_routines.jl")


@test CutCell.node_to_dof_id(9,1,2) == 17
@test CutCell.node_to_dof_id(5,2,2) == 10

nodeids = [14,15,16,19,20,21,24,25,26]
edofs = CutCell.element_dofs(nodeids,2)
testedofs = [27,28,29,30,31,32,37,38,39,40,41,42,47,48,49,50,51,52]
@test allequal(edofs,testedofs)

rows,cols = CutCell.element_dofs_to_operator_dofs(1:4,[5,6,7])
testrows = vcat(1:4,1:4,1:4)
@test allequal(testrows,rows)
testcols = vcat([5,5,5,5],[6,6,6,6],[7,7,7,7])
@test allequal(testcols,cols)

a = [5  6  7
	 6  4  3
	 7  3  1.]
A = sparse(a)
rhs = zeros(3)
CutCell.apply_dirichlet_bc!(A,rhs,2,8.)
a = Array(A)
testa = [5  0. 7
	     0. 4  0.
		 7  0. 1.]
@test allapprox(a,testa)
testrhs = [-48,32,-24.]
@test allapprox(testrhs,rhs)
