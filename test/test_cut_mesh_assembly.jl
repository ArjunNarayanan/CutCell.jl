lambda, mu = (1.0, 2.0)
stiffness = CutCell.plane_strain_voigt_hooke_matrix(lambda, mu)
stiffnesses = [stiffness, stiffness]

cutmeshbfs = CutCell.CutMeshBilinearForms(
    basis,
    cutmeshquads,
    stiffnesses,
    cellsign,
    cellmap,
)

@test length(cutmeshbfs.cellmatrices) == 4
testcelltomatrix = [
    3 1 1
    4 0 0
]
@test allequal(testcelltomatrix, cutmeshbfs.celltomatrix)
