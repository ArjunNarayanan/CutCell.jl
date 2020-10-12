struct BilinearForms
    cellmatrices::Any
    celltomatrix::Any
    ncells::Any
    function BilinearForms(cellmatrices, celltomatrix)
        nphase, ncells = size(celltomatrix)
        @assert nphase == 2
        @assert all(celltomatrix .>= 0)
        @assert all(celltomatrix .<= length(cellmatrices))
        new(cellmatrices, celltomatrix, ncells)
    end
end

function BilinearForms(basis, cellquads, stiffnesses, cellsign, cellmap)
    ncells = length(cellsign)

    uniformquad = uniform_cell_quadrature(cellquads)
    uniformbf1 = bilinear_form(basis, uniformquad, stiffnesses[+1], cellmap)
    uniformbf2 = bilinear_form(basis, uniformquad, stiffnesses[-1], cellmap)

    cellmatrices = [uniformbf1, uniformbf2]
    celltomatrix = zeros(Int, 2, ncells)

    for cellid = 1:ncells
        if cellsign[cellid] == +1
            celltomatrix[1, cellid] = 1
        elseif cellsign[cellid] == -1
            celltomatrix[2, cellid] = 2
        else
            pquad = cellquads[+1, cellid]
            pbf = bilinear_form(basis, pquad, stiffnesses[+1], cellmap)
            push!(cellmatrices, pbf)
            celltomatrix[1, cellid] = length(cellmatrices)

            nquad = cellquads[-1, cellid]
            nbf = bilinear_form(basis, nquad, stiffnesses[-1], cellmap)
            push!(cellmatrices, nbf)
            celltomatrix[2, cellid] = length(cellmatrices)
        end
    end
    return BilinearForms(cellmatrices, celltomatrix)
end

function BilinearForms(basis, cellquads, stiffnesses, cutmesh)
    cellsign = cell_sign(cutmesh)
    cellmap = cell_map(cutmesh, 1)
    return BilinearForms(basis, cellquads, stiffnesses, cellsign, cellmap)
end

function Base.getindex(cbf::BilinearForms, s, cellid)
    (s == -1 || s == +1) ||
        error("Use ±1 to index into 1st dimension of CellQuadratures, got index = $s")
    row = s == +1 ? 1 : 2
    return cbf.cellmatrices[cbf.celltomatrix[row, cellid]]
end

function Base.getindex(cbf::BilinearForms, s)
    (s == -1 || s == +1) ||
        error("Use ±1 to index into 1st dimension of CellQuadratures, got index = $s")
    idx = s == +1 ? 1 : 2
    return cbf.cellmatrices[idx]
end

function Base.show(io::IO, bf::BilinearForms)
    ncells = bf.ncells
    nuniquematrices = length(bf.cellmatrices)
    str = "BilinearForms\n\tNum. Cells: $ncells\n\tNum. Unique Cell Matrices: $nuniquematrices"
    print(io, str)
end
