using Plots

function plot_cell_quadrature_points(cellquads,cutmesh,cellsign)
    plt = plot(aspect_ratio=:equal,xlims=(0,1),ylims=(0,1))
    ncells = CutCell.number_of_cells(cutmesh)
    for cellid in 1:ncells
        cellmap = CutCell.cell_map(cutmesh,cellid)
        s = CutCell.cell_sign(cutmesh,cellid)
        if s == 0 || cellsign == s
            quad = cellquads[cellsign,cellid]
            coords = cellmap(quad.points)
            scatter!(plt,coords[1,:],coords[2,:],legend=false)
        end
    end
    return plt
end

function plot_interface_quadrature_points(interfacequads,cutmesh)
    plt = plot(aspect_ratio=:equal,xlims=(0,1),ylims=(0,1))
    ncells = CutCell.number_of_cells(cutmesh)
    for cellid in 1:ncells
        cellmap = CutCell.cell_map(cutmesh,cellid)
        s = CutCell.cell_sign(cutmesh,cellid)
        if s == 0
            quad = interfacequads[cellid]
            coords = cellmap(quad.points)
            scatter!(plt,coords[1,:],coords[2,:],legend=false)
        end
    end
    return plt
end
