function hessian_matrix(poly, x)
    h = hessian(poly, x)
    return [
        h[1] h[2]
        h[2] h[3]
    ]
end


function saye_newton_iterate(
    xguess,
    xq,
    func,
    grad,
    hess,
    cellmap,
    tol,
    r;
    maxiter = 20,
    condtol = 1e5eps(),
)
    dim = length(xguess)
    jac = jacobian(cellmap)

    x0 = copy(xguess)
    gp = grad(x0)
    l0 = gp' * ((xq - cellmap(x0)) .* jac) / (gp' * gp)

    x1 = copy(x0)
    l1 = l0

    for counter = 1:maxiter
        vp = func(x0)
        gp = grad(x0)
        hp = hess(x0)

        gf = vcat(((cellmap(x0) - xq) .* jac) + l0 * gp, vp)
        hf = [
            diagm(jac .^ 2)+l0*hp gp
            gp' 0.0
        ]

        if inv(cond(hf)) > condtol
            δ = hf \ gf
            normδx = norm(δ[1:dim])
            if normδx > 0.5r
                δ *= 0.5r / normδx
            end
            x1 = x0 - δ[1:dim]
            l1 = l0 - δ[end]
        else
            error("Chopp method not implemented")
        end

        if norm(x1 - xguess) > r
            error("Did not converge in ball of radius $r")
        elseif norm(x1 - x0) < tol
            return x1, counter
        else
            x0 = x1
            l0 = l1
        end
    end
    error("Did not converge in $maxiter iterations")
end

function project_on_zero_levelset(
    xguess,
    func,
    grad,
    tol,
    r;
    maxiter = 20,
    normtol = 1e-8,
    perturbation = 0.1,
)
    x0 = copy(xguess)
    x1 = copy(x0)
    dim = length(xguess)

    for counter = 1:maxiter
        vf = func(x0)
        gf = grad(x0)

        if norm(gf) < normtol
            x0 = x0 + perturbation * (rand(dim) .- 0.5)
        else
            δ = vf / (gf' * gf) * gf
            normδ = norm(δ)
            if normδ > 0.5r
                δ *= 0.5r / normδ
            end

            x1 = x0 - δ

            if abs(func(x1)) < tol
                flag = norm(x1 - xguess) < r
                return x1, flag
            else
                x0 = x1
            end
        end
    end
    error("Did not converge after $maxiter iterations")
end

function reference_seed_points(n)
    @assert n > 0
    xrange = range(-1.0, stop = 1.0, length = n + 2)
    points = ImplicitDomainQuadrature.tensor_product_points(xrange[2:n+1]', xrange[2:n+1]')
end

function seed_cell_zero_levelset(xguess, func, grad; tol = 1e-12, r = 2.5)
    dim, nump = size(xguess)
    pf = [project_on_zero_levelset(xguess[:, i], func, grad, tol, r) for i = 1:nump]
    flags = [p[2] for p in pf]
    valididx = findall(flags)
    validpoints = [p[1] for p in pf[valididx]]
    return hcat(validpoints...)
end

function seed_zero_levelset(nump, levelset, levelsetcoeffs, cutmesh)
    refpoints = reference_seed_points(nump)
    refseedpoints = []
    spatialseedpoints = []
    seedcellids = Int[]
    cellsign = cell_sign(cutmesh)
    cellids = findall(cellsign .== 0)
    for cellid in cellids
        cellmap = cell_map(cutmesh, cellid)
        nodeids = nodal_connectivity(cutmesh.mesh, cellid)
        update!(levelset, levelsetcoeffs[nodeids])

        xk = seed_cell_zero_levelset(refpoints, levelset, x -> vec(gradient(levelset, x)))

        numseedpoints = size(xk)[2]

        append!(seedcellids, repeat([cellid], numseedpoints))
        push!(refseedpoints, xk)
        push!(spatialseedpoints, cellmap(xk))
    end

    refseedpoints = hcat(refseedpoints...)
    spatialseedpoints = hcat(spatialseedpoints...)
    return refseedpoints, spatialseedpoints, seedcellids
end

function distance_to_zero_levelset(
    querypoints,
    refseedpoints,
    spatialseedpoints,
    seedcellids,
    levelset,
    levelsetcoeffs,
    mesh,
    tol;
    boundingradius = 2.5,
)

    dim,numquerypoints = size(querypoints)
    signeddistance = zeros(numquerypoints)
    tree = KDTree(spatialseedpoints)
    seedidx, seeddists = nn(tree, querypoints)

    for (idx, sidx) in enumerate(seedidx)
        xguess = refseedpoints[:, sidx]
        xquery = querypoints[:, idx]
        guesscellid = seedcellids[sidx]
        cellmap = cell_map(mesh, guesscellid)
        update!(levelset, levelsetcoeffs[nodal_connectivity(mesh, guesscellid)])

        refclosestpt, iter = saye_newton_iterate(
            xguess,
            xquery,
            levelset,
            x -> vec(gradient(levelset, x)),
            x -> hessian_matrix(levelset, x),
            cellmap,
            tol,
            boundingradius,
        )

        spatialclosestpt = cellmap(refclosestpt)

        g = vec(transform_gradient(gradient(levelset,refclosestpt),jacobian(cellmap)))
        s = sign(g'*(xquery - spatialclosestpt))

        signeddistance[idx] = s*norm(spatialclosestpt - xquery)
    end
    return signeddistance
end
