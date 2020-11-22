function newton_iterate(func, grad, x0, tol; maxiter = 20)
    xk = copy(x0)
    xnext = copy(x0)
    for counter = 1:maxiter
        f = func(xk)
        g = grad(xk)
        xnext = xk - g \ f
        if norm(xnext - xk) < tol
            println("Converged in $counter iteration")
            return xnext
        else
            xk = xnext
        end
    end
    error("Did not converge in $maxiter iterations")
end

function hessian_matrix(poly, x)
    h = hessian(poly, x)
    return [
        h[1] h[2]
        h[2] h[3]
    ]
end

function saye_newton_iterate(xguess, xq, poly, tol, r; maxiter = 20, condtol = 1e5eps())
    dim = length(xguess)
    x0 = copy(xguess)
    gp = vec(gradient(poly, x0))
    l0 = gp' * (xq - x0) / (gp' * gp)

    x1 = copy(x0)
    l1 = l0

    for counter = 1:maxiter
        gp = vec(gradient(poly, x0))
        vp = poly(x0)
        hp = hessian_matrix(poly, x0)

        gf = vcat(x0 - xq + l0 * gp, vp)
        hf = [
            I+l0*hp gp
            gp' 0.0
        ]

        if inv(cond(hf)) > condtol
            δ = hf \ gf
            normδx = norm(δ[1:dim])
            if normδx > 0.5r
                δ *= 0.5r/normδx
            end
            x1 = x0 - δ[1:dim]
            l1 = l0 - δ[end]
        else
            δ1 = -vp / (gp' * gp) * gp
            l1 = gp' * (xq - x0) / (gp' * gp)
            δ2 = xq - x0 - l1 * gp
            normδ2 = norm(δ2)
            if normδ2 > 0.1r
                δ2 *= 0.1r/normδ2
            end
            x1 = x0 + δ1 + δ2
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
