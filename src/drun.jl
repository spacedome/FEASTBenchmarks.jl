using Distributed
@everywhere using NonlinearEigenproblems
@everywhere using FEASTSolver
@everywhere using BenchmarkTools
@everywhere using NonlinearEigenproblems.RKHelper
@everywhere using Random
@everywhere using LinearAlgebra
# using ProgressMeter
@everywhere using SuiteSparse
@everywhere using SharedArrays
@everywhere using FEASTSolver: beyn_svd_step!, update_R!, iter_debug_print, residuals


@everywhere SuiteSparse.UMFPACK.umf_ctrl[8] = 0;
@everywhere LinearAlgebra.BLAS.set_num_threads(1)


function info(Λ, X, residuals, c, r)
    inside(x) = in_contour(x, c, r)
    in_eig = Λ[inside.(Λ)]
    in_res = residuals[inside.(Λ)]
    print("number inside : ")
    println(size(Λ[inside.(Λ)])[1])
    if sum(inside.(Λ)) > 0
        in_res_conv = in_res[in_res .<= 1e-5]
        in_eig_conv = in_eig[in_res .<= 1e-5]
        print("number inside converged : ")
        println(size(in_eig_conv)[1])
        print("max res inside: ")
        println(maximum(residuals[inside.(Λ)]))
        if size(in_res_conv, 1) > 0
            print("max res inside non spurious: ")
            println(maximum(in_res_conv))
        end
    end
    # display(in_eig_conv)
end

function dnlfeast!(T, X::AbstractMatrix{ComplexF64}, nodes::Integer, iter::Integer;
    c=complex(0.0, 0.0), r=1.0, debug=false, ϵ=10e-12, store=true, spurious=1e-5,
    factorizer=lu, left_divider=ldiv!)

    N, m₀ = size(X)
    Λ, res = zeros(ComplexF64, m₀), Array{Float64}(undef, m₀)
    θ = LinRange(π / nodes, 2 * π - π / nodes, nodes)
    Q₀, Q₁ = similar(X, ComplexF64), similar(X, ComplexF64)
    A, B = zeros(ComplexF64, m₀, m₀), zeros(ComplexF64, m₀, m₀)

    qt, rt = qr!(X)
    # X = SharedMatrix(Matrix(qt))
    # R = SharedMatrix(similar(X, ComplexF64))
    X .= Matrix(qt)
    R = similar(X, ComplexF64)

    @everywhere z_all = $([(r * exp(θ[i] * im) + c) for i = 1:nodes])

    if store
        # facts = pmap(x -> lu(T(x)), z_all) #@showprogress
        # facts = [@spawnat (procs()[i%length(procs())+1]) lu(T(z_all[i])) for i=1:nodes]
        temp_T = [@spawnat (procs()[i%length(procs())+1]) T(z_all[i]) for i=1:nodes]
        facts = [@spawnat (procs()[i%length(procs())+1]) lu(fetch(temp_T[i])) for i=1:nodes]
        foreach(finalize, temp_T)
        finalize(temp_T)
    end

    for nit = 0:iter

        # moment_reduce(a, b) = (a[1] + b[1], a[2] + b[2])
        # Q₀, Q₁ = @showprogress @distributed (moment_reduce) 
            
        moments = [
        @spawnat (procs()[i%length(procs())+1]) if nit == 0
            z = (r * exp(θ[i] * im) + c)
            if store
                (fetch(facts[i]) \ sdata(X)) .* (r * exp(θ[i] * im) / nodes)
            else
                (T(z) \ X) .* (r * exp(θ[i] * im) / nodes)
            end
        else
            z = (r * exp(θ[i] * im) + c)
            resolvent = (1 ./ (z .- Λ)) .* (r * exp(θ[i] * im) / nodes)
            if store
                rmul!(X - (fetch(facts[i]) \ sdata(R)),  Diagonal(resolvent))
            else
                rmul!(X - T(z) \ R,  Diagonal(resolvent))
            end
        end for i = 1:nodes]

        LinearAlgebra.BLAS.set_num_threads(6)

        Q₀ .= sum(fetch(moments[i]) for i=1:nodes)
        Q₁ .= sum(fetch(moments[i]).*z_all[i] for i=1:nodes)

        foreach(finalize, moments)
        finalize(moments)
        
        # moments = if nit == 0
        #     f(fact) = fact \ X 
        #     Tinv = @showprogress pmap(f, facts)
        #     for i=1:nodes
        #         Tinv[i] .*= (r * exp(θ[i] * im) / nodes)
        #     end
        #     Tinv
        # else
        #     f(fact) = fact \ R 
        #     Tinv = @showprogress pmap(f, facts)
        #     for i=1:nodes
        #         z = (r * exp(θ[i] * im) + c)
        #         resolvent = (1 ./ (z .- Λ)) .* (r * exp(θ[i] * im) / nodes)
        #         Tinv[i] .= X - Tinv[i]
        #         rmul!(Tinv[i],  Diagonal(resolvent))
        #     end
        #     Tinv
        # end
        # display(moments)
        # Q₀ .= sum(moments)
        # for i=1:nodes
        #     moments[i] .*= (r * exp(θ[i] * im) + c)
        # end
        # Q₁ .= sum(moments)


		beyn_svd_step!(Q₀, Q₁, A, B, X, Λ)

        update_R!(X, R, Λ, T)
        res .= residuals(R, Λ, T)

        LinearAlgebra.BLAS.set_num_threads(1)

        if debug
            iter_debug_print(nit, Λ, res, c, r, spurious)
        end

		res_inside = res[in_contour.(Λ, c, r)]
        if size(res_inside, 1) > 0 && maximum(res_inside) < ϵ
            break
        end
		if nit > 1 && sum(res_inside .< spurious) > 0 && maximum(res_inside[res_inside .< spurious]) < ϵ
			break
		end
    end
    
    foreach(finalize, facts)
    finalize(facts)
    foreach(finalize, [R, A, B, Q₀, Q₁])

    normalize!(X)
    Λ, X, res
end


@everywhere nep = nep_gallery("nlevp_native_gun")
@everywhere T(x) = compute_Mder(nep, x)

n = size(nep, 1)
C = complex(62500.0, 0.0)
R = 50000

b_samples = 5

suite = BenchmarkGroup()

suite["nlfeast"] = BenchmarkGroup(["gun"])
# suite["nleigs"] = BenchmarkGroup(["gun"])

for n_nodes = 12:6:12
    suite["nlfeast"][string(n_nodes)] = @benchmarkable dnlfeast!(T, rand(ComplexF64,n,32), $(n_nodes), 20, c=C, r=R, debug=false, ϵ=10e-10, store=true) samples=b_samples seconds=300 evals=1
end

# suite["nleigs"]["static"] = @benchmarkable begin
#     verbose = 2 #displaylevel
#     nep, Σ, Ξ, v, nodes, funres = gun_init()
#     lambda, X, res, solution_info = nleigs(nep, Σ, Ξ=Ξ, logger=verbose > 0 ? 1 : 0, minit=70, maxit=100, v=v, nodes=nodes, static=true, errmeasure=funres, return_details=verbose > 1)
# end samples=b_samples seconds=300 evals=1

# suite["nleigs"]["r2"] = @benchmarkable begin
#     verbose = 2 #displaylevel
#     nep, Σ, Ξ, v, nodes, funres = gun_init()
#     lambda, X, res, solution_info = nleigs(nep, Σ, Ξ=Ξ, logger=verbose > 0 ? 1 : 0, minit=60, maxit=100, v=v, nodes=nodes, errmeasure=funres, return_details=verbose > 1)
# end samples=b_samples seconds=300 evals=1

results = run(suite, verbose=true)
display(results)
med = median(results)
display(med)

