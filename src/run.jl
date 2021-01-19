using NonlinearEigenproblems
using FEASTSolver
using BenchmarkTools
using NonlinearEigenproblems.RKHelper
using Random
using LinearAlgebra
using SuiteSparse

SuiteSparse.UMFPACK.umf_ctrl[8] = 0;
LinearAlgebra.BLAS.set_num_threads(1)

function gun_init()
    nep = gun_nep()

    gam = 300^2 - 200^2
    mu = 250^2
    sigma2 = 108.8774
    xmin = gam*(-1) + mu
    xmax = gam*1 + mu

    # define target set Σ
    npts = 1000
    halfcircle = xmin .+ (xmax-xmin) * (cis.(range(0, stop = pi, length = round(Int, pi/2*npts) + 2)) / 2 .+ .5)
    Σ = [halfcircle; xmin]

    # sequence of interpolation nodes
    Z = [2/3, (1+im)/3, 0, (-1+im)/3, -2/3]
    nodes = gam*Z .+ mu

    # define the set of pole candidates
    Ξ = -10 .^ range(-8, stop = 8, length = 10000) .+ sigma2^2

    # options
    Random.seed!(1)
    v = randn(size(nep, 1)) .+ 0im

    funres = (λ, v) -> gun_residual(λ, v, nep.nep1.A..., nep.nep2.spmf.A...)

    return nep, Σ, Ξ, v, nodes, funres
end

function gun_nep()
    nep = nep_gallery("nlevp_native_gun")
    K, M = get_Av(nep.nep1);
    W1, W2 = get_Av(nep.nep2);
    c1 = LowRankMatrixAndFunction(W1, get_fv(nep.nep2)[1])
    c2 = LowRankMatrixAndFunction(W2, get_fv(nep.nep2)[2])
    return SumNEP(PEP([K, M]), LowRankFactorizedNEP([c1, c2]))
end

function gun_residual(λ, v, K, M, W1, W2)
    # constants
    sigma1 = 0
    sigma2 = 108.8774

    nK = 1.474544889815002e+05   # opnorm(K, 1)
    nM = 2.726114618171165e-02   # opnorm(M, 1)
    nW1 = 2.328612251920476e+00  # opnorm(W1, 1)
    nW2 = 3.793375498194695e+00  # opnorm(W2, 1)

    # Denominator
    den = nK + abs(λ) * nM + sqrt(abs(λ-sigma1^2)) * nW1 + sqrt(abs(λ-sigma2^2)) * nW2

    # 2-norm of A(lambda)*x
    norm((K + M*λ + W1*im*sqrt(λ) + W2*im*sqrt(λ - sigma2^2)) * v) / den
end


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


nep = nep_gallery("nlevp_native_gun")
T(x) = compute_Mder(nep, x)

n = size(nep, 1)
C = complex(62500.0, 0.0)
R = 50000

b_samples = 3

suite = BenchmarkGroup()

suite["nlfeast"] = BenchmarkGroup(["gun"])
suite["nleigs"] = BenchmarkGroup(["gun"])

for n_nodes = 8:8:64
    suite["nlfeast"][string(n_nodes)] = @benchmarkable nlfeast!(T, rand(ComplexF64,n,32), $(n_nodes), 20, c=C, r=R, debug=false, ϵ=10e-10, store=true) samples=b_samples seconds=300 evals=1
end

suite["nleigs"]["static"] = @benchmarkable begin
    verbose = 0 #displaylevel
    nep, Σ, Ξ, v, nodes, funres = gun_init()
    lambda, X, res, solution_info = nleigs(nep, Σ, Ξ=Ξ, logger=verbose > 0 ? 1 : 0, minit=70, maxit=100, v=v, nodes=nodes, static=true, errmeasure=funres, return_details=verbose > 1)
end samples=b_samples seconds=300 evals=1

suite["nleigs"]["r2"] = @benchmarkable begin
    verbose = 0 #displaylevel
    nep, Σ, Ξ, v, nodes, funres = gun_init()
    lambda, X, res, solution_info = nleigs(nep, Σ, Ξ=Ξ, logger=verbose > 0 ? 1 : 0, minit=60, maxit=100, v=v, nodes=nodes, errmeasure=funres, return_details=verbose > 1)
end samples=b_samples seconds=300 evals=1

results = run(suite, verbose=true)
display(results)
med = median(results)
display(med)

