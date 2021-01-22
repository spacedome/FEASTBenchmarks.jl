using Distributed
using NonlinearEigenproblems
using FEASTSolver
using BenchmarkTools
using NonlinearEigenproblems.RKHelper
using Random
using LinearAlgebra
using ProgressMeter
using SharedArrays
using SuiteSparse


SuiteSparse.UMFPACK.umf_ctrl[8] = 0;
LinearAlgebra.BLAS.set_num_threads(1)

nep = nep_gallery("nlevp_native_gun")
T(x) = compute_Mder(nep, x)
