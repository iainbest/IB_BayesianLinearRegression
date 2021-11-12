using IB_BayesianLinearRegression
using LinearAlgebra
using Test

@testset "IB_BayesianLinearRegression.jl" begin
    # Write your tests here.

    # BayesianLinearRegression.jl tests
    @test posterior(ones(10,2),ones(10),1.0,1.0)[3] == [11.0 10.0;10.0 11.0]

    # EvidenceApproximation.jl tests



end
