module IB_BayesianLinearRegression

using LinearAlgebra
# using IPFitting, ACE, JuLIP
using IPFitting, ACE1, JuLIP

# Write your package code here.

include("BayesianLinearRegression.jl")
include("EvidenceApproximation.jl")

end
