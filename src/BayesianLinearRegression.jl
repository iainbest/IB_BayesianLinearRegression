# module BayesianLinearRegression

export design_matrix, posterior, posterior_predictive, design_matrix2

# using IPFitting, ACE, JuLIP
# using LinearAlgebra

"""
    design_matrix(cfgs,basis)

Computes N*M design matrix given N atomic configurations (observations) and length(M) basis 

### Arguments

- `cfgs::Vector{Dat}`: atomic configs from database with DFT energies, forces, virials 
- `basis::RPIBasis`: type specific for ACE potential. Basis functions.

Note that we don't extract individual basis functions, but instead get basis evaluated on specific configuration/atoms object 
by calling energy(basis,at), with B the basis and at is the configuration - this returns a vector of observations, i.e. a row 
in our design matrix.
"""
function design_matrix(cfgs::Vector{Dat},basis::JuLIP.MLIPs.IPSuperBasis{JuLIP.MLIPs.IPBasis})
    # initialize matrix of correct size
    Φ = Matrix{Float64}(undef, length(cfgs), length(basis))
    # for ACE potential, evaluate basis functions for each config/atoms object
    for (count,i) in enumerate(cfgs)
        Φ[count,:] = energy(basis, i.at)
    end
    return Φ
end

"""
    design_matrix(cfgs,basis)

If configs are just atoms objects -> use this form (not Dats?)

    """
function design_matrix(cfgs_atoms::Vector{Atoms{Float64}},basis::JuLIP.MLIPs.IPSuperBasis{JuLIP.MLIPs.IPBasis})
    # initialize matrix of correct size
    Φ = Matrix{Float64}(undef, length(cfgs_atoms), length(basis))
    # for ACE potential, evaluate basis functions for each config/atoms object
    for (count,i) in enumerate(cfgs_atoms)
        Φ[count,:] = energy(basis, i)
    end
    return Φ
end


"""
    posterior(Φ, y, α, β)

Computes mean and covariance matrix of the posterior distribution.

### Arguments

- `Φ::Matrix{Float64}`: Design matrix (N*M)
- `y::Vector{Float64}`: length(N). Target variable (you want to predict?)
- `α::Float64`: hyperparameter for precision of weights (before we see data, assume weights must be around zero with precision α)
- `β::Float64`: hyperparameter for noise precision of data

"""
function posterior(Φ, y, α, β)
    S_N_inv = α * I(size(Φ, 2)) + β * Φ' * Φ
    S_N = inv(S_N_inv)
    m_N = β * S_N * Φ' * y
    return m_N, S_N, S_N_inv
end;


"""
    posterior_predictive(Φ_test, m_N, S_N, β)

Computes mean and variances of the posterior predictive distribution. (NOTE: WIP)

### Arguments

- `Φ_test::Matrix{Float64}`: Design matrix evaluated on test input (CHECK THIS)
- `m_N::Vector{Float64}`: Mean of posterior distribution
- `S_N::Float64`: Covariance of posterior distribution
- `β::Float64`: hyperparameter for noise precision of data

Returns predicted output vector, epistemic variance and predictive (total) variance. 

NOTE: only computes variances (diagonal elements of covariance matrix) and sums them - need to extend to full covariance matrix
"""
function posterior_predictive(Φ_test, m_N, S_N, β)
    y = Φ_test * m_N
    # Only compute variances (diagonal elements of covariance matrix)
#     y_epi = sum(Φ_test * S_N * Φ_test)
    y_epi = sum((Φ_test * S_N) .* Φ_test,dims=2)[:,1]
    y_var = 1/β .+ y_epi
    return y, y_epi, y_var
end

# end