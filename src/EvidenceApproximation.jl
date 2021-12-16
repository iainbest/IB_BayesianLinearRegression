# module EvidenceApproximation

# using LinearAlgebra

export get_λ_s,γ,α,inv_β,evidence_approximation

"""
    get_λ_s(Φ)

Compute eigenvalues of ``Φ^{T} Φ``, given the design matrix Φ.

### Arguments

- `Φ::Matrix{Float64}`: design matrix

Note that within evidence approximation, we use these values multiplied by β (noise precision) - i.e. eigenvalues of ``β Φ^{T} Φ``.
Since eigenvalues of ``Φ^{T} Φ`` remain constant, calculate this only once and then multiply by updated β in iterative process.
"""
function get_λ_s(Φ)
    λ_s = eigvals(Φ' * Φ)
    return λ_s
end

"""
    γ(α,λ_is)

Compute value of γ, given weight precision α and eigenvalues λ_is of ``β Φ^{T} Φ``.

### Arguments

- `α::Float64`: weight precision parameter.
- `λ_is::Vector{Float64}`: eigenvalues of ``β Φ^{T} Φ``.

γ is used within iterative updates of both α and β, and has been factored out here for convenience following 'Pattern Recognition 
and Machine Learning, Bishop, 2006, pg168'. Alternatively, measures effective number of 'well determined parameters'.
"""
function γ(α,λ_is)
    return sum(λ_is ./ (λ_is .+ α))
end

"""
    α(γ,mN)

Compute value of α, given γ and mean of posterior distribution mN.

### Arguments

- `γ::Float64`: 
- `mN::Vector{Float64}`: mean of posterior distribution. 

Implicit equation for α hyperparameter since γ and mN depend on some initial choice for α.
"""
function α(γ,mN)
    return γ ./ (mN' * mN)[1]
end


# function inv_β(N,γ,y,mN,Φ)
#     prefactor = 1/(N-γ)
#     summation = 0
#     for i in 1:N
#         summation += (y[i] - (mN' * Φ[i,:])[1])^2
#     end
#     return prefactor*summation
# end

"""
    inv_β(γ,y,mN,Φ)  

Compute value of 1/β, given γ, target vector y, mean of posterior distribution mN and design matrix Φ.

### Arguments

- `γ::Float64`: 
- `y::Vector{Float64}`
- `mN::Vector{Float64}`: mean of posterior distribution. 
- `Φ::Matrix{Float64}`: design matrix

Implicit equation for β hyperparameter since γ and mN depend on some initial choice for β.
"""
function inv_β(γ,y,mN,Φ)
    N = size(Φ)[1]
    prefactor = 1/(N-γ)
    summation = 0
    for i in 1:N
        summation += (y[i] - (mN' * Φ[i,:])[1])^2
    end
    return prefactor*summation
end

"""
    evidence_approximation(Φ,y,init_α,init_β,num_iterations=300,tol=0.001) 

Compute list of hyperparameters following iterative, implicit updates.

### Arguments

- `Φ::Matrix{Float64}`: design matrix
- `y::Vector{Float64}`: 
- `init_α::Float64`: initial guess for α 
- `init_β::Float64`: initial guess for β
- `num_iterations::Int64=300`: number of iterations to perform
- `tol::Float64=0.001`: tolerance for convergence check

From some initial guess for α and β, perform some number of iterations, at each point re-evaluating implicit equations for α and β. 
Follows approach in 'Pattern Recognition and Machine Learning, Bishop, 2006, pg168'.
"""
function evidence_approximation(Φ,y,init_α,init_β,num_iterations=300,tol=1e-3)

    # calculate eigenvalues to be multiplied by beta
    λ_s = get_λ_s(Φ)

    # initialize some lists for alpha and beta
    alpha_list = fill(-1.0,num_iterations+1)
    beta_list = fill(-1.0,num_iterations+1)

    # set initial values for alpha and beta in lists
    alpha_list[1] = init_α
    beta_list[1] = init_β

    # set initial values for alpha and beta
    alpha = init_α
    beta = init_β

    # calculate old posterior mean (for convergence check)
    mN_old, SN_old, SN_inv_old = posterior(Φ, y, alpha, beta)

    for i in 1:num_iterations
        # multiply by beta to get correct eigenvalues
        λ_is = beta * λ_s

        # calculate gamma
        gamma = γ(alpha,λ_is)
        
        # calculate mean of posterior distribution
        mN, SN, SN_inv = posterior(Φ, y, alpha, beta) # reuse previous function to compute posterior
        
        # update implicit alpha and beta
        alpha = α(gamma,mN)
        beta = 1.0/(inv_β(gamma,y,mN,Φ))
        
        # append to lists
        alpha_list[i+1] = alpha
        beta_list[i+1] = beta

        # check for convergence
        if i != 1 && sum(abs.(mN_old - mN)) < tol
            # @show i
            # converged; return alpha and beta lists
            return alpha_list[1:i+1], beta_list[1:i+1]
        else
            # not converged, update old posterior mean and continue
            mN_old = mN
        end
        
    end

    # show final guesses for alpha, beta
    # @show alpha_list[end]
    # @show beta_list[end]

    return alpha_list,beta_list
end

# end