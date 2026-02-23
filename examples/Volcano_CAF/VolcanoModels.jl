# examples/Volcano_CAF/VolcanoModels.jl

module VolcanoModels

using ComponentArrays
using LatentSDEBayes
using SpecialFunctions: loggamma

# We export these so 'using .VolcanoModels' in the notebook makes them visible
export VolcanoSDEParams, VolcanoObsParams, VolcanoParams
export VolcanoLatent, VolcanoObs, fluxes_scalar

# =========================================
# 1. PARAMETERS (Nested Design)
# =========================================

Base.@kwdef struct VolcanoSDEParams{T}
    beta::T    = 1.2
    gamma::T   = 69.0
    kappa::T   = 0.57
    I_ext::T   = 0.025
    epsilon::T = 4.3
    m::T       = 34.5
    sigma_P::T = 0.0027
    sigma_delta::T = 0.005
    M::T       = 10.0      
end

Base.@kwdef struct VolcanoObsParams{T}
    lake_base::T    = 50.0
    lake_scale::T   = 9.0
    lake_sigma::T   = 0.02

    degas_scale::T  = 690.0
    degas_decay::T  = 0.23
    degas_thresh::T = 1e-4
    degas_sigma::T  = 0.001
    degas_nu::T     = 4.0
end

Base.@kwdef struct VolcanoParams{T}
    sde::VolcanoSDEParams{T} = VolcanoSDEParams{T}()
    obs::VolcanoObsParams{T} = VolcanoObsParams{T}()
end

# =========================================
# 2. LATENT MODEL (The SDE)
# =========================================
struct VolcanoLatent <: AbstractLatentModel end

LatentSDEBayes.state_template(::VolcanoLatent) = ComponentArray(P = 0.0, delta = 0.0)

function LatentSDEBayes.drift!(du, ::VolcanoLatent, u, p, t)
    inv_beta = 1.0 / p.sde.beta
    Pc = -0.5 * (1.0 + inv_beta)
    cubic = (u.P + 1.0) * (u.P + inv_beta) * (2*u.P + 1.0 + inv_beta)

    # Slow dynamics: dP/dt
    du.P = -2 * p.sde.gamma * cubic - p.sde.kappa * (u.delta - 0.5) + p.sde.I_ext

    # Fast dynamics: d(delta)/dt
    arg = p.sde.m * (u.P - Pc)
    sig = arg >= 0 ? 1/(1+exp(-arg)) : (e=exp(arg); e/(1+e))
    du.delta = p.sde.epsilon * u.delta * (1 - u.delta) * (sig - u.delta)
    return nothing
end

function LatentSDEBayes.diffusion!(g, ::VolcanoLatent, u, p, t)
    g.P     = p.sde.sigma_P
    g.delta = p.sde.sigma_delta * u.delta * (1 - u.delta)
    return nothing
end

# =========================================
# 3. OBSERVATION MODEL (The Likelihood)
# =========================================

function fluxes_scalar(P, delta, beta, M)
    delta = clamp(delta, 1e-6, 1.0 - 1e-6) 
    d2 = delta^2; d4 = delta^4; ln_d = log(delta)
    Q1 = (pi/(8*beta)) * (4*(1-beta)*d4*ln_d - 2*beta*d2 + (2*beta-M)*d4 + P*(-2*beta*d2 + beta*(2-M)*d4))
    Q2 = (pi/(8*beta)) * (4*(beta-1)*d4*ln_d - (3*beta-2)*d4 + 2*(2*beta-1)*d2 - beta + P*(-beta*(d2-1)^2))
    return Q1, Q2
end

struct VolcanoObs <: AbstractObsModel end

function LatentSDEBayes.init_obs_state(::VolcanoObs, theta)
    return (q2_prev = 0.0, s_prev = 0.0)
end

function LatentSDEBayes.loglik_step(::VolcanoObs, data_k, u, p, obs_st)
    q1, q2 = fluxes_scalar(u.P, u.delta, p.sde.beta, p.sde.M) 
    QT = q1 + q2

    # Lake height channel
    mu_H = p.obs.lake_base + p.obs.lake_scale * QT
    z_H  = (data_k.yH - mu_H) / p.obs.lake_sigma
    ll   = -0.5*z_H*z_H - log(p.obs.lake_sigma) - 0.5*log(2*pi)

    # Degassing channel
    drop  = obs_st.q2_prev - q2
    drive = max(0.0, drop - p.obs.degas_thresh)
    s_new = obs_st.s_prev*(1 - p.obs.degas_decay) + drive * p.obs.degas_scale
    
    nu = p.obs.degas_nu
    sd = p.obs.degas_sigma
    z_D = (data_k.yD - s_new) / sd
    
    ll += loggamma((nu+1)/2) - loggamma(nu/2) - 0.5*log(nu*pi) - log(sd) - ((nu+1)/2)*log1p(z_D*z_D/nu)

    return (ll, (q2_prev = q2, s_prev = s_new))
end

end # module