module LatentSDEBayes

using ComponentArrays
using Distributions
using SpecialFunctions
using StableHashTraits
using Random
using LinearAlgebra

# 1. Include core files
include("Interfaces.jl")
include("StateViews.jl")
include("RNG.jl")
include("Parameters.jl")

# (You will uncomment these later as you build them)
# include("Propagators/Propagators.jl")
# include("ParticleFilter/ParticleFilter.jl")
# include("Inference/Inference.jl")
# include("Simulation.jl")

# 2. Export the user-facing API defined in your blueprint
export AbstractLatentModel, AbstractObsModel, AbstractStepper
export state_template, drift!, diffusion!, loglik_step, init_obs_state
export noise_type, noise_dim, state_dim
export DiagonalNoise, CorrelatedNoise
# export EMStepper, PFConfig, MCMCConfig
# export pf_loglik, pmmh, make_rng, RNGConfig
# export ParameterSpec, pack, unpack

end