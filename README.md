# LatentSDEBayes.jl

**Bayesian inference for latent stochastic differential equations via particle MCMC.**

## Why This Package?

In many physical systems — volcanic conduits, subsurface reservoirs, ocean circulation, neural dynamics — the governing processes happen in a **latent space** we cannot observe directly. We measure proxy signals at the surface (tilt, gas flux, seismicity) that are noisy, nonlinear transformations of the hidden state. The scientific models describe the physics of the latent dynamics, but the data live in observation space. The central challenge is: **how do we infer the parameters of a model we can't see, using only indirect, noisy measurements?**

The answer is a **state-space model** — a two-layer probabilistic framework:

$$\text{Latent dynamics:} \qquad d\mathbf{x} = f(\mathbf{x},\,\theta)\,dt + \sigma(\mathbf{x},\,\theta)\,dW_t$$

$$\text{Observations:} \qquad y_k \sim p\!\left(y \;\middle|\; \mathbf{x}(t_k),\,\theta\right)$$

$$\text{Goal:} \qquad p(\theta \mid y_1, \dots, y_T) \;=\; ?$$

The **latent layer** is an SDE describing the hidden physics (e.g., pressure and annulus thickness inside a volcanic conduit). The **observation layer** maps the hidden state to measurable quantities with noise (e.g., lava lake height, SO₂ flux). Neither layer alone is sufficient — the SDE alone has no data, and the data alone has no physics. The state-space model couples them, and Bayes' theorem tells us how to update our beliefs about $\theta$ given the observations. The difficulty is that computing the marginal likelihood $p(y_{1:T} \mid \theta)$ requires integrating over all possible latent trajectories — an intractable integral in continuous state space. This package solves that using **sequential Monte Carlo** (particle filtering) to approximate the integral, wrapped inside **MCMC** to sample the posterior.

**LatentSDEBayes.jl** lets you define both layers in Julia, then handles the hard part. You write the physics; the package runs the inference.

**What makes it different:** named state access everywhere (`u.P`, `u.delta`, `theta.sde.gamma` — never `u[1]`, `p[3]`), zero-allocation particle filter inner loop, fully typed workspaces with no `Vector{Any}`, and deterministic RNG so every run is reproducible from a single seed.

## Architecture

```
┌─────────────────────────── You define ───────────────────────────┐
│                                                                  │
│   drift!(du, model, u, θ, t) ──┐                                │
│                                 ├──→ Latent SDE                  │
│   diffusion!(g, model, u, θ, t)┘        │                       │
│                                    propagate!                    │
│                                   (EMStepper)                    │
│                                          ↓                       │
│                                    ┌──────────┐                  │
│                                    │ Particle  │                 │
│                                    │  Cloud    │                 │
│                                    └────┬─────┘                  │
│                                         │ weight                 │
│   loglik_step(obs, yₖ, uₖ, θ, state) ──┘                       │
│                                                                  │
│         ┌──────────┐    ┌────────┐    ┌──────────┐               │
│         │ pf_loglik│───→│  PMMH  │───→│ Posterior │              │
│         │ (filter) │    │ (mcmc) │    │  Samples  │              │
│         └──────────┘    └────────┘    └──────────┘               │
│                                                                  │
└─────────────────────── Package handles ──────────────────────────┘
```

**Source layout:**

| Directory | What it does |
|:----------|:------------|
| `Interfaces.jl` | Abstract types, traits (`DiagonalNoise`, `CorrelatedNoise`), required method signatures |
| `StateViews.jl` | Zero-copy named views over raw particle vectors |
| `Parameters.jl` | `FieldPath`, `ParameterSpec`, `CompiledSpec` — pack/unpack/transform for MCMC |
| `Propagators/` | `EMStepper` (Euler-Maruyama); plug in your own via `AbstractStepper` |
| `ParticleFilter/` | Bootstrap PF with ESS-based resampling, preallocated `PFWorkspace` |
| `Inference/` | PMMH sampler, random-walk proposals, chain storage with named access |

## Quick Start

```julia
using Pkg; Pkg.activate("path/to/LatentSDEBayes.jl")
using LatentSDEBayes, ComponentArrays, Random, Distributions
```

### 1. Define your SDE

```julia
# The model struct is a zero-size dispatch tag — it carries no data.
# Its type tells the package which drift!/diffusion! methods to call.
struct MyModel <: AbstractLatentModel end

# State template: defines the DIMENSION and NAMES of your latent variables.
# This model has 2 latent states: P (pressure) and δ (annulus thickness).
LatentSDEBayes.state_template(::MyModel) = ComponentArray(P=0.0, δ=0.0)

# Noise structure: DiagonalNoise() means each state has independent Wiener noise.
# Use CorrelatedNoise() if your noise covariance is a full matrix.
LatentSDEBayes.noise_type(::MyModel) = DiagonalNoise()

function LatentSDEBayes.drift!(du, ::MyModel, u, θ, t)
    # du is the output buffer (same shape as u), θ holds ALL parameters.
    # Access state as u.P, u.δ and parameters as θ.sde.γ, θ.sde.κ, etc.
    du.P = -2θ.sde.γ * (u.P + 1) * (u.P + 1/θ.sde.β) * (2u.P + 1 + 1/θ.sde.β) -
            θ.sde.κ * (u.δ - 0.5) + θ.sde.I
    Pc = -0.5(1 + 1/θ.sde.β)
    du.δ = θ.sde.ε * u.δ * (1 - u.δ) * (1/(1 + exp(-θ.sde.m*(u.P - Pc))) - u.δ)
    return nothing
end

function LatentSDEBayes.diffusion!(g, ::MyModel, u, θ, t)
    # g is the output buffer for noise amplitudes (same shape as u).
    g.P = θ.sde.σP                          # constant noise amplitude
    g.δ = θ.sde.σδ * u.δ * (1 - u.δ)        # state-dependent (vanishes at boundaries)
    return nothing
end
```

Parameters live in nested `@kwdef` structs — you choose how many and what they're called:

```julia
# SDE parameters: as many as your physics needs
Base.@kwdef struct SDEParams{T}
    β::T  = 1.2;   γ::T  = 69.0;  κ::T = 0.57
    I::T  = 0.025; ε::T  = 4.3;   m::T = 34.5
    σP::T = 0.0027; σδ::T = 0.005
end

# Observation parameters: as many as your measurement model needs
Base.@kwdef struct ObsParams{T}
    base::T  = 50.0;  scale::T = 9.0;  σ::T = 0.02
end

# Top-level container (the package always receives this)
Base.@kwdef struct MyParams{T}
    sde::SDEParams{T} = SDEParams{T}()
    obs::ObsParams{T} = ObsParams{T}()
end
```

### 2. Define your observation model

```julia
# Like the SDE model, this is a dispatch tag.
struct MyObs <: AbstractObsModel end

# If your observation has memory (e.g., a leaky integrator), define a state struct.
# If not, just return nothing from init_obs_state.
struct MyObsState{T}
    s_prev::T    # example: accumulated degassing signal
end

LatentSDEBayes.init_obs_state(::MyObs, θ) = MyObsState(0.0)

function LatentSDEBayes.loglik_step(::MyObs, yₖ, uₖ, θ, state)
    # Map latent state → predicted observation
    predicted = θ.obs.base + θ.obs.scale * uₖ.P
    z = (yₖ - predicted) / θ.obs.σ
    ll = -0.5z^2 - log(θ.obs.σ) - 0.5log(2π)
    # Return (log-likelihood, updated state)
    return (ll, state)
end
```

### 3. Simulate and estimate likelihood

```julia
# One seed → fully reproducible (simulation, PF, MCMC)
rng = Random.Xoshiro(42)

model = MyModel()
θ     = MyParams{Float64}()
u0    = ComponentArray(P=-0.9, δ=0.5)

# Forward simulation (Euler-Maruyama internally)
result = simulate(rng, model, θ, u0, (0.0, 50.0), 0.01; save_dt=0.1)

# Particle filter marginal likelihood: p(y₁:ₜ | θ)
stepper = EMStepper(0.01)                    # internal SDE step size
cfg     = PFConfig(N=500, ess_frac=0.5)      # 500 particles, resample at 50% ESS
logL    = pf_loglik(rng, model, MyObs(), stepper, data, θ, u0, cfg;
                    obs_times=t_obs)
```

### 4. Run MCMC

```julia
# ParameterSpec controls which parameters are FREE (sampled) vs FIXED.
# Only parameters listed in `free` are varied during MCMC.
# Everything else stays at its default value — no code changes needed.
spec = ParameterSpec(
    # These will be sampled:
    free = [FieldPath(:sde, :γ), FieldPath(:sde, :κ), FieldPath(:obs, :σ)],

    # Set priors on the free parameters (your choice of distribution):
    priors = [
        FieldPath(:sde, :γ) => Uniform(40, 100),       # bounded uniform
        FieldPath(:sde, :κ) => Normal(0.5, 0.2),       # Gaussian prior
        FieldPath(:obs, :σ) => LogNormal(log(0.02), 0.5),  # positive-only
    ],

    # Optional: transform to unconstrained space for better sampling
    transforms = [
        FieldPath(:obs, :σ) => LogTransform(),   # sample in log-space
    ],
)

# Compile for fast pack/unpack (uses @generated functions — zero overhead)
cspec = compile(spec, θ)

# Run PMMH: the particle filter is called inside each MCMC iteration
chain = pmmh(rng, model, MyObs(), stepper, data, θ, u0, cfg, cspec;
             n_iter=5000, obs_times=t_obs)

# Access posterior samples by name
chain[:sde_γ]           # Vector{Float64} of posterior samples for γ
chain[:sde_κ]           # ... for κ
acceptance_rate(chain)   # should be 15–40% for good mixing
```

> **Tip:** Start with 2–3 free parameters to verify the pipeline, then gradually free more. Parameters not in the `free` list remain fixed at their `@kwdef` defaults — you can always add them later without changing your model code.

## Example: Volcano Core-Annular Flow

The `examples/volcano_CAF/` directory contains four interactive Pluto notebooks that build up from the physics to full inference:

| Notebook | What it covers |
|:---------|:--------------|
| `1_volcano_latent_sde.jl` | SDE dynamics, flux regime heatmap, parameter sensitivity |
| `2_observation_model.jl` | Lake height (Gaussian) and degassing (leaky integrator + Student-t) channels |
| `3_likelihood_calc.jl` | Particle filter likelihood, parameter profiling, ESS diagnostics |
| `4_inference.jl` | Full PMMH posterior sampling *(coming soon)* |

```bash
cd examples/volcano_CAF
julia -e 'using Pluto; Pluto.run(notebook="1_volcano_latent_sde.jl")'
```

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/your-username/LatentSDEBayes.jl")
```

**Requirements:** Julia ≥ 1.10, ComponentArrays, Distributions.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{LatentSDEBayes,
  author = {Ron Chuk},
  title  = {LatentSDEBayes.jl: Bayesian inference for latent SDEs via particle MCMC},
  year   = {2025},
  url    = {https://github.com/your-username/LatentSDEBayes.jl}
}
```
