# LatentSDEBayes.jl

**Bayesian inference for latent stochastic differential equations via particle MCMC.**

## Why This Package?

In many physical systems — volcanic conduits, subsurface reservoirs, ocean circulation, neural dynamics — the governing processes happen in a **latent space** we cannot observe directly. We measure proxy signals at the surface (tilt, gas flux, seismicity) that are noisy, nonlinear transformations of the hidden state. The scientific models describe the physics of the latent dynamics, but the data live in observation space.

The standard approach is a **state-space model**:

```
Latent:       dx = f(x, θ) dt + σ(x, θ) dW        ← your physics (SDE)
Observation:  yₖ ~ p(y | x(tₖ), θ)                 ← your measurement model
Question:     p(θ | y₁, ..., yₜ) = ?                ← what we want
```

**LatentSDEBayes.jl** lets you define both layers in Julia, then handles the hard part: estimating the marginal likelihood $p(y_{1:T} \mid \theta)$ via a bootstrap particle filter and sampling the posterior $p(\theta \mid y_{1:T})$ via Particle Marginal Metropolis–Hastings (PMMH). You write the physics; the package runs the inference.

**What makes it different:** named state access everywhere (`u.P`, `u.delta`, `theta.sde.gamma` — never `u[1]`, `p[3]`), zero-allocation particle filter inner loop, fully typed workspaces with no `Vector{Any}`, and deterministic RNG so every run is reproducible from a single seed.

## Architecture

```
┌──────────────────────── You define ────────────────────────┐
│                                                            │
│   drift!(du, model, u, θ, t)     diffusion!(g, model, u, θ, t)   │
│              ↓                            ↓                │
│         ┌─────────┐                 ┌──────────┐           │
│         │ Latent  │── propagate! ──→│ Particle │           │
│         │  SDE    │  (EMStepper)    │  Cloud   │           │
│         └─────────┘                 └────┬─────┘           │
│                                          │ weight          │
│   loglik_step(obs, yₖ, uₖ, θ, state) ───┘                │
│              ↓                                             │
│         ┌──────────┐    ┌────────┐    ┌─────────┐          │
│         │ pf_loglik│───→│  PMMH  │───→│ Posterior│          │
│         │ (filter) │    │ (mcmc) │    │ Samples  │          │
│         └──────────┘    └────────┘    └─────────┘          │
│                                                            │
└────────────────────── Package handles ─────────────────────┘
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
using LatentSDEBayes, ComponentArrays, Random
```

### 1. Define your SDE

```julia
struct MyModel <: AbstractLatentModel end

LatentSDEBayes.state_template(::MyModel) = ComponentArray(P=0.0, δ=0.0)
LatentSDEBayes.noise_type(::MyModel) = DiagonalNoise()

function LatentSDEBayes.drift!(du, ::MyModel, u, θ, t)
    du.P = -2θ.sde.γ * (u.P + 1) * (u.P + 1/θ.sde.β) * (2u.P + 1 + 1/θ.sde.β) -
            θ.sde.κ * (u.δ - 0.5) + θ.sde.I
    Pc = -0.5(1 + 1/θ.sde.β)
    du.δ = θ.sde.ε * u.δ * (1 - u.δ) * (1/(1 + exp(-θ.sde.m*(u.P - Pc))) - u.δ)
    return nothing
end

function LatentSDEBayes.diffusion!(g, ::MyModel, u, θ, t)
    g.P = θ.sde.σP
    g.δ = θ.sde.σδ * u.δ * (1 - u.δ)
    return nothing
end
```

### 2. Define your observation model

```julia
struct MyObs <: AbstractObsModel end
struct MyObsState{T}; s_prev::T; end   # stateful channel memory

LatentSDEBayes.init_obs_state(::MyObs, θ) = MyObsState(0.0)

function LatentSDEBayes.loglik_step(::MyObs, yₖ, uₖ, θ, state)
    predicted = θ.obs.base + θ.obs.scale * uₖ.P
    z = (yₖ - predicted) / θ.obs.σ
    ll = -0.5z^2 - log(θ.obs.σ) - 0.5log(2π)
    return (ll, state)
end
```

### 3. Simulate and estimate likelihood

```julia
# Reproducible from one seed
rng = Random.Xoshiro(42)

model = MyModel()
θ     = MyParams()                                     # your @kwdef struct
u0    = ComponentArray(P=-0.9, δ=0.5)

# Forward simulation
result = simulate(rng, model, θ, u0, (0.0, 50.0), 0.01; save_dt=0.1)

# Particle filter likelihood
stepper = EMStepper(0.01)
cfg     = PFConfig(N=500, ess_frac=0.5)
logL    = pf_loglik(rng, model, MyObs(), stepper, data, θ, u0, cfg;
                    obs_times=t_obs)
```

### 4. Run MCMC

```julia
spec = ParameterSpec(
    free   = [FieldPath(:sde, :γ), FieldPath(:sde, :κ)],
    priors = [FieldPath(:sde, :γ) => Uniform(40, 100),
              FieldPath(:sde, :κ) => Normal(0.5, 0.2)],
)
cspec = compile(spec, θ)

chain = pmmh(rng, model, MyObs(), stepper, data, θ, u0, cfg, cspec;
             n_iter=5000, obs_times=t_obs)

chain[:sde_γ]           # posterior samples for γ
acceptance_rate(chain)   # should be 15–40%
```

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
