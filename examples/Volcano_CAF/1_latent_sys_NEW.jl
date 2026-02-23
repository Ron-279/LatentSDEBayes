### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 31849016-108c-11f1-0a89-7d2c419c003c
begin
    import Pkg
    Pkg.activate(@__DIR__)
    Pkg.develop(path=joinpath(@__DIR__, "..", ".."))
    Pkg.precompile()
end

# ╔═╡ 34db4daa-65ec-49d2-9d82-f9f0c24bd050
begin
    using LatentSDEBayes
    using ComponentArrays, StochasticDiffEq, Plots, Colors, Measures, PlutoUI, Markdown
    
    macro bind(def, element)
        quote
            local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
            local el = $(esc(element))
            global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
            el
        end
    end
end

# ╔═╡ e6fc84fb-f85a-4d42-8d5f-280379bfdb67
begin
    include("VolcanoModels.jl")
    # This 'dot' is essential to load the local module we just defined in the file
    using .VolcanoModels
end

# ╔═╡ 44c54bee-3317-4b49-ae0b-cefd39c08def
md" # Imports & Setup"

# ╔═╡ c2df7836-1193-4b4b-a739-8ce145a4136d
TableOfContents()

# ╔═╡ bf6c27f9-f310-4433-9af4-dc72a5d4a788
md" #  SDE"

# ╔═╡ 755f4102-df9f-4229-ad13-0e6758eed194
md""" ```math
\begin{align}
\text{Slow Dynamics} : \quad \dot{P} &= -2\gamma (P+1)\left(P+\frac{1}{\beta}\right)\left(2P+1+\frac{1}{\beta}\right) - \kappa(\delta-0.5) + I \\[10pt]
\text{Fast Dynamics} : \quad \dot{\delta} &= \varepsilon\,\delta(1-\delta)\Big[\sigma\big(m(P-P_c)\big)-\delta\Big] \nonumber\\[8pt]
\text{with} \quad \sigma(x) &= \frac{1}{1+e^{-x}} \quad \text{and} \quad P_c = -\frac{1}{2}\left(1+\frac{1}{\beta}\right) \nonumber
\end{align}

```
"""

# ╔═╡ d66c6003-36dc-4c8f-ac82-a210b2ad8677
md"""
| Symbol | Description |
|:------:|:------------|
| $P(t)$ | Slow pressure gradient vairable|
| $\delta(t)$ | Fast core-radius fraction variable |
| $\gamma$ | Relaxation strength |
| $\kappa$ | Linear coupling gain|
| $I$ | External forcing |
| $\varepsilon$ | Time-scale separation parameter |
| $m$ | Sigmoid steepness |
| $P_c$ | Sigmoid midpoint in $P$ |
| $\sigma(\cdot)$ | Logistic switch mapping $\mathbb{R}\to(0,1)$ |
"""


# ╔═╡ d767b864-5d80-4ff8-8e32-095c79d7d634
md" ## Parameters"

# ╔═╡ 7ece7f90-beb6-427f-96d5-3e97025584ab
begin
β = 1.2
M = 10.0
println("Fixed Material Parameters: β = ", β, ", M = ", M)
end

# ╔═╡ ba39c664-b1cb-4825-a78a-0371e58e5b67
md"""
**γ:** $(@bind γ Slider(0.0:1.0:100.0, default=69.0, show_value=true))  
**κ:** $(@bind κ Slider(0.0:0.005:2.0, default=0.57, show_value=true))  
**I:** $(@bind I Slider(-0.5:0.005:0.5, default=0.025, show_value=true))  
**m:** $(@bind m Slider(0.0:0.5:70.0, default=34.5, show_value=true))  
**ε = 1/τ:** $(@bind ε Slider(0.1:0.1:15.0, default=4.3, show_value=true))  
**Slow noise σP:** $(@bind σP Slider(0.0:1e-5:5e-3, default=27e-4, show_value=true))  
**Fast noise σδ:** $(@bind σδ Slider(0.0:1e-4:2e-1, default=5e-3, show_value=true))  
"""

# ╔═╡ 5e2faacc-0749-45af-837f-edd29c3fcdbe
# ╠═╡ disabled = true
#=╠═╡
begin
    # Freeze the Pluto slider globals into our strict package parameters
    p_sde = VolcanoSDEParams(
        beta = β, 
        gamma = float(γ),
        kappa = float(κ),
        I_ext = float(I),
        epsilon = float(ε),
        m = float(m),
        sigma_P = float(σP),
        sigma_delta = float(σδ),
        M = M # Fixed viscosity
    )
end
  ╠═╡ =#

# ╔═╡ 1e0349ee-5fa2-44d3-895d-e27370ff5ea3
#=╠═╡
begin
    function simulate_new_SDE(P0, δ0, p_sde; tspan=(0.0, 200.0), dt=0.01, saveat=0.1)
        # 1. Use the ComponentArray state
        u0 = ComponentArray(P = P0, delta = δ0)
        
        # 2. Bundle parameters
        p_master = (sde = p_sde, obs = nothing)
        
        # 3. Wrap our LatentSDEBayes interface for StochasticDiffEq
        f!(du, u, p, t) = drift!(du, VolcanoLatent(), u, p, t)
        g!(du, u, p, t) = diffusion!(du, VolcanoLatent(), u, p, t)
        
        # 4. Solve!
        prob = SDEProblem(f!, g!, u0, tspan, p_master)
        sol = solve(prob, EM(); dt=dt, saveat=saveat)
        
        return sol
    end
    
    # Run the simulation
    sol = simulate_new_SDE(-0.9, 0.5, p_sde)
    
    # Extract arrays for plotting
    P_sim = [u.P for u in sol.u]
    δ_sim = [u.delta for u in sol.u]
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╟─44c54bee-3317-4b49-ae0b-cefd39c08def
# ╠═31849016-108c-11f1-0a89-7d2c419c003c
# ╠═34db4daa-65ec-49d2-9d82-f9f0c24bd050
# ╠═e6fc84fb-f85a-4d42-8d5f-280379bfdb67
# ╠═c2df7836-1193-4b4b-a739-8ce145a4136d
# ╟─bf6c27f9-f310-4433-9af4-dc72a5d4a788
# ╟─755f4102-df9f-4229-ad13-0e6758eed194
# ╟─d66c6003-36dc-4c8f-ac82-a210b2ad8677
# ╟─d767b864-5d80-4ff8-8e32-095c79d7d634
# ╠═7ece7f90-beb6-427f-96d5-3e97025584ab
# ╟─ba39c664-b1cb-4825-a78a-0371e58e5b67
# ╠═5e2faacc-0749-45af-837f-edd29c3fcdbe
# ╠═1e0349ee-5fa2-44d3-895d-e27370ff5ea3
