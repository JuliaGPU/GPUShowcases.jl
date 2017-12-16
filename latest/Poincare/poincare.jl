using CLArrays, GPUArrays
using FileIO, Interpolations, Colors, ColorVectorSpace, FixedPointNumbers

function poincare_inner{N}(rv, result, c, π, ::Val{N}, n)
    # find next spiking neuron
    ϕ₁, ϕ₂, ϕ₃ = rv[1], rv[2], rv[3]
    πh = π / 2f0
    π2 = π * 2f0
    for unused = 1:N
        if ϕ₁ > ϕ₂
            if ϕ₁ > ϕ₃
                # first neuron is spiking
                dt = πh - ϕ₁
                # evolve phases till next spike time
                ϕ₁ = -πh
                ϕ₂ = atan(tan(ϕ₂ + dt) - c)
                ϕ₃ += dt
                # save state of neuron 2 and 3
                x = Cuint(max(round(((ϕ₂ + πh) / π) * (Float32(n) - 1f0)) + 1f0, 1f0))
                y = Cuint(max(round(((ϕ₃ + πh) / π) * (Float32(n) - 1f0)) + 1f0, 1f0))
                i1d = GPUArrays.gpu_sub2ind((n, n), (x, y))
                @inbounds if i1d <= Cuint(n * n) && i1d > Cuint(0)
                    accum = result[i1d]
                    # this is unsafe, since it could read + write from different threads, but good enough for the stochastic kind of process we're doing
                    result[i1d] = accum + 1f0
                end
                continue
            end
        else
            if ϕ₂ > ϕ₃
                # second neuron is spiking
                dt = πh - ϕ₂
                # evolve phases till next spike time
                ϕ₁ += dt
                ϕ₂ = -πh
                ϕ₃ = atan(tan(ϕ₃ + dt) - c)
                continue
            end
        end
        # third neuron is spikinga
        dt = πh - ϕ₃
        # evolve phases till next spike time
        ϕ₁ += dt
        ϕ₂ = atan(tan(ϕ₂ + dt) - c)
        ϕ₃ = -πh
    end
    return
end

function poincare_inner(n, seeds::GPUArray, result, c, π, val::Val{N}) where N
    foreach(poincare_inner, seeds, result, c, Float32(pi), val, n)
end

c = 1f0; divisor = 2^10
srand(2)
N = 10^10
ND = Cuint(2048)
AT = CLArray
result = AT(zeros(Float32, ND, ND))
_n = div(N, divisor)
jl_seeds = [ntuple(i-> rand(Float32), Val{3}) for x in 1:divisor]
seeds = AT(jl_seeds)
poincare_inner(ND, seeds, Base.RefValue(result), c, Float32(pi), Val{_n}())

cmap = interpolate(([
    RGB(0.0, 0.0, 0),
    RGB(0.2, 0.2, 0.9),
    RGB(0.2, 0.6, 0.9),
    RGB(0.7, 0.7, 0.98),
    RGB(0.8, 0.8, 0.9),
    RGB(0.82, 0.8, 1.0)
]), BSpline(Linear()), OnCell())

cn = length(cmap)
resultcpu = Array(result)

img_color = map(resultcpu) do val
    val = log(1.0 + log(1.0 + val))
    val = clamp(1.0 - val, 0f0, 1f0);
    idx = (val * (cn - 1)) + 1.0
    RGB{N0f8}(cmap[idx])
end
#save as an image
save(joinpath(@__DIR__, "poincare.png"), img_color)
