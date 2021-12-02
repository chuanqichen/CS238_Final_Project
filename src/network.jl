using Flux
include("utils.jl")

# Custom function for network with multiple heads (policy and value function)
struct Split{T}
    paths::T
  end

Split(paths...) = Split(paths)

Flux.@functor Split

(m::Split)(x::AbstractArray) = tuple(map(f -> f(x), m.paths))

# Network
net = Chain(
    Dense(256, 300, relu),
    Dense(300, 150, relu),
    Split(
        Chain(Dense(150, 4), logsoftmax),  # policy head
        Dense(150, 1, σ) # Value head
    )
) |> device

# Loss function
loss_π(p̂, p) = -sum(p .* p̂)
loss_v(v̂, r) = Flux.mse(v̂, r)
function loss(sample::NamedTuple)
    s, p, r = sample
    p̂, v̂ = only(net(s))
    return loss_π(p̂, p) + loss_v(v̂, r)
end

function loss(s, p, r)
    p̂, v̂ = only(net(s))
    return loss_π(p̂, p) + loss_v(v̂, r)
end