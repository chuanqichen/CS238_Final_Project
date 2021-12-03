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
unsqueeze_batch_maybe(array::Array) = (ndims(array) == 1) ? Flux.unsqueeze(array, 2) : array 

net = Chain(
    Dense(256, 300), BatchNorm(300, relu), Dropout(0.1),
    Dense(300, 150), BatchNorm(150, relu), Dropout(0.1), 
    Split(
        Chain(Dense(150, 4), softmax),  # policy head
        Dense(150, 1, σ) # Value head
    )
) |> device

# Loss function
loss_π(p̂, p) = Flux.crossentropy(p̂, p)
loss_v(v̂, r) = Flux.mse(v̂, r)
function loss(s, p, r)
    p̂, v̂ = only(net(s))
    return loss_π(p̂, p) + loss_v(v̂, r)
end