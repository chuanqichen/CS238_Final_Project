using Flux
# Custom function for multiple heads
struct Split{T}
    paths::T
  end
  
Split(paths...) = Split(paths)

Flux.@functor Split

(m::Split)(x::AbstractArray) = tuple(map(f -> f(x), m.paths))


net = Chain(
    Dense(256, 300, relu),
    Dense(300, 150, relu),
    Split(
        Chain(Dense(150, 4), logsoftmax),  # policy head
        Dense(150, 1, σ) # Value head
    )
)

loss_π(p̂, p) = -sum(p .* p̂)
loss_v(v̂, r) = Flux.mse(v̂, r)
function loss(state, sample)
    p̂, v̂ = net(state)
    s, p, r = sample
    return loss_π(p̂, p) + loss_v(v̂, r)
end



