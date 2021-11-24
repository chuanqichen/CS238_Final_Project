using Parameters
using Game2048: bitboard_to_array, Dirs
include("board.jl")

@with_kw mutable struct MonteCarloTreeSearch
    env # problem
    N # visit counts
    Q # action value estimates
    d # depth
    m # number of simulations
    c # exploration constant
    # π::Function # tree policy. Options: UCB, UCB+NN, 
    U::Function # value estimate. Options: random rollout, greedy rollout, or critic
end

function (π::MonteCarloTreeSearch)(s)
    for k in 1:π.m
        simulate!(π, s)
    end
    possible_actions = valid_actions(s)

    dir = argmax(
        Dict(a=>π.Q[(s,a)] for a in possible_actions)
    )
    return Integer(dir)
end

bonus(Nsa, Ns) = Nsa == 0 ? Inf : sqrt(log(Ns)/Nsa)

function explore(π::MonteCarloTreeSearch, s)
    env, N, Q, c = π.env,  π.N, π.Q, π.c
    possible_actions = valid_actions(s)

    Ns = sum(N[(s,a)] for a in possible_actions)
    Ns = (Ns == 0) ? Inf : Ns
    dir = argmax(
        Dict(a=>Q[(s,a)] + c*sqrt(log(Ns)/N[(s,a)]) for a in possible_actions)
    )
    return Integer(dir)
end



function simulate!(π::MonteCarloTreeSearch, s, d=π.d)
    if d ≤ 0
        return π.U(s)
    end
    env, N, Q, c = π.env, π.N, π.Q, π.c
    γ = env.γ
    𝒜 = valid_actions(s)

    if !haskey(N, (s, first(𝒜)))
        for a in 𝒜
            N[(s,a)] = 0
            Q[(s,a)] = 0.0
        end
        return π.U(s)
    end
    a = explore(π, s)
    r = act!(env, a) #no reward
    s′ = env.board
    q = γ * simulate!(π, s′, d-1)
    N[(s,a)] += 1
    Q[(s,a)] += (q-Q[(s,a)])/N[(s,a)]
    return q
end