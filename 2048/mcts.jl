using Parameters
using Game2048: bitboard_to_array, Dirs
include("game.jl")

"""
Vanilla MCTS without any neural network. Search mcts(s) returns best action
via Q-values. 
"""
@with_kw mutable struct MonteCarloTreeSearch
    env # problem
    N # visit counts
    Q # action value estimates
    d # depth
    m # number of simulations
    c # exploration constant
    # π::Function # tree policy. Options: UCB exploration, policy
    U::Function # value estimate. Options: random rollout, greedy rollout, or critic
end

function (π::MonteCarloTreeSearch)(s)
    @unpack env = π

    for _ in 1:π.m
        env_copy = rli.clone(env)
        π.env = env_copy
        search!(π, s, env_copy.curr_step, env_copy.max_step)
    end
    π.env = env
    possible_actions = valid_actions(s)
    
    best_action = argmax(
        Dict(a=>π.Q[(s,a)] for a in possible_actions)
    )
    return best_action
end

function search!(π::MonteCarloTreeSearch, s, curr_step, max_step, d=π.d)
    @unpack env, N, Q, c = π
    @unpack goal, γ = env
    if d ≤ 0
        return π.U(s, env.goal, curr_step, max_step)
    end

    𝒜 = rli.actions(env)
    if !haskey(N, (s, first(𝒜)))
        for a in 𝒜
            N[(s,a)] = 0
            Q[(s,a)] = 0.0
        end
        return π.U(s, env.goal, curr_step, max_step)
    end

    a = explore(π, s)
    s′, r, terminated = transition(s, a, goal, curr_step, max_step) #no reward
    if terminated
        return r
    end
    q = γ * search!(π, s′, curr_step+1, max_step, d-1)
    N[(s,a)] += 1
    Q[(s,a)] += (q-Q[(s,a)])/N[(s,a)]
    return q
end

bonus(Nsa, Ns) = Nsa == 0 ? Inf : sqrt(log(Ns)/Nsa)

function explore(π::MonteCarloTreeSearch, s)
    env, N, Q, c = π.env,  π.N, π.Q, π.c
    𝒜 = rli.actions(env)

    Ns = sum(N[(s,a)] for a in 𝒜)
    Ns = (Ns == 0) ? Inf : Ns
    dir = argmax(
        Dict(a=>Q[(s,a)] + c*sqrt(log(Ns)/N[(s,a)]) for a in 𝒜)
    )
    return Integer(dir)
end

