using Parameters
using Game2048: bitboard_to_array, Dirs
using CommonRLInterface; const rli = CommonRLInterface;
include("game.jl")

"""
MCTS integrated with neural network. Search mcts(s) returns probability distribution
with value evaluation. 
"""
@with_kw mutable struct MonteCarloTreeSearchNN
    env # problem
    nn # network

    N_sa::Dict{Tuple, Int} # visit counts for (s,a)
    Q::Dict{Tuple, Float64} # action values
    P # Policy distribution over actions from NN
    Outcomes
    Valids

    d::Int # depth
    m::Int # number of simulations
    c::Float64 # exploration constant
end

function (π::MonteCarloTreeSearchNN)(s, τ::Float64 = 1.0)::Vector{Float64}
    @unpack env, m, N_sa = π
    for _ in m
        search!(π, s, curr_step, max_step, π.d)
    end

    # Counts provide a good estimation of the improved policy (via UCB maximization)
    𝒜 = rli.actions(env)
    counts = [(haskey(N_sa, (s,a)) ? N_sa[(s,a)] : 0) for a in 𝒜]

    if τ == 0 # greedy action selection
        best_action_idx = rand(finall(x->x==maximum(counts), counts))
        probs = zeros(length(𝒜))
        probs[best_action_idx] .= 1
        return probs
    end

    counts = [count ^ (1.0/temp) for count in counts]
    denom = sum(counts)
    probs = counts ./ denom
    return probs
end

function search!(π::MonteCarloTreeSearchNN, s, curr_step, max_step, d)
    @unpack N_sa, Q, P, Outcomes, Valid = π
    @unpack d, m, c = π
    @unpack env, nn = π
    @unpack goal, γ = env

    if !haskey(Outcomes, s)
        Outcomes[s] = rli.reward(s, goal, curr_step, max_step)
    end
    if Outcomes[s] != 0.0 # Backup on terminal state
        return Outcomes[s]
    end
    if d ≤ 0 # Backup on horizon depth state
        return nothing #! insert neural network value prediction here
    end

    if !haskey(P, s) # Expansion on leaf node state
        p, v = nn #! insert neural network predictions here
        𝒜_size = length(rli.actions(env))
        valid_mask = valid_action_mask(s, 𝒜_size)
        p .*= valid_mask
        sum_p = sum(p)

        if sum_p > 0 # renormalize for distribution over valid actions
            p ./= sum_p
        else # if no valid actions, uniform distribution
            p .= 1/𝒜_size 
        end

        P[s] = p
        for a in 𝒜_size
            N_sa[(s,a)] = 0
            Q[(s,a)] = 0.0
        end

        return v
    end

    best_action = selection(s, rli.actions(env), N_sa, Q, P, c)
    s′, r, terminated = transition(s, best_action, goal, curr_step, max_step)
    if terminated
        return r
    end

    v = search!(π, s′, curr_step+1, max_step, d-1)

    Q[(s,best_action)] = (N_sa[(s,best_action)] * Q[(s, best_action)] + v) / (N_sa[(s,best_action)] + 1)
    N_sa[(s,best_action)] += 1

    return v
end

function selection(s, 𝒜, N_sa, Q, P, c::Float64)::Int # UCT 
    N_s = sum(N[(s,a)] for a in 𝒜)
    N_s = (N_s == 0) ? Inf : N_s

    𝒜_valid = valid_actions(s)
    uct(s,a) = Q[(s,a)] + c * P[s][a] * sqrt(N_s) / (1 + N_sa[(s,a)])
    best_action = argmax(Dict(
        a => uct(s,a) for a in 𝒜_valid
    ))
    return best_action
end