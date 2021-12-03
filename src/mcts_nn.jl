using Parameters
using Flux; import Flux
using Game2048: bitboard_to_array, Dirs
using CommonRLInterface; const rli = CommonRLInterface;
using AlphaZero; 
include("game.jl")

"""
MCTS integrated with neural network. Search mcts(s) returns probability distribution
with value evaluation. 
"""
@with_kw mutable struct MonteCarloTreeSearchNN
    env # problem
    net # network

    N_sa::Dict{Tuple, Int} = Dict() # visit counts for (s,a)
    Q::Dict{Tuple, Float64} = Dict() # action values
    P = Dict() # Policy distribution over actions from net
    Outcomes = Dict()

    d::Int # depth
    m::Int # number of simulations
    c::Float64 # exploration constant
end

function (π::MonteCarloTreeSearchNN)(s; τ::Float64 = 1.0)::Vector{Float64}
    @unpack env, m = π
    for _ in 1:m
        search!(π, s, env.curr_step, env.max_step, π.d)
    end

    # Counts provide a good estimation of the improved policy (via UCB maximization)
    𝒜 = rli.actions(env)
    counts = [(haskey(π.N_sa, (s,a)) ? π.N_sa[(s,a)] : 0) for a in 𝒜]

    if τ == 0 # greedy action selection
        best_action_idx::Int = rand(findall(x->x==maximum(counts), counts))
        probs = zeros(length(𝒜))
        probs[best_action_idx] = 1.0
        return probs
    end

    counts = [count ^ (1.0/τ) for count in counts]
    denom = sum(counts)
    # probs = (denom==0.0) ? 0.25 * ones(4) : counts ./ denom
    probs = counts ./ denom
    return probs
end

function search!(π::MonteCarloTreeSearchNN, s, curr_step::Int, max_step::Int, d::Int)::AbstractFloat
    @unpack env, net = π
    @unpack T, R, terminated, goal, γ = env

    net = net |> cpu

    if d ≤ 0 # Backup on horizon depth state
        p, v = only(net(unsqueeze_batch_maybe(s)))
        return only(v)
    end
    if !haskey(π.Outcomes, s)
        π.Outcomes[s] = R(s, goal, curr_step, max_step)
    end
    if terminated(s, goal, curr_step, max_step) # Backup on terminal state
        return π.Outcomes[s]
    end


    if !haskey(π.P, s) # Expansion on leaf node state
        p, v = only(net(unsqueeze_batch_maybe(s)))
        𝒜 = rli.actions(env)
        valid_mask = valid_action_mask(s, length(𝒜))
        p .*= valid_mask
        sum_p = sum(p)

        if sum_p > 0 # renormalize for distribution over valid actions
            p ./= sum_p
        else # if no valid actions, uniform distribution
            p .= 1/length(𝒜)
        end

        π.P[s] = p
        for a in 𝒜
            π.N_sa[(s,a)] = 0
            π.Q[(s,a)] = 0.0
        end

        return only(v)
    end

    best_action = selection(s, rli.actions(env), π.N_sa, π.Q, π.P, π.c)
    s′, _, r, done = T(s, best_action, goal, curr_step, max_step)

    v = search!(π, s′, curr_step+1, max_step, d-1)

    π.Q[(s,best_action)] = (π.N_sa[(s,best_action)] * π.Q[(s, best_action)] + v) / (π.N_sa[(s,best_action)] + 1)
    π.N_sa[(s,best_action)] += 1

    return v
end

function selection(s, 𝒜, N_sa, Q, P, c::Float64)::Int # UCT 
    N_s = sum(N_sa[(s,a)] for a in 𝒜)
    N_s = (N_s == 0) ? Inf : N_s

    𝒜_valid = valid_actions(s)
    uct(s,a) = Q[(s,a)] + c * P[s][a+1] * sqrt(N_s) / (1 + N_sa[(s,a)])
    best_action = argmax(Dict(
        a => uct(s,a) for a in 𝒜_valid
    ))
    return best_action
end