using Game2048: bitboard_to_array, Dirs
using ProgressBars

struct MDP_mcts
    γ
    𝒮
    𝒜
    T
    R
    TR
end

struct MonteCarloTreeSearch
    𝒫 # problem
    N # visit counts
    Q # action value estimates
    d # depth
    m # number of simulations
    c # exploration constant
    U # value function estimate
end

function (π::MonteCarloTreeSearch)(s)
    for k in 1:π.m
        simulate!(π, s)
    end
    possibles = possible_moves(s)
    possible_actions = possibles[1]

    dir = argmax(
        Dict(a=>π.Q[(s,a)] for a in possible_actions)
    )
    return dir
end

bonus(Nsa, Ns) = Nsa == 0 ? Inf : sqrt(log(Ns)/Nsa)

function explore(π::MonteCarloTreeSearch, s)
    N, Q, c = π.N, π.Q, π.c
    (possible_actions, _) = possible_moves(s)

    Ns = sum(N[(s,a)] for a in possible_actions)
    Ns = (Ns == 0) ? Inf : Ns
    dir = argmax(
        Dict(a=>Q[(s,a)] + c*sqrt(log(Ns)/N[(s,a)]) for a in possible_actions)
    )
    return dir
end



function simulate!(π::MonteCarloTreeSearch, s, d=π.d)
    if d ≤ 0
        return π.U(s)
    end
    𝒫, N, Q, c = π.𝒫, π.N, π.Q, π.c
    TR, γ = 𝒫.TR, 𝒫.γ
    (𝒜, _) = possible_moves(s)
    if !haskey(N, (s, first(𝒜)))
        for a in 𝒜
            N[(s,a)] = 0
            Q[(s,a)] = 0.0
        end
        return π.U(s)
    end
    a = explore(π, s)
    s_prime = TR(s,a) #no reward
    q = γ*simulate!(π, s_prime, d-1)
    N[(s,a)] += 1
    Q[(s,a)] += (q-Q[(s,a)])/N[(s,a)]
    return q
end