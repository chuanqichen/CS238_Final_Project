module DM

# Textbook: Algorithms for Decision Making 
using Base.Threads
using DecisionMakingProblems
import DecisionMakingProblems.MDP
export DecisionMakingProblems, MonteCarloTreeSearch, MonteCarloTreeSearchTreeMT, MonteCarloTreeSearchMT

function lookahead(𝒫::MDP, U, s, a)
    𝒮, T, R, γ = 𝒫.𝒮, 𝒫.T, 𝒫.R, 𝒫.γ
    return R(s,a) + γ*sum(T(s,a,s′)*U(s′) for s′ in 𝒮)
end

function lookahead(𝒫::MDP, U::Vector, s, a)
    𝒮, T, R, γ = 𝒫.𝒮, 𝒫.T, 𝒫.R, 𝒫.γ
    return R(s,a) + γ*sum(T(s,a,s′)*U[i] for (i,s′) in enumerate(𝒮))
end

struct ValueFunctionPolicy
    𝒫 # problem
    U # utility function
end

function greedy(𝒫::MDP, U, s)
    u, a = findmax(a->lookahead(𝒫, U, s, a), 𝒫.𝒜)
    return (a=a, u=u)
end

#Chapter 9 

#forward search functions
struct RolloutLookahead
    𝒫 # problem
    π # rollout policy
    d # depth
end

randstep(𝒫::MDP, s, a) = 𝒫.TR(s, a)

function rollout(𝒫, s, π, d, isterminal::Function = s -> false)
    ret = 0.0
    for t in 1:d
        a = π(s)
        s, r = randstep(𝒫, s, a)
        ret += 𝒫.γ^(t-1) * r
        if isterminal(s)
            break
        end
    end
    return ret
end

function (π::RolloutLookahead)(s)
    U(s) = rollout(π.𝒫, s, π.π, π.d)
    return greedy(π.𝒫, U, s).a
end

struct MonteCarloTreeSearch
    𝒫::MDP # problem
    N::Dict # visit counts
    Q::Dict # action value estimates
    d::Integer # depth
    m::Integer # number of simulations
    c::AbstractFloat # exploration constant
    U::Function # value function estimate
end

struct MonteCarloTreeSearchTreeMT 
    𝒫::MDP # problem
    N::Vector{T} where T <: Dict  # visit counts
    Q::Vector{T}  where T <: Dict # action value estimates
    d::Integer # depth
    m::Integer # number of simulations
    c::AbstractFloat # exploration constant
    U::Function # value function estimate
    n::Integer # number of trees to have in parallel
end

function (π::MonteCarloTreeSearch)(s)
    for k in 1:π.m
        simulate!(π, s)
    end
    return argmax(a->π.Q[(s,a)], π.𝒫.𝒜)
end

function (π::MonteCarloTreeSearchTreeMT)(s)
    @threads for i in 1:π.n
        for k in 1:π.m
            simulate!(π, s, i)
        end
    end
    Q = reduce((d1, d2) -> combine_dicts(+, d1, d2), π.Q)
    return argmax(a->Q[(s,a)], π.𝒫.𝒜)
end

function combine_dicts(op::Function, d1::T, d2::T) where T <: Dict
   dout = T() 
   klist = union(keys(d1), keys(d2))
   for k in klist 
        if haskey(d1, k) && haskey(d2, k)
            dout[k] = op(d1[k], d2[k])
        elseif haskey(d1, k)
            dout[k] = d1[k]
        else # elseif(haskey(d2, k))
            dout[k] = d2[k]
        end
   end
   return dout
end

function simulate!(π::MonteCarloTreeSearch, s, d=π.d)
    if d ≤ 0
        return π.U(s)
    end
    𝒫, N, Q, c = π.𝒫, π.N, π.Q, π.c
    𝒜, TR, γ = 𝒫.𝒜, 𝒫.TR, 𝒫.γ
    if !haskey(N, (s, first(𝒜)))
        for a in 𝒜
            N[(s,a)] = 0
            Q[(s,a)] = 0.0
        end
        return π.U(s)
    end
    a = explore(π, s)
    s′, r = TR(s,a)
    q = r + γ*simulate!(π, s′, d-1)
    N[(s,a)] += 1
    Q[(s,a)] += (q-Q[(s,a)])/N[(s,a)]
    return q
end

function simulate!(π::MonteCarloTreeSearchTreeMT, s, i, d=π.d)
    if d ≤ 0
        return π.U(s)
    end
    𝒫, N, Q, c = π.𝒫, π.N[i], π.Q[i], π.c
    𝒜, TR, γ = 𝒫.𝒜, 𝒫.TR, 𝒫.γ
    if !haskey(N, (s, first(𝒜)))
        for a in 𝒜
            N[(s,a)] = 0
            Q[(s,a)] = 0.0
        end
        return π.U(s)
    end
    a = explore(π, s, i)
    s′, r = TR(s,a)
    q = r + γ*simulate!(π, s′, i, d-1)
    N[(s,a)] += 1
    Q[(s,a)] += (q-Q[(s,a)])/N[(s,a)]
    return q
end

bonus(Nsa, Ns) = Nsa == 0 ? Inf : sqrt(log(Ns)/Nsa)

function explore(π::MonteCarloTreeSearch, s)
    𝒜, N, Q, c = π.𝒫.𝒜, π.N, π.Q, π.c
    Ns = sum(N[(s,a)] for a in 𝒜)
    return argmax(a->Q[(s,a)] + c*bonus(N[(s,a)], Ns), 𝒜)
end

function explore(π::MonteCarloTreeSearchTreeMT, s, i)
    𝒜 = π.𝒫.𝒜
    Q, N = π.Q[i], π.N[i]
    c = π.c
    Ns = sum(N[(s,a)] for a in 𝒜)
    return argmax(a->Q[(s,a)] + c*bonus(N[(s,a)], Ns), 𝒜)
end

function Base.findmax(f::Function, xs)
    f_max = -Inf
    x_max = first(xs)
    for x in xs
        v = f(x)
        if v > f_max
            f_max, x_max = v, x
        end
    end
    return f_max, x_max
end

Base.argmax(f::Function, xs) = findmax(f, xs)[2]

end # module