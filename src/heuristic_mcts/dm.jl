module DM

# Textbook: Algorithms for Decision Making 
using Base.Threads
using DecisionMakingProblems
import DecisionMakingProblems.MDP
export DecisionMakingProblems, MonteCarloTreeSearch, MonteCarloTreeSearchTreeMT, MonteCarloTreeSearchMT

function lookahead(š«::MDP, U, s, a)
    š®, T, R, Ī³ = š«.š®, š«.T, š«.R, š«.Ī³
    return R(s,a) + Ī³*sum(T(s,a,sā²)*U(sā²) for sā² in š®)
end

function lookahead(š«::MDP, U::Vector, s, a)
    š®, T, R, Ī³ = š«.š®, š«.T, š«.R, š«.Ī³
    return R(s,a) + Ī³*sum(T(s,a,sā²)*U[i] for (i,sā²) in enumerate(š®))
end

struct ValueFunctionPolicy
    š« # problem
    U # utility function
end

function greedy(š«::MDP, U, s)
    u, a = findmax(a->lookahead(š«, U, s, a), š«.š)
    return (a=a, u=u)
end

#Chapter 9 

#forward search functions
struct RolloutLookahead
    š« # problem
    Ļ # rollout policy
    d # depth
end

randstep(š«::MDP, s, a) = š«.TR(s, a)

function rollout(š«, s, Ļ, d, isterminal::Function = s -> false)
    ret = 0.0
    for t in 1:d
        a = Ļ(s)
        s, r = randstep(š«, s, a)
        ret += š«.Ī³^(t-1) * r
        if isterminal(s)
            break
        end
    end
    return ret
end

function (Ļ::RolloutLookahead)(s)
    U(s) = rollout(Ļ.š«, s, Ļ.Ļ, Ļ.d)
    return greedy(Ļ.š«, U, s).a
end

struct MonteCarloTreeSearch
    š«::MDP # problem
    N::Dict # visit counts
    Q::Dict # action value estimates
    d::Integer # depth
    m::Integer # number of simulations
    c::AbstractFloat # exploration constant
    U::Function # value function estimate
end

struct MonteCarloTreeSearchTreeMT 
    š«::MDP # problem
    N::Vector{T} where T <: Dict  # visit counts
    Q::Vector{T}  where T <: Dict # action value estimates
    d::Integer # depth
    m::Integer # number of simulations
    c::AbstractFloat # exploration constant
    U::Function # value function estimate
    n::Integer # number of trees to have in parallel
end

function (Ļ::MonteCarloTreeSearch)(s)
    for k in 1:Ļ.m
        simulate!(Ļ, s)
    end
    return argmax(a->Ļ.Q[(s,a)], Ļ.š«.š)
end

function (Ļ::MonteCarloTreeSearchTreeMT)(s)
    @threads for i in 1:Ļ.n
        for k in 1:Ļ.m
            simulate!(Ļ, s, i)
        end
    end
    Q = reduce((d1, d2) -> combine_dicts(+, d1, d2), Ļ.Q)
    return argmax(a->Q[(s,a)], Ļ.š«.š)
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

function simulate!(Ļ::MonteCarloTreeSearch, s, d=Ļ.d)
    if d ā¤ 0
        return Ļ.U(s)
    end
    š«, N, Q, c = Ļ.š«, Ļ.N, Ļ.Q, Ļ.c
    š, TR, Ī³ = š«.š, š«.TR, š«.Ī³
    if !haskey(N, (s, first(š)))
        for a in š
            N[(s,a)] = 0
            Q[(s,a)] = 0.0
        end
        return Ļ.U(s)
    end
    a = explore(Ļ, s)
    sā², r = TR(s,a)
    q = r + Ī³*simulate!(Ļ, sā², d-1)
    N[(s,a)] += 1
    Q[(s,a)] += (q-Q[(s,a)])/N[(s,a)]
    return q
end

function simulate!(Ļ::MonteCarloTreeSearchTreeMT, s, i, d=Ļ.d)
    if d ā¤ 0
        return Ļ.U(s)
    end
    š«, N, Q, c = Ļ.š«, Ļ.N[i], Ļ.Q[i], Ļ.c
    š, TR, Ī³ = š«.š, š«.TR, š«.Ī³
    if !haskey(N, (s, first(š)))
        for a in š
            N[(s,a)] = 0
            Q[(s,a)] = 0.0
        end
        return Ļ.U(s)
    end
    a = explore(Ļ, s, i)
    sā², r = TR(s,a)
    q = r + Ī³*simulate!(Ļ, sā², i, d-1)
    N[(s,a)] += 1
    Q[(s,a)] += (q-Q[(s,a)])/N[(s,a)]
    return q
end

bonus(Nsa, Ns) = Nsa == 0 ? Inf : sqrt(log(Ns)/Nsa)

function explore(Ļ::MonteCarloTreeSearch, s)
    š, N, Q, c = Ļ.š«.š, Ļ.N, Ļ.Q, Ļ.c
    Ns = sum(N[(s,a)] for a in š)
    return argmax(a->Q[(s,a)] + c*bonus(N[(s,a)], Ns), š)
end

function explore(Ļ::MonteCarloTreeSearchTreeMT, s, i)
    š = Ļ.š«.š
    Q, N = Ļ.Q[i], Ļ.N[i]
    c = Ļ.c
    Ns = sum(N[(s,a)] for a in š)
    return argmax(a->Q[(s,a)] + c*bonus(N[(s,a)], Ns), š)
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