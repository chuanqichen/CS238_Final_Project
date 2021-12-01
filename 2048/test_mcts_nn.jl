using Parameters
using CommonRLInterface; const rli = CommonRLInterface;

using Game2048: move, initbboard, Dirs
using AlphaZero
include("game.jl")
include("mcts_nn.jl")
include("alphazero.jl")
include("network.jl")

# Environment Spec
goal = 2048
γ = 1.0
env = Env2048(
    goal = goal,
    γ = γ,
    state_repr = Vector
    )

# Network Spec

d = 3
m = 3
c = 0.9



mcts_nn = MonteCarloTreeSearchNN(
    env = env, 
    net = net,
    d = d, 
    m = m, 
    c = c, 
)

trainer = AlphaZeroTrainer(
    env = env,
    mcts_nn = mcts_nn,
    net = net,
)

learn!(trainer)