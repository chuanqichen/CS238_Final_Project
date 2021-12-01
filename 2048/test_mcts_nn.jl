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

# MCTS Spec
d = 200
m = 10
c = 0.9

# AlphaZero Spec
num_iters = 1000
num_episodes = 5000
num_samples_iter = 1e6
num_samples_iter_history = 20

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
    num_iters = num_iters,
    num_episodes = num_episodes,
    num_samples_iter = num_samples_iter,
    num_samples_iter_history = num_samples_iter_history
)

learn!(trainer)