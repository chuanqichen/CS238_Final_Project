using DrWatson
@quickactivate
using Parameters
using CommonRLInterface; const rli = CommonRLInterface;

using Game2048: move, initbboard, Dirs
using AlphaZero
include("../src/game.jl")
include("../src/mcts_nn.jl")
include("../src/alphazero.jl")
include("../src/network.jl")

# Environment Spec
goal = 2048
γ = 1.0

#* DEBUGGING
# MCTS Spec
d = 5
m = 1
c = 0.9

# AlphaZero Spec
num_iters = 3
num_episodes = 3
num_samples_iter = 1e6
num_samples_iter_history = 20


#* TRAINING
# # MCTS Spec
# d = 200
# m = 5
# c = 0.9

# # AlphaZero Spec
# num_iters = 200
# num_episodes = 1000
# num_samples_iter = 1e6
# num_samples_iter_history = 20

#* INSTANTIATION

env = Env2048(
    goal = goal,
    γ = γ,
    state_repr = Vector
    )

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