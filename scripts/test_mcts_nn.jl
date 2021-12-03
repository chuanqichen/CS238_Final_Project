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
max_step = Inf

# * DEBUGGING
# d = 10
# m = 2
# c = 0.9


# num_iters = 3
# num_episodes = 1
# num_epochs = 1
# num_samples_iter = 2e4
# num_samples_iter_history = 2
# num_evals = 1

#* TRAINING
# MCTS Spec
d = Inf
m = 5
c = 3

# AlphaZero Spec
num_epochs = 8

num_iters = 3000
num_episodes = 20
num_samples_iter = 5e4
num_samples_iter_history = 1
num_evals = 9


#* INSTANTIATION

env = Env2048(
    goal = goal,
    γ = γ,
    state_repr = Vector,
    max_step = max_step
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
    num_epochs = num_epochs,
    num_iters = num_iters,
    num_episodes = num_episodes,
    num_samples_iter = num_samples_iter,
    num_samples_iter_history = num_samples_iter_history
)

learn!(trainer)