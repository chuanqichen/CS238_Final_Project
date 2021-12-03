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

# * DEBUGGING
# d = 4
# m = 2
# c = 0.9


# num_iters = 3
# num_episodes = 2
# num_epochs = 1
# num_samples_iter = 2e4
# num_samples_iter_history = 3
# num_evals = 1

#* TRAINING
# MCTS Spec
d = 1000
m = 5
c = 4

# AlphaZero Spec
num_epochs = 10

num_iters = 3000
num_episodes = 20
num_samples_iter = 2e4
num_samples_iter_history = 2
num_evals = 9


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
    num_epochs = num_epochs,
    num_iters = num_iters,
    num_episodes = num_episodes,
    num_samples_iter = num_samples_iter,
    num_samples_iter_history = num_samples_iter_history
)

learn!(trainer)