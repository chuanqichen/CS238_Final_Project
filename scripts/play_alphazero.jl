using DrWatson
@quickactivate
using Parameters
using CommonRLInterface; const rli = CommonRLInterface;
using BSON: @save, @load
using Game2048: move, initbboard, Dirs
using AlphaZero
include("../src/game.jl")
include("../src/mcts_nn.jl")
include("../src/alphazero.jl")
include("../src/network.jl")

weight_fp  = "outputs/2021-12-03-12-29-58/iter_00010.bson"
@load weight_fp net

n = 200

goal = 2048
γ = 1.0
max_step = Inf

d = Inf
m = 20
c = 2.0
τ = 0.0 # 0 is greedy


env = Env2048(
    goal = goal,
    γ = γ,
    state_repr = Vector,
    max_step = max_step
    )

mcts_nn = MonteCarloTreeSearchNN(
    env = env, 
    net = Flux.testmode!(net),
    d = d, 
    m = m, 
    c = c, 
)


tiles, scores, boards = play_n_games(deepcopy(env), deepcopy(mcts_nn), n,  τ=τ)
best_tile, best_score, best_board, bested = compare_scores(0, 0, tiles, scores, boards)
println("Best Tile:  $(best_tile)")
println("Best Score: $(best_score)")
display(best_board)
