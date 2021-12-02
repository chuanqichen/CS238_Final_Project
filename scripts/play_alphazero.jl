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

weight_fp  = "outputs/2021-12-02-14-17-24/best_iter_00051.bson" 
@load weight_fp net

goal = 2048
γ = 1.0
d = 800
m = 10
c = 0.9
τ = 0.0 # 0 is greedy

n = 100

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

# tile, score = play_game(deepcopy(env), deepcopy(mcts_nn), τ=τ)
# println("\nScore: $(score)")
# println("Maximum Tile: $(tile)")

tiles, scores = play_n_games(deepcopy(env), deepcopy(mcts_nn), n,  τ=τ)
best_tile, best_score, bested = compare_scores(0, 0, tiles, scores)
# println("\nTiles:  $([lpad(s, 6) for s in tiles])")
# println("Scores: $([lpad(s, 6) for s in scores])")
println("Best Tile:  $(best_tile)")
println("Best Score: $(best_score)")
