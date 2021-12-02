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

@load outputdir("2021-12-01-10-32-46/iter_00004.bson") net

goal = 2048
γ = 1.0
d = 200
m = 4
c = 0.9
τ = 0.0 # 0 is greedy


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

function play(env, mcts_nn, τ)
    rli.reset!(env)
    while !rli.terminated(env)
        curr_board = rli.state(env)
        valid_actions = rli.valid_actions(env)
        action_probs = mcts_nn(curr_board, τ = τ)
        action_to_take = sample(rli.actions(env), Weights(action_probs))
        println("\n $valid_actions -> $action_to_take")
        display(env.board); 
        rli.act!(env, action_to_take)
    end
    println("\nScore: $(get_value(env.board))")
    println("Maximum Tile: $(maximum_tile_value(env.board))")
end

play(env, mcts_nn, τ)