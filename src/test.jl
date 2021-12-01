using Parameters
using CommonRLInterface; const rli = CommonRLInterface;

using Game2048: move, initbboard, Dirs
include("game.jl")
include("mcts.jl")
include("simulate.jl")

# Environment Spec
goal = 2048
γ = 1.0
env = Env2048(
    goal = goal,
    γ = γ,
    state_repr = Vector
    )
println(actions(env))

# Algorithm Spec
# U(s) = simulate_game(s)
U(s, goal, curr_step, max_step) = rollout(s, rli.actions(env), goal, curr_step, max_step)
# U_1(s) = simulate_weighted(s)
d = 2
exploration_factor = 0.9
m = 100
println("depth: ", d)
println("num mcts: ", m)


mcts = MonteCarloTreeSearch(
    env = env, 
    N = Dict(),
    Q = Dict(),
    d = d, 
    m = m, 
    c = exploration_factor, 
    U = U, 
)

# function play()
#     init_board = initbboard()
#     possible = keys(valid_transitions(init_board))

#     curr_board = init_board
#     while (length(possible)>0)
#         action_to_take = mcts(curr_board)
#         curr_board = move(curr_board, Dirs(action_to_take))
#         curr_board = add_tile(curr_board)
#         possible = keys(valid_transitions(curr_board))
#         display(curr_board); println()
#     end
#     display(curr_board); println()
#     println("Score: $(get_value(curr_board))")
#     println("Maximum Tile: $(maximum_tile_value(curr_board))")
# end

function play()
    reset!(env)
    while !rli.terminated(env)
        curr_board = rli.state(env)
        valid_actions = rli.valid_actions(env)
        action_to_take = mcts(curr_board)
        println("\n $valid_actions -> $action_to_take")
        display(env.board); 
        rli.act!(env, action_to_take)
    end
    println("\nScore: $(get_value(env.board))")
    println("Maximum Tile: $(maximum_tile_value(env.board))")
end

play()
