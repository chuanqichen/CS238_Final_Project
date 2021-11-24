include("board.jl")
include("mcts.jl")
include("simulate.jl")
using Game2048: move, initbboard, Dirs

goal = 2048
γ = 1.0
env = Env2048(goal, γ)
println(actions(env))
println([Int(a) for a in actions(env)])

U(s) = simulate_game(s)
# U_1(s) = simulate_weighted(s)
d = 2
exploration_factor = 0.9
m = 1000
println("d: ", d)
println("m: ", m)


mcts_struct = MonteCarloTreeSearch(
    𝒫 = env, 
    N = Dict(),
    Q = Dict(),
    d = d, 
    m = m, 
    c = exploration_factor, 
    U = U, 
)

function play()
    init_board = initbboard()
    possible = keys(valid_transitions(init_board))


    curr_board = init_board
    while (length(possible)>0)
        action_to_take = mcts_struct(curr_board)
        curr_board = move(curr_board, action_to_take)
        curr_board = add_tile(curr_board)
        possible = keys(valid_transitions(curr_board))
        display(curr_board)
    end
    display(curr_board)
    println()
    println(get_value(curr_board))
        

end
play()
