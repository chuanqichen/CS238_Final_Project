include("board.jl")
include("mcts.jl")
include("simulate.jl")
using Game2048: move, initbboard, Dirs
discount_factor = 1

TR(s, a) = move(s, a)
P_2048 = MDP_mcts(discount_factor,nothing,  instances(Dirs),nothing, nothing, TR )
println(P_2048.𝒜)
println([Integer(a) for a in P_2048.𝒜])

U(s) = simulate_game(s)
U_1(s) = simulate_weighted(s)
d = 2
exploration_factor = 0.9
m = 1000
println("d: ", d)
println("m: ", m)


mcts_struct = MonteCarloTreeSearch(
    P_2048, 
    Dict(),
    Dict(),
    d, 
    m, 
    exploration_factor, 
    U, 
)

function play()
    init_board = initbboard()
    (possible, _) = possible_moves(init_board)


    curr_board = init_board
    while (length(possible)>0)
        action_to_take = mcts_struct(curr_board)
        curr_board = move(curr_board, action_to_take)
        curr_board = add_tile(curr_board)
        possible, _ = possible_moves(curr_board)
        display(curr_board)
    end
    display(curr_board)
    println()
    println(get_value(curr_board))
        

end
play()
