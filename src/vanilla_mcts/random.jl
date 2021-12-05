include("board.jl")
include("mcts.jl")
include("simulate.jl")
using ProgressBars
using Game2048: move, initbboard, Dirs
discount_factor = 1

TR(s, a) = move(s, a)
P_2048 = MDP_mcts(discount_factor, nothing,  instances(Dirs), nothing, nothing, TR )
println(P_2048.𝒜)
println([Integer(a) for a in P_2048.𝒜])

U(s) = simulate_game(s)
U_1(s) = simulate_weighted(s)
#U_nn(s) = trained_network(s)
d = 2
exploration_factor = 0.9
m = 100
println("d: ", d)
println("m: ", m)

function get_mcts_struct(m, d)
    mcts_struct = MonteCarloTreeSearch(
        P_2048, 
        Dict(),
        Dict(),
        d, 
        m, 
        exploration_factor, 
        U, 
    )
    return mcts_struct
end

function play(mcts_struct, display)
    init_board = initbboard()
    (possible, _) = possible_moves(init_board)


    curr_board = init_board
    while (length(possible)>0)
        action_to_take = mcts_struct(curr_board)
        curr_board = move(curr_board, action_to_take)
        curr_board = add_tile(curr_board)
        possible, _ = possible_moves(curr_board)
    end
    if display
        display(curr_board)
    
        println(get_value(curr_board))
    end 
    return get_value(curr_board)
end
using Base.Iterators

iters = 1000
val = 0
for i in 1:iters
    global val
    init_board = initbboard()
    val += U(init_board)
end
print(val/iters)