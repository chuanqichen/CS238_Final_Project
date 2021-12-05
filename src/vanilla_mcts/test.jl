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

ds = [1, 2, 3, 4, 5]
ms = [10, 100, 1000]
iters = 100
results = zeros((length(ds), length(ms)))
timers = zeros((length(ds), length(ms)))

for (i, d) in enumerate(ds)
    for (j, m) in tqdm(enumerate(ms))
        val_sum = 0.0
        start = time()

        for iter in 1:iters
            mcts_struct = get_mcts_struct(m, d)
            val = play(mcts_struct, false)
            end_ = time()
            val_sum +=val
        end
        end_ = time()
        time_taken = end_-start
        avg = val_sum/iters
        avg_time = time_taken/iters
        timers[i, j] = avg_time

        println("m: ", m, ", d: ", d, "Avg: ", avg)
        results[i, j] = avg
    end
end


using Plots
f = open("timers.txt", "w")
f2 = open("results.txt", "w")
for i in 1:length(ds)
    for j in 1:length(ms)
      
        print(f, ms[j])
        print(f, ",")
        print(f, ds[i])
        print(f, ": ")

        
        print(f2, ms[j])
        print(f2, ",")
        print(f2, ds[i])
        print(f2, ": ")


        println(f, timers[i, j])
        println(f2, results[i, j])
    end
end
close(f)
close(f2)
  
plot(ds, results[:, 1], xlabel="d", ylabel="Score", label="m=10")
plot!(ds, results[:, 2], xlabel="d", ylabel="Score", label="m=100")
plot!(ds, results[:, 3], xlabel="d", ylabel="Score", label="m=1000")
