module GM2048
using Base.Threads
using DecisionMakingProblems
include("dm.jl")

# https://github.com/algorithmsbooks/
import DecisionMakingProblems.MDP
using DecisionMakingProblems
using DecisionMakingProblems: TwentyFortyEight, MDP
Board = DecisionMakingProblems.Board
Action = DecisionMakingProblems.TwentyFortyEightAction
initial_board = DecisionMakingProblems.initial_board
print_board = DecisionMakingProblems.print_board
score_board = DecisionMakingProblems.score_board

function bitboard_to_array(board)::Array{Int8, 2}
    ROWMASK=UInt16(2^16-1)
    CELLMASK=UInt16(2^4-1)
    outboard = Array{Int8, 2}(undef, 4, 4)

    rowid = 1
    # take one row at a time
    for row_shift in 48:-16:0
        row = (board >> row_shift) & ROWMASK

        # populate the right cell for each column
        for colid in 4:-1:1
            outboard[rowid, colid] = row & CELLMASK
            row >>= 4
        end
        rowid += 1
    end
    outboard
end

get_value(board) = sum(2^power for power in bitboard_to_array(board))


function HeuristicScore(s; maxWeight = 1.0, mergeWeight = 0.1, mono2Weight  = 1.0, emptyWeight  = 2.7, sum_pow = 2, mono2power= 1)
    array = bitboard_to_array(s)
    # Heuristic score
    sum = 0;
    empty::Int = 0;
    merges = 0;
    monotonicity_left = 0;
    monotonicity_right = 0;
    monotonicity_up = 0;
    monotonicity_down = 0;
    
    for j in 1:4
        prev = 0;
        counter = 0;
        line = array[j, :]
        for i in 1:4
            rank = line[i];
            sum += rank^sum_pow;
            if rank == 0 
                empty += 1
            else 
                if (prev == rank) 
                    counter +=1
                elseif (counter > 0) 
                    merges += 1 + counter;
                    counter = 0;
                end
                prev = rank;
            end 
        end 
        
        if (counter > 0) 
            merges += 1 + counter;
        end 

        for i  in 2:4
            if (line[i-1] > line[i]) 
                monotonicity_left += line[i-1] ^ mono2power - line[i] ^ mono2power;
            else 
                monotonicity_right += line[i] ^ mono2power - line[i-1] ^ mono2power;
            end
        end    
    end 
    
    for j in 1:4
       line = array[:, j]
        for i  in 2:4
            if (line[i-1] > line[i]) 
                monotonicity_up += line[i-1] ^ mono2power - line[i] ^ mono2power;
            else 
                monotonicity_down += line[i] ^ mono2power - line[i-1] ^ mono2power;
            end
        end    
    end 
    
    monotonicity_value = (min(monotonicity_left, monotonicity_right) - 
                          min(monotonicity_up, monotonicity_down) ) / log(2)
    max_value = maximum(array)/log(2)
        
    heuristic_score = max_value * maxWeight
       + merges * mergeWeight
       - monotonicity_value * mono2Weight
       + log(empty) * emptyWeight
    return heuristic_score
end 

function MonteCarloTreeSearchMT(U::Function; d = 10, m = 100, c = 100.0, n = 10)
    DM.MonteCarloTreeSearchTreeMT(
        MDP(TwentyFortyEight(γ=0.99)), # 𝒫, MDP problem 
        [Dict{Tuple{Board, Action}, Int64}() for _ in 1:n], # N, visit counts for each state/action pair
        [Dict{Tuple{Board, Action}, Float32}() for _ in 1:n], # Q, action value estimates for each state/action pair
        d, # maximum depth = 10 by default
        m, # number of simulations = 100 by default
        c, # exploration constant = 100 by default
        U, # value function estimate 
        n  # number of parallel trees
    )
end

"""
Play 2048 to completion using the given policy.
The final score is returned.
Note that this core is "correct" in that we track whether 2 or 4 tiles are generated
and update the score appropriately.
"""
function play_game_using_policy(π::Function; max_illegal = 10, show_board=false)
    s = initial_board()
    # Number of moves.
    moveno = 0

    # Number of illegal moves.
    num_illegal = 0

    # Cumulative penalty for obtaining free 4 tiles, as
    # when computing the score of merged tiles we cannot distinguish between
    # merged 2-tiles and spawned 4 tiles.
    scorepenalty = score_board(s)

    while !DecisionMakingProblems.isdone(s) && num_illegal < max_illegal

        moveno += 1
        a = π(s)
        if a == DecisionMakingProblems.NONE
            break
        end

        s′ = DecisionMakingProblems.move(s, a)
        if s′ == s
            moveno -= 1
            num_illegal += 1
            continue
        else
            num_illegal = 0
        end

        tile = DecisionMakingProblems.draw_tile()
        if tile == 2
            scorepenalty += 4
        end
        s = DecisionMakingProblems.insert_tile_rand(s′, tile)
    end

    if show_board
        print_board(s)
    end 

    return 2^DecisionMakingProblems.get_max_rank(s), score_board(s) - scorepenalty, moveno
end

function play_game_using_policy_mt(π::Function; max_illegal = 10, n=10)
    rs = Vector{Tuple{Int64, Float32, Int64}}(undef, n)
    @threads for i in 1:n
        rs[i] = play_game_using_policy(π; max_illegal)
    end
    return getindex.(rs, 1), getindex.(rs, 2), getindex.(rs, 3)
end    

function PlayGames_with_HeuristicMCTS(;n=100) 
    mcts_U_policy = DM.MonteCarloTreeSearch(
        MDP(TwentyFortyEight(γ=0.99)), # 𝒫
        Dict{Tuple{UInt64, UInt8}, Float64}(), # Q
        Dict{Tuple{UInt64, UInt8}, Float64}(), # N
        100, # d
        300, # m
        50.0, # c
        s -> HeuristicScore(s; maxWeight = 1.0, mergeWeight = 0.1, mono2Weight  = 2.0, emptyWeight  = 2.7, sum_pow = 2, mono2power= 1) # U(s) # U
    );

    games_results_U_policy = [play_game_using_policy(s -> mcts_U_policy(s)) for _ in 1:n]
    max_scores, sum_scores = getindex.(games_results_U_policy, 1), getindex.(games_results_U_policy, 2)
    win_2048_rate = count(i->i==2048, max_scores)/size(max_scores)[1]
    return win_2048_rate, max_scores, sum_scores
end


end #module 