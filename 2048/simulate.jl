include("game.jl")
using Game2048: move, add_tile

function simulate_game(curr_board)
    while true
        possible = keys(valid_transitions(curr_board))
        if length(possible)==0
            break
        end
        curr_board = move(curr_board, rand(possible))
        curr_board = add_tile(curr_board)     
    end
    
    return get_value(curr_board)
end

function power(x)
    return 2^x
end

function simulate_weighted(curr_board)
    weights = [0 1 2 3; 7 6 5 4; 8 9 10 11; 15 14 13 12]*10
    k_max = 20
    k = 1
    while k<k_max
        k+=1
        possible_transitions = valid_transitions(curr_board)
        possible, boards = keys(possible_transitions), values(possible_transitions)
        if length(possible)==0
            break
        end
        best_move = argmax(
            Dict(
                a=> sum(power.(bitboard_to_array(boards[a])).*weights) for a in possible
            )
        )

        
        curr_board = move(curr_board, best_move)
        curr_board = add_tile(curr_board)     
    end
    arr = power.(bitboard_to_array(curr_board))
    return sum(arr.*weights)
end