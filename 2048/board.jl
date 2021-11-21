
using Game2048: Dirs, bitboard_to_array, move

function possible_moves(board)
    moves = []
    boards = Dict()
    for dir in instances(Dirs)
        temp_board = move(board, dir)
        if temp_board != board
            push!(moves, dir)
            boards[dir] = temp_board
        end
    end
    return (moves, boards)
end

function get_linear_value(board)
    return sum(bitboard_to_array(board))
end
function get_value(board)
    arr = bitboard_to_array(board)

    sum_all = 0
    for i in 1:size(arr, 1)
        for j in 1:size(arr, 2)
            sum_all += 2^arr[i, j]
        end
    end
    return sum_all
end



