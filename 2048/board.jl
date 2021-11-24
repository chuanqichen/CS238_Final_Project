using Parameters
using CommonRLInterface;
const rli = CommonRLInterface;

using Game2048: Dirs, bitboard_to_array, move, Bitboard

@with_kw mutable struct Env2048 <: AbstractEnv
    goal::Int; @assert ispow2(goal)
    γ::Float64
    board::Bitboard
end

function Env2048(goal::Integer=2048, γ::Float64=1.0)
    Env2048(goal=goal, γ=γ, board=initbboard())
end

function rli.reset!(env::Env2048)
    env.board = initbboard()
end
rli.actions(env::Env2048) = instances(Dirs)
rli.observe(env::Env2048) = env.board
rli.terminated(env::Env2048) = (env.goal == (2 ^ maximum(bitboard_to_array(env.board))))
function rli.act!(env::Env2048, a) 
    new_board = move(env.board, a)
    env.board = new_board
end

@provide rli.valid_actions(env::Env2048) = keys(valid_transitions(env.board))

function valid_transitions(board::Bitboard)::Dict{Dirs, Bitboard}
    boards = Dict()
    for direction in instances(Dirs)
        temp_board = move(board, direction)
        if temp_board != board
            boards[direction] = temp_board
        end
    end
    return boards
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



