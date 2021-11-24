using Parameters
using AlphaZero; import AlphaZero.GI
using CommonRLInterface; const rli = CommonRLInterface;


using Game2048: Bitboard, Dirs, initbboard, move, bitboard_to_array

@with_kw mutable struct Env2048 <: AbstractEnv
    goal::Int = 2048; @assert ispow2(goal)
    γ::Float64 = 1.0
    board::Bitboard = initbboard()
    max_step::Int = 1000 + max(0, log2(goal)-11) * 1000
    curr_step::Int = 0
end

rli.reset!(env::Env2048) = (env.board = initbboard())
rli.actions(env::Env2048) = Vector(0:Dirs.size-1)
rli.observe(env::Env2048) = env.board
rli.terminated(env::Env2048) = (env.goal == maximum_tile_value(env.board)) || (env.curr_step > env.max_step)
function rli.act!(env::Env2048, action::Integer) 
    new_board = move(env.board, Dirs(action))
    env.board = new_board
    env.curr_step += 1
    
    if rli.terminated(env) && (env.goal == maximum_tile_value(env.board))
        return 1.0
    else
        return 0.0
    end
end

@provide rli.player(env::Env2048) = 1 # 2048 is single-player Game2048
@provide rli.players(env::Env2048) = [1]
@provide rli.valid_actions(env::Env2048) = valid_actions(env.board)
@provide function rli.valid_action_mask(env::Env2048)
    valid_ints = rli.valid_actions(env)
    mask = falses(length(rli.actions(env)))
    mask[valid_ints .+ 1] .= 1
    return mask
end
@provide rli.clone(env::Env2048) = Env2048(goal = env.goal, γ = env.γ, board = env.board, curr_step = env.curr_step)
@provide rli.state(env::Env2048) = env.board
@provide rli.setstate!(env::Env2048, new_board::Bitboard) = (env.board = new_board)

GI.render(env::Env2048) = display(env.board)
GI.vectorize_state(env::Env2048, state::Bitboard) = convert(Array{Float32},bitboard_to_array(state))
GI.heuristic_value(::Env2048) = 0.0
GI.action_string(env::Env2048, action) = string(Dirs(action))

GameSpec() = CommonRLInterfaceWrapper.Spec(Env2048())


maximum_tile_value(board::Bitboard) = 2 ^ maximum(bitboard_to_array(board))

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
valid_actions(s::Bitboard) = [Integer(action) for action in keys(valid_transitions(s))]

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

# AlphaZero.Scripts.test_game(GameSpec())

