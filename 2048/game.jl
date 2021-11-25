using Parameters
using AlphaZero; import AlphaZero.GI
using CommonRLInterface; const rli = CommonRLInterface;

import Game2048
using Game2048: Bitboard, Dirs, initbboard, move, add_tile, bitboard_to_array

@with_kw mutable struct Env2048 <: AbstractEnv
    goal::Int = 2048; @assert ispow2(goal)
    γ::Float64 = 1.0
    board::Bitboard = initbboard()
    max_step::Int = 1000 + max(0, log2(goal)-11) * 1000
    curr_step::Int = 0
    bb_repr::Bool = false
end

# Mandatory functions for CommonRLInterface
rli.reset!(env::Env2048) = (env.board = initbboard())
rli.actions(env::Env2048) = Vector(0:Dirs.size-1)
rli.observe(env::Env2048) = env.bb_repr ? env.board : bitboard_to_array(env.board) 
rli.terminated(env::Env2048) = terminated(env.board, env.goal, env.curr_step, env.max_step)
function rli.act!(env::Env2048, action::Integer)
    s′, r, terminated = transition(env.board, action, env.goal, env.curr_step, env.max_step)
    env.board = s′
    env.curr_step += 1
    return r
end

maximum_tile_value(board::Bitboard) = 2 ^ maximum(bitboard_to_array(board))
goal_reached(board::Bitboard, goal::Int) = (goal == maximum_tile_value(board))

function terminated(s, goal::Int, curr_step::Int, max_step::Int)
    s = isa(s, Bitboard) ? s : array_to_bitboard(s)
    can_move = length(valid_actions(s)) > 0
    return  (goal_reached(s, goal) || curr_step >= max_step || !can_move) ? true : false
end

function reward(s, goal::Int, curr_step::Int, max_step::Int)
    s = isa(s, Bitboard) ? s : array_to_bitboard(s)
    done = terminated(s, goal, curr_step, max_step)
    r = 0.0
    if done
        r =  (goal_reached(s, goal)) ? 1.0 : 0.0
    end
    return r
end

function transition(s, action::Integer, goal, curr_step::Int, max_step::Int)
    s = isa(s, Bitboard) ? s : array_to_bitboard(s)
    s′ = move(s, Dirs(action)) |> add_tile
    new_step = curr_step + 1
    done = terminated(s′, goal, new_step, max_step)
    r = reward(s′, goal, new_step, max_step)
    return (s′=s′, r=r, terminated=done)
end

# Optional Functions for GameInterface of AlphaZero.jl
GameSpec() = CommonRLInterfaceWrapper.Spec(Env2048())

@provide rli.player(env::Env2048) = 1 # 2048 is single-player Game2048
@provide rli.players(env::Env2048) = [1]
@provide rli.valid_actions(env::Env2048) = valid_actions(env.board)
@provide function rli.valid_action_mask(env::Env2048)
    valid_ints = rli.valid_actions(env)
    mask = falses(length(rli.actions(env)))
    mask[valid_ints .+ 1] .= 1
    return mask
end
@provide rli.clone(env::Env2048) = deepcopy(env) 
@provide rli.state(env::Env2048) = deepcopy(rli.observe(env)) # fully observable
@provide function rli.setstate!(env::Env2048, new_state::Union{Bitboard, Matrix}, curr_step::Int=0)
    if !isa(new_state, Bitboard)
        new_state = array_to_bitboard(new_state)
    end
    env.board = new_state
    env.curr_step = curr_step
end

GI.render(env::Env2048) = display(env.board)
GI.vectorize_state(env::Env2048, state::Bitboard) = convert(Array{Float32},bitboard_to_array(state))
GI.heuristic_value(::Env2048) = 0.0
GI.action_string(env::Env2048, action) = string(Dirs(action))

function GI.symmetries(env::Env2048, state::Bitboard)
    board_rotl0 = bitboard_to_array(state)
    board_rotl1 = board_rotl0 |> rotl90
    board_rotl2 = board_rotl1 |> rotl90
    board_rotl3 = board_rotl2 |> rotl90
    board_rolt0_vflip = reverse(board_rotl0, dims=2)
    board_rolt1_vflip = reverse(board_rotl1, dims=2)
    board_rolt2_vflip = reverse(board_rotl2, dims=2)
    board_rolt3_vflip = reverse(board_rotl3, dims=2)

    return [
       (board_rotl1       |> array_to_bitboard, [3,4,2,1]),
       (board_rotl2       |> array_to_bitboard, [2,1,4,3]),
       (board_rotl3       |> array_to_bitboard, [4,3,1,2]),
       (board_rolt0_vflip |> array_to_bitboard, [2,1,3,4]),
       (board_rolt1_vflip |> array_to_bitboard, [4,3,2,1]),
       (board_rolt2_vflip |> array_to_bitboard, [1,2,4,3]),
       (board_rolt3_vflip |> array_to_bitboard, [3,4,1,2])
    ]
end

function GI.symmetries(env::Env2048, state::Matrix)
    board_rotl0 = state
    board_rotl1 = board_rotl0 |> rotl90
    board_rotl2 = board_rotl1 |> rotl90
    board_rotl3 = board_rotl2 |> rotl90
    board_rolt0_vflip = reverse(board_rotl0, dims=2)
    board_rolt1_vflip = reverse(board_rotl1, dims=2)
    board_rolt2_vflip = reverse(board_rotl2, dims=2)
    board_rolt3_vflip = reverse(board_rotl3, dims=2)

    return [
       (board_rotl1      , [3,4,2,1]),
       (board_rotl2      , [2,1,4,3]),
       (board_rotl3      , [4,3,1,2]),
       (board_rolt0_vflip, [2,1,3,4]),
       (board_rolt1_vflip, [4,3,2,1]),
       (board_rolt2_vflip, [1,2,4,3]),
       (board_rolt3_vflip, [3,4,1,2])
    ]
end

function valid_transitions(s)::Dict{Dirs, Bitboard}
    board = isa(s, Bitboard) ? s : array_to_bitboard(s)
    boards = Dict()
    for direction in instances(Dirs)
        temp_board = move(board, direction)
        if temp_board != board
            boards[direction] = temp_board
        end
    end
    return boards
end
valid_actions(s::Union{Bitboard,Matrix})::Vector{Int} = [Integer(action) for action in keys(valid_transitions(s))]


weight_array = reshape(Vector(60:-4:0), (4,4))
function array_to_bitboard(state)
    bb = UInt64(0)
    state′ = transpose(state)
    for i in eachindex(state')
        bb_component = UInt(state'[i]) << weight_array[i]
        bb |= bb_component
    end
    return Bitboard(bb)
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

Game2048.move(s::Matrix, direction::Dirs) = move(array_to_bitboard(s), direction)

# AlphaZero.Scripts.test_game(GameSpec())