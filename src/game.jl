using Parameters
using StatsBase
using Flux
using AlphaZero; import AlphaZero.GI
using CommonRLInterface; const rli = CommonRLInterface;

import Game2048
using Game2048: Bitboard, Dirs, initbboard, move, add_tile, bitboard_to_array

@with_kw mutable struct Env2048 <: AbstractEnv
    goal::Int = 2048; @assert ispow2(goal)
    γ::Float64 = 1.0
    state_repr = Vector # Vector, Matrix, Bitboard (not used but kept anyway)
    max_step::Int = 1000 + max(0, log2(goal)-11) * 1000

    board::Bitboard = initbboard()
    curr_step::Int = 0
    T = transition
    R = reward
    symmetries = symmetries
end

function bitboard_to_state(state_repr, bitboard::Bitboard)
    if state_repr == Bitboard
        return bitboard
    elseif state_repr <: Vector
        return bitboard |> bitboard_to_array |> array_to_bitvector |> Vector{Float32}
    elseif state_repr <: Matrix
        return bitboard |> bitboard_to_array
    end
end

function state_to_bitboard(s)::Bitboard
    if     isa(s, Bitboard)
        return s
    elseif isa(s, Vector)
        return BitVector(s.>0) |> bitvector_to_array |> array_to_bitboard
    elseif isa(s, Matrix)
        return s |> array_to_bitboard
    end
end

# Mandatory functions for CommonRLInterface
function rli.reset!(env::Env2048)
    env.board = initbboard()
    env.curr_step = 0
end
rli.actions(env::Env2048) = Vector(0:Dirs.size-1)
rli.observe(env::Env2048) = bitboard_to_state(env.state_repr, env.board)
rli.terminated(env::Env2048) = terminated(env.board, env.goal, env.curr_step, env.max_step)
function rli.act!(env::Env2048, action::Integer)
    _, board_next, r, _ = transition(env.board, action, env.goal, env.curr_step, env.max_step)
    env.board = board_next
    env.curr_step += 1
    return r
end

maximum_tile_value(board::Bitboard) = 2 ^ maximum(bitboard_to_array(board))
goal_reached(board::Bitboard, goal::Int) = (goal == maximum_tile_value(board))

function terminated(s, goal::Int, curr_step::Int, max_step::Int)::Bool
    s = state_to_bitboard(s)
    can_move = length(valid_actions(s)) > 0
    return  (goal_reached(s, goal) || curr_step >= max_step || !can_move) ? true : false
end

function reward(s, goal::Int, curr_step::Int, max_step::Int)
    s = state_to_bitboard(s)
    done = terminated(s, goal, curr_step, max_step)
    r = 0.0
    if done
        r =  (goal_reached(s, goal)) ? 1.0 : 0.0
    end
    return r
end

function transition(s, action::Integer, goal, curr_step::Int, max_step::Int)
    board = state_to_bitboard(s)
    board_next = move(s, Dirs(action)) |> add_tile
    s′ = bitboard_to_state(typeof(s), board_next)

    new_step = curr_step + 1
    done = terminated(board_next, goal, new_step, max_step)
    r = reward(board_next, goal, new_step, max_step)
    return (s′=s′, board_next = board_next,r=r, done=done)
end

# Optional Functions for GameInterface of AlphaZero.jl
GameSpec() = CommonRLInterfaceWrapper.Spec(Env2048())

@provide rli.player(env::Env2048) = 1 # 2048 is single-player Game2048
@provide rli.players(env::Env2048) = [1]
@provide rli.valid_actions(env::Env2048) = valid_actions(env.board)
@provide rli.valid_action_mask(env::Env2048) = valid_action_mask(env.board, length(rli.actions(env)))


function valid_transitions(s)::Dict{Dirs, Bitboard}
    board = state_to_bitboard(s)
    boards = Dict()
    for direction in instances(Dirs)
        temp_board = move(board, direction)
        if temp_board != board
            boards[direction] = temp_board
        end
    end
    return boards
end
valid_actions(s)::Vector{Int} = [Integer(action) for action in keys(valid_transitions(s))]


function valid_action_mask(s, 𝒜_size)
    valid_action_idc::Vector{Int} = valid_actions(s)
    mask = falses(𝒜_size)
    mask[valid_action_idc .+ 1] .= 1
    return mask
end

@provide rli.clone(env::Env2048) = deepcopy(env) 
@provide rli.state(env::Env2048) = deepcopy(rli.observe(env)) # fully observable
@provide rli.setstate!(env::Env2048, new_state) = (env.board = state_to_bitboard(new_state))

function GI.vectorize_state(env::Env2048, state)
    if env.state_repr <: Matrix
        state = Matrix{Float32}(state)
    elseif env.state_repr <: Vector
        state = Vector{Float32}(state)
    end
    return state
end
function symmetries(state)
    input_type = typeof(state)
    if isa(state, Vector)
        state = BitVector(state.>0) |> bitvector_to_array
    end

    board_rotl0 = state
    board_rotl1 = board_rotl0 |> rotl90
    board_rotl2 = board_rotl1 |> rotl90
    board_rotl3 = board_rotl2 |> rotl90
    board_rolt0_vflip = reverse(board_rotl0, dims=2)
    board_rolt1_vflip = reverse(board_rotl1, dims=2)
    board_rolt2_vflip = reverse(board_rotl2, dims=2)
    board_rolt3_vflip = reverse(board_rotl3, dims=2)
    
    transform(bm) = bitboard_to_state(input_type, array_to_bitboard(bm))
    return [
        (board_rotl1       |> transform, [3,4,2,1]),
        (board_rotl2       |> transform, [2,1,4,3]),
        (board_rotl3       |> transform, [4,3,1,2]),
        (board_rolt0_vflip |> transform, [2,1,3,4]),
        (board_rolt1_vflip |> transform, [4,3,2,1]),
        (board_rolt2_vflip |> transform, [1,2,4,3]),
        (board_rolt3_vflip |> transform, [3,4,1,2])
     ]
end

GI.symmetries(::GI.AbstractGameSpec, state) = symmetries(state)

GI.render(env::Env2048) = display(env.board)
GI.heuristic_value(::Env2048) = 0.0
GI.action_string(env::Env2048, action) = string(Dirs(action))


shift_array_64 = reshape(Vector(60:-4:0), (4,4))
function array_to_bitboard(state)
    bb = UInt64(0)
    state′ = transpose(state)
    for i in eachindex(state')
        bb_component = UInt(state'[i]) << shift_array_64[i]
        bb |= bb_component
    end
    return Bitboard(bb)
end

shift_array_256 = reshape(Vector(240:-16:0), (4,4))
function array_to_bitvector(bm)
    bv = BitVector(zeros(256))
    for i in eachindex(bm')
        bv_component = ((Flux.onehot(bm'[i], 0:255)) |> reverse |> BitVector) << shift_array_256[i]
        bv .|= bv_component
    end
    return bv
end

function bitvector_to_array(bv)
    bm = zeros(Int8, 4, 4)
    for i in 1:16
        start = (i-1)*16 + 1
        onehot_vec = bv[start:start+15]
        power = findfirst(x->!iszero(x), reverse(onehot_vec)) - 1
        bm'[i] = power
    end
    return bm
end


get_value(board::Bitboard) = sum(2^power for power in bitboard_to_array(board))


Game2048.move(s, direction::Dirs) = move(state_to_bitboard(s), direction)


function play_game(env::Env2048, mcts_nn; τ::Float64 = 0.0, verbose::Bool = false)
    rli.reset!(env)
    while !rli.terminated(env)
        curr_board = rli.state(env)
        valid_actions = rli.valid_actions(env)
        action_probs = mcts_nn(curr_board, τ = τ)
        action_to_take = sample(rli.actions(env), Weights(action_probs))
        if verbose
            println("\n $valid_actions -> $action_to_take")
            display(env.board); 
        end
        rli.act!(env, action_to_take)
    end
    return maximum_tile_value(env.board), get_value(env.board)
end

function play_n_games(env, mcts_nn, n::Int; τ::Float64 = 0.0, verbose::Bool = false)
    tiles = zeros(n)
    scores = zeros(n)
    for i in 1:n
        tiles[i], scores[i] = play_game(deepcopy(env), deepcopy(mcts_nn), τ = τ, verbose = verbose)
    end
    return tiles, scores
end

function compare_scores(best_tiles, best_score, tiles::Vector, scores::Vector)
    n = length(tiles)
    bested_record = falses(n)
    best_tile, best_score, bested = 0, 0, false
    for i in 1:n
        best_tile, best_score, bested = compare_score(best_tile, best_score, tiles[i], scores[i])
        bested_record[i] = bested
    end
    bested = (sum(bested_record) > round(n/2, RoundDown)) ? true : false

    return best_tile, best_score, bested
end

function compare_score(best_tile, best_score, tile::Number, score::Number)
    old_score = best_score
    old_tile = best_tile

    if tile > best_tile
        best_score = score
        best_tile = tile
    elseif tile == best_tile
        if score > best_score
            best_score = score
        end
    end
    bested = (best_score != old_score || best_tile != old_tile) ? true : false

    return best_tile, best_score, bested
end

# AlphaZero.Scripts.test_game(GameSpec())