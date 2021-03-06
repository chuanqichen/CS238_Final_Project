using Random
using Dates
using DrWatson
using Parameters
using StatsBase
using DataStructures
using Flux
using ProgressMeter
using Game2048
using CommonRLInterface; const rli = CommonRLInterface;
using BSON: @save
using CUDA

include("game.jl")
include("mcts_nn.jl")
include("network.jl")
include("utils.jl")

struct Sample
    s::Array{Float32}
    p::Array{Float32, 1}
    r::Array{Float32, 1}
end

"""
AlphaZero trainer that uses MCTS_NN as a policy improvement operator. 
Trains the neural network to predict both action distribution for policy and value estimation for critic
"""
@with_kw mutable struct AlphaZeroTrainer
    env
    mcts_nn::MonteCarloTreeSearchNN
    net
    opt = Flux.Optimiser(ClipNorm(1), ADAM(1e-3))

    num_epochs::Int = 1

    num_iters::Int = 1 # num of Generalized Policy Iteration (GPI) loops
    num_episodes::Int = 1 # num of games to play per GPI Iteration
    num_samples_iter::Int = 1e6 # number of samples to train the network per GPI iteration
    num_eval::Int = 1 # number of games to play after each GPI loop to test new trained mdoel & (maybe) save it
    num_step_until_greedy::Int = 15 # first this many steps get exploratory action probability output from MCTSNN, greedy afterwards

    samples_iter = CircularBuffer{Sample}(num_samples_iter)
    samples_episode = CircularBuffer{Sample}(num_samples_iter)

end

function play_one_episode!(trainer::AlphaZeroTrainer)
    @unpack env, mcts_nn = trainer
    @unpack num_step_until_greedy = trainer

    empty!(trainer.samples_episode)
    rli.reset!(env)
    r = 0.0 # whether game was won
    while !rli.terminated(env)
        s = rli.state(env)
        τ = Float64(env.curr_step < num_step_until_greedy)
        action_probs = mcts_nn(s, τ=τ)
        action = sample(rli.actions(env), Weights(action_probs))
        r = rli.act!(env, action)

        # Gather experience + data augmentation w/ symmetries
        push!(trainer.samples_episode, Sample(s, action_probs, [-Inf]))
        for (aug_s, perm_action_mapping) in env.symmetries(s)
            push!(trainer.samples_episode, Sample(aug_s, action_probs[perm_action_mapping], [-Inf]))
        end
    end

    r = max(r, env.curr_step / env.max_step) # step-based reward engineering
    for sample in trainer.samples_episode
        sample.r[1] = r
    end
    
    return trainer.samples_episode
end

function learn!(trainer::AlphaZeroTrainer)
    @unpack env, opt, mcts_nn, num_epochs = trainer
    @unpack num_iters, num_episodes, num_samples_iter, num_eval = trainer

    output_subdir = outputdir(Dates.format(now(), "Y-mm-dd-HH-MM-SS"))
    save_hp(trainer, output_subdir)

    best_tile, best_score = 0, 0
    @showprogress for i in 1:num_iters
        println("\nGPI Iteration $i")
        println("Policy Evaluation")
        empty!(trainer.samples_iter)
        for j in 1:num_episodes
            append!(trainer.samples_iter, play_one_episode!(trainer))
        end

        println("Policy Improvement")
        samples_s = Flux.batch([sample.s for sample in trainer.samples_iter]) |> device
        samples_p = Flux.batch([sample.p for sample in trainer.samples_iter]) |> device
        samples_r = Flux.batch([sample.r for sample in trainer.samples_iter]) |> device
        dl = Flux.DataLoader((samples_s, samples_p, samples_r), batchsize=128, shuffle=true)
        trainer.net = trainer.net |> device

        trainmode!(trainer.net)
        Flux.@epochs num_epochs Flux.train!(loss, params(trainer.net), dl, opt) # , cb = () -> println(loss(first(dl)...))
        testmode!(trainer.net)
        trainer.net = trainer.net |> cpu
 
        # Compare model and save best one
        tiles, scores, boards = play_n_games(deepcopy(env), deepcopy(mcts_nn), num_eval, τ=0.0)
        best_tile, best_score, _, bested = compare_scores(best_tile, best_score, tiles, scores, boards)

        net = trainer.net
        bested ? (@save nn_best_weight_path(output_subdir, i) net) : nothing
        (i % 10 == 0) ? (@save nn_weight_path(output_subdir, i) net) : nothing
    end
end