using Random
using Dates
using DrWatson
using Parameters
using StatsBase
using DataStructures
using Flux
using ProgressMeter
using Game2048
# using AlphaZero; import AlphaZero.GI;
using CommonRLInterface; const rli = CommonRLInterface;
using BSON: @save
using CUDA

include("game.jl")
include("mcts_nn.jl")
include("network.jl")
include("utils.jl")

"""
AlphaZero trainer that uses MCTS_NN as a policy improvement operator. 
Trains the neural network to predict both action distribution for policy and value estimation for critic
"""
@with_kw mutable struct AlphaZeroTrainer
    env
    mcts_nn::MonteCarloTreeSearchNN
    net
    opt = ADAM(3e-4)

    num_epochs::Int = 1

    num_iters::Int = 1 # num of Generalized Policy Iteration (GPI) loops
    num_episodes::Int = 1 # num of games to play per GPI Iteration
    num_samples_iter::Int = 1e6 # number of samples to train the network per GPI iteration
    num_samples_iter_history::Int = 20 # number of GPI of samples to keep: for staleness control + offline saving
    num_eval::Int = 1 # number of games to play after each GPI loop to test new trained mdoel & (maybe) save it

    samples_iter_history = CircularBuffer(num_samples_iter_history) # stores training data generatd from play!

end

function play_one_episode!(trainer::AlphaZeroTrainer)
    @unpack env, mcts_nn = trainer

    samples_no_reward = []
    rli.reset!(env)
    r = 0.0 # whether game was won
    while !rli.terminated(env)
        s = rli.state(env)
        τ = 1.0
        action_probs = mcts_nn(s, τ=τ)
        action = sample(rli.actions(env), Weights(action_probs))
        r = rli.act!(env, action)
        push!(samples_no_reward, (s=s, p=action_probs))

        # data augmentation w/ symmetries
        for (aug_s, perm_action_mapping) in env.symmetries(s)
            perm_action_probs = action_probs[perm_action_mapping]
            push!(samples_no_reward, (s=aug_s, p=perm_action_probs))
        end
    end

    samples = [( 
        s = convert(Vector{Float16}, x.s),
        p = convert(Vector{Float16}, x.p),
        r = convert(Vector{Float16}, [r])
     ) for x in samples_no_reward]
    return samples
end

function learn!(trainer::AlphaZeroTrainer)
    @unpack env, net, opt, mcts_nn, num_epochs = trainer
    @unpack num_iters, num_episodes, num_samples_iter, num_samples_iter_history, num_eval = trainer

    output_subdir = outputdir(Dates.format(now(), "Y-mm-dd-HH-MM-SS"))
    save_hp(trainer, output_subdir)

    best_tile, best_score = 0, 0
    @showprogress for i in 1:num_iters
        println("\nGPI Iteration $i")
        println("Policy Evaluation")
        iter_samples = CircularBuffer(num_samples_iter)
        for j in 1:num_episodes
            append!(iter_samples, play_one_episode!(trainer))
        end
        push!(trainer.samples_iter_history, iter_samples)

        println("Policy Improvement")
        training_samples = []
        for iter_samples in trainer.samples_iter_history
            append!(training_samples, iter_samples)
        end
        shuffle!(training_samples)

        samples_s = Flux.batch([sample.s for sample in training_samples]) |> device
        samples_p = Flux.batch([sample.p for sample in training_samples]) |> device
        samples_r = Flux.batch([sample.r for sample in training_samples]) |> device
        dl = Flux.DataLoader((samples_s, samples_p, samples_r), batchsize=32)

        Flux.@epochs num_epochs Flux.train!(loss, params(net), dl, opt) 
 
        # Compare model and save best one
        tiles, scores = play_n_games(deepcopy(env), deepcopy(mcts_nn), num_eval,  τ=0.0)
        best_tile, best_score, bested = compare_scores(best_tile, 0, tiles, scores)

        bested ? (@save nn_best_weight_path(output_subdir, i) net) : nothing
        (i % 10 == 0) ? (@save nn_weight_path(output_subdir, i) net) : nothing
    end
end