using Random
using Dates
using DrWatson
using Parameters
using StatsBase
using DataStructures;
using ProgressMeter
using Game2048
using AlphaZero; import AlphaZero.GI;
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

    num_iters::Integer = 1 # num of Generalized Policy Iteration (GPI) loops
    num_episodes::Integer = 1 # num of games to play per GPI Iteration
    num_samples_iter::Integer = 1e6 # number of samples to train the network per GPI iteration
    num_samples_iter_history::Integer = 20 # number of GPI of samples to keep: for staleness control + offline saving

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
        action_probs = mcts_nn(s, τ)
        action = sample(rli.actions(env), Weights(action_probs))
        r = rli.act!(env, action)
        push!(samples_no_reward, (s=s, p=action_probs))

        # data augmentation w/ symmetries
        for (aug_s, perm_action_mapping) in env.symmetries(s)
            perm_action_probs = action_probs[perm_action_mapping]
            push!(samples_no_reward, (s=aug_s, p=perm_action_probs))
        end
    end

    samples = [(s=x.s, p=x.p, r=[r]) for x in samples_no_reward]
    return samples
end

function learn!(trainer::AlphaZeroTrainer)
    @unpack env, net, opt, mcts_nn = trainer
    @unpack num_iters, num_episodes, num_samples_iter, num_samples_iter_history = trainer

    output_subdir = outputdir(Dates.format(now(), "Y-mm-dd-HH-MM-SS"))
    save_hp(trainer, output_subdir)

    @showprogress for i in 1:num_iters
        println("GPI Iteration $i")
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

        samples_s = Flux.stack([sample.s for sample in training_samples],1) |> permutedims
        samples_p = Flux.stack([sample.p for sample in training_samples],1) |> permutedims
        samples_r = Flux.stack([sample.r for sample in training_samples],1) |> permutedims

        samples_s = samples_s |> gpu
        samples_p = samples_p |> gpu
        samples_r = samples_r |> gpu
        
        data = [(samples_s, samples_p, samples_r)]
        Flux.train!(loss, params(net), data, opt) 
 
        if i % 1 == 0
            @save nn_weight_path(output_subdir, i) net
        end
    end
end