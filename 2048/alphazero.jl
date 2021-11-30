using Random
using Parameters
using StatsBase
using DataStructures;
using Game2048
using AlphaZero; import AlphaZero.GI;
using CommonRLInterface; const rli = CommonRLInterface;

include("game.jl")
include("mcts_nn.jl")
include("network.jl")

"""
AlphaZero trainer that uses MCTS_NN as a policy improvement operator. 
Trains the neural network to predict both action distribution for policy and value estimation for critic
"""
@with_kw mutable struct AlphaZeroTrainer
    env
    mcts_nn::MonteCarloTreeSearchNN
    net
    opt = ADAM(3e-4)

    num_sims::Integer = 1 # num of MCTS simulations to run on a state to produce one decision
    num_iters::Integer = 1 # num of Generalized Policy Iteration (GPI) loops
    num_episodes::Integer = 1 # num of games to play per GPI Iteration
    num_samples_iter::Integer = 1e6 # number of samples to train the network per GPI iteration
    num_samples_iter_history::Integer = 20 # number of iterations to keep in case we need to train offline

    samples_iter_history = CircularBuffer(num_samples_iter_history) # stores training data generatd from play!

end

function play_one_episode!(trainer::AlphaZeroTrainer)
    @unpack env, mcts_nn = trainer
    @unpack symmetries = env

    samples_no_reward = []
    rli.reset!(env)
    r=0.0
    while !rli.terminated(env)
        s = rli.state(env)
        τ = 1.0
        action_probs = mcts_nn(s, τ)
        action = sample(rli.actions(env), Weights(action_probs))
        r = rli.act!(env, action)
        append!(samples_no_reward, (s=s, p=action_probs))

        # data augmentation w/ symmetries
        for (aug_s, perm_action_mapping) in symmetries(s)
            perm_action_probs = action_probs[perm_action_mapping]
            append!(samples_no_reward, (s=aug_s, p=perm_action_probs))
        end
    end

    samples = [(s=x.s, p=x.p, r=r) for x in samples_no_reward]
    return samples
end

function learn!(trainer::AlphaZeroTrainer)
    @unpack env, net, opt, mcts_nn, samples_iter_history = trainer
    @unpack num_iters, num_episodes, num_sims, num_samples_iter, num_samples_iter_history = trainer

    for i in 1:num_iters # Play! Policy Evaluation
        iter_samples = CircularBuffer(num_samples_iter)
        for j in 1:num_episodes
            append!(iter_samples, play_one_episode!(trainer))
        end
        append(samples_iter_history, iter_samples)

        # Train! Policy Improvement
        training_samples = []
        for iter_samples in samples_iter_history
            append!(training_samples, iter_samples)
        end
        shuffle!(training_samples)
        x = [sample.s for sample in training_samples]
        Flux.train!(loss, params(net), [x, training_samples], opt)

    end
end