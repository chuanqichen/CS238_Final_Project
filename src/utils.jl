using DrWatson
using Flux

device = gpu

outputdir(args...) = projectdir("outputs", args...)

nn_weight_path(output_subdir::String, i::Int) = outputdir(output_subdir, "iter_$(lpad(i,5,"0")).bson")

function save_hp(trainer, output_subdir::String)
    @unpack num_iters, num_episodes, num_samples_iter, num_samples_iter_history = trainer
    @unpack d, m, c = trainer.mcts_nn
    @unpack goal, γ, state_repr = trainer.env

    out = Dict()
    out["num_iters"] = num_iters
    out["num_episodes"] = num_episodes
    out["num_samples_iter"] = num_samples_iter
    out["num_samples_iter_history"] = num_samples_iter_history

    out["d"] = d
    out["m"] = m
    out["c"] = c

    out["goal"] = goal
    out["discount"] = γ
    out["state_repr"] = string(state_repr)

    wsave(outputdir(output_subdir, "params.jld2"), out)
end