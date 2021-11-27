using AlphaZero
include("game.jl")


Network = NetLib.SimpleNet

netparams = NetLib.SimpleNetHP(
  width=40,
  depth_common=3,
  use_batch_norm=true,
  batch_norm_momentum=1.)



self_play = SelfPlayParams(
sim=SimParams(
    num_games=100,
    num_workers=64,
    batch_size=64,
    use_gpu=true,
    reset_every=2,
    flip_probability=0.,
    alternate_colors=false),
mcts=MctsParams(
    num_iters_per_turn=60,
    cpuct=2.0,
    prior_temperature=1.0,
    temperature=PLSchedule([0, 20, 30], [1.0, 1.0, 0.3]),
    dirichlet_noise_ϵ=0.25,
    dirichlet_noise_α=1.0))

arena = ArenaParams(
sim=SimParams(
    num_games=32,
    num_workers=128,
    batch_size=128,
    use_gpu=true,
    reset_every=2,
    flip_probability=0.5,
    alternate_colors=true),
mcts=MctsParams(
    self_play.mcts,
    temperature=ConstSchedule(0.2),
    dirichlet_noise_ϵ=0.05),
update_threshold=0.05)

learning = LearningParams(
    use_gpu=true,
    use_position_averaging=true,
    samples_weighing_policy=LOG_WEIGHT,
    batch_size=1024,
    loss_computation_batch_size=1024,
    optimiser=Adam(lr=2e-3),
    l2_regularization=1e-4,
    nonvalidity_penalty=1.,
    min_checkpoints_per_epoch=1,
    max_batches_per_checkpoint=100,
    num_checkpoints=1)

params = Params(
    arena=arena,
    self_play=self_play,
    learning=learning,
    num_iters=5,
    ternary_rewards=false,
    use_symmetries=true,
    memory_analysis=nothing,
    mem_buffer_size=PLSchedule(
    [      0,        15],
    [400_000, 1_000_000]))



benchmark_sim = SimParams(
    arena.sim;
    num_games=40,
    num_workers=50,
    batch_size=100)

    benchmark = [
        Benchmark.Single(
          Benchmark.Full(self_play.mcts),
          benchmark_sim),
        Benchmark.Single(
          Benchmark.NetworkOnly(),
          benchmark_sim)]
      
    

experiment = Experiment(
        "2048", GameSpec(), params, Network, netparams, benchmark)

#AlphaZero.Scripts.dummy_run(experiment)
AlphaZero.Scripts.train(experiment)