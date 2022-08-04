from gym.envs.registration import register


register(
    id='GraphEnv-v15',
    entry_point='gym_graph.envs:Env15',
)

register(
    id='GraphEnv-v16',
    entry_point='gym_graph.envs:Env16',
)

register(
    id='GraphEnv-v20',
    entry_point='gym_graph.envs:Env20',
)
