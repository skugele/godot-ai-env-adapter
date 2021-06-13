from gym.envs.registration import register

register(
    id='simple-animat-v0',
    entry_point='gym_godot.envs:SimpleAnimatWorld',
)