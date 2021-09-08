from gym.envs.registration import register

register(
    id='Navigation-v0',
    entry_point='gym_navigation.envs:NavigationEnv',
)
register(
    id='NavigationGoal-v0',
    entry_point='gym_navigation.envs:NavigationGoalEnv',
)
