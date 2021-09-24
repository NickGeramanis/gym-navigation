from gym.envs.registration import register

register(
    id='Navigation-v0',
    entry_point='gym_navigation.envs.navigation_env:NavigationEnv')
register(
    id='NavigationGoal-v0',
    entry_point='gym_navigation.envs.navigation_goal_env:NavigationGoalEnv')
