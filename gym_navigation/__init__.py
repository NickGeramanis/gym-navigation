"""This __init__ file is used to register the environments."""
from gymnasium.envs.registration import register

register(
    id='NavigationTrack-v0',
    entry_point='gym_navigation.envs.navigation_track:NavigationTrack')
register(
    id='NavigationGoal-v0',
    entry_point='gym_navigation.envs.navigation_goal:NavigationGoal')
