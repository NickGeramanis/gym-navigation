import gym

# env = gym.make('gym_navigation:NavigationTrack-v0', track_id=1)
env = gym.make('gym_navigation:NavigationGoal-v0', track_id=1)
env.action_space.seed(42)

observation, info = env.reset(seed=42, return_info=True)

for _ in range(1000):
    observation, reward, done, info = env.step(env.action_space.sample())
    env.render()

    if done:
        observation, info = env.reset(return_info=True)

env.close()