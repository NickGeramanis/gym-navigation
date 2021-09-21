import gym

env_name = 'gym_navigation:Navigation-v0'
env = gym.make(env_name, track_id=1)

for i_episode in range(10):
    done = False
    env.reset()

    while not done:
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

env.close()
