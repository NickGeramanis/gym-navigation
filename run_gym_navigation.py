import gym

env_name = 'gym_navigation:NavigationGoal-v0'
#env_name = 'gym_navigation:Navigation-v0'
env = gym.make(env_name)

for i_episode in range(20):
    done = False
    observation = env.reset()
    env.pose[0] = 0
    env.pose[1] = 9
    env.pose[2] = 0

    while not done:
        env.render()
        input()
        #print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

env.close()
