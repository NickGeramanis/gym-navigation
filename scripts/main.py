import gym


env = gym.make('gym_navigation:NavigationTrack-v0',
               render_mode='human',
               new_step_api=True,
               track_id=1)
# env = gym.make('gym_navigation:NavigationGoal-v0',
#                render_mode='human',
#                new_step_api=True,
#                track_id=2)
# env.action_space.seed(42)

observation, info = env.reset(seed=42, return_info=True)
print(f'observation={observation} info={info}')

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(f'action={action} '
          f'observation={observation} '
          f'reward={reward} '
          f'terminated={terminated} '
          f'info={info}')

    if terminated:
        observation, info = env.reset(return_info=True)
        print(f'observation={observation} info={info}')

env.close()
