import gym
from RL_ import DeepQNetwork
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


env = gym.make('MountainCar-v0')
env = env.unwrapped


RL = DeepQNetwork(n_action =3, #env.action_space.n#,
                  n_features = 2,#env.observation_space.shape[0],
                  e_greedy_increment=0.0002,Double_DQN = False)

total_steps = 0

done_n = 0

for i_episode in range(1000):

    observation = env.reset()
    while True:
        env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        position, velocity = observation_

        reward = abs(position + 0.5)

        RL.store_transition(observation, action, reward, observation_)

        if done:
            done_n+=1
            print("\nsuccess\n"+str(done_n))
            break

        if total_steps == 1000:
            print("Start learning")

        if total_steps > 1000:
            RL.learn()

        observation = observation_
        total_steps += 1
