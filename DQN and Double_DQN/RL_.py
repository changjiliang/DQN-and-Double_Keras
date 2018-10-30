import numpy as np
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K

import random

class DeepQNetwork:
    def __init__(self,n_action, n_features,e_greedy_increment=None,memory_size = 500, Double_DQN = None):
        self.n_actioin = n_action
        self.n_features = n_features
        self.lr = 0.01
        self.gamma = 0.9
        self.epsilon_max = 0.9
        self.replace_target_iter = 300
        #self.memory = deque(maxlen=2000)
        self.memory_size = memory_size
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        self.batch_size = 32
        self.epsilon = 0.9
        self.model_eval = self._build_model()
        self.model_target = self._build_model()
        self.learn_step_counter = 0
        self.batch_size = 32
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.Double_DQN = Double_DQN


    def _loss(self,y_true, y_pred):
        loss = tf.reduce_mean(tf.squared_difference(y_true, y_pred))
        return loss

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(32, input_dim=self.n_features, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.n_actioin, activation='linear'))
        model.compile(loss=self._loss,
                      optimizer=Adam(lr=self.lr))
        return model

    def update_target_model(self):
        self.model_target.set_weights(self.model_eval.get_weights())
    '''
    def store_transition(self, s, a, r, s_):
        self.memory_counter = 0
        self.memory.append((s, a, r, s_))
        self.memory_counter += 1
    '''
    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0


        transition = np.hstack((s, [a, r], s_))


        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            action_value = self.model_eval.predict(observation)
            action = np.argmax(action_value)
        else:
            action = np.random.randint(0, self.n_actioin)
        return action
    '''
    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.update_target_model()
            #print("\nparams have been updated\n")

        if self.memory_counter > self.batch_size:
            minibatch = random.sample(self.memory, self.batch_size)
            for s, a, r, s_ in minibatch:
                q_eval = self.model_eval.predict(s)
                q_next = self.model_target.predict(s_)
                q_target = q_eval.copy()
                q_target[0][a] = r + self.gamma * np.amax(q_next[0])
                self.model_eval.fit(s, q_target, epochs=1, verbose=0)
            self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
            self.learn_step_counter += 1
    '''
    def learn(self):
        if self.learn_step_counter == 0:
            self.DD = True
        else:
            self.DD = False
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.update_target_model()  #更新参数
            #print('\ntarget_params_replaced\n')
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        ########Double DQN########
        if self.Double_DQN:
            if self.DD:
                print('I am using Double_DQN')
            q_next = self.model_target.predict(batch_memory[:, -self.n_features:])
            q_eval_next = self.model_eval.predict(batch_memory[:, -self.n_features:])
            q_eval = self.model_eval.predict(batch_memory[:, :self.n_features])

            q_target = q_eval.copy()

            batch_index = np.arange(self.batch_size, dtype=np.int32)
            eval_act_index = batch_memory[:, self.n_features].astype(int)
            reward = batch_memory[:, self.n_features + 1]

            max_action_next = np.argmax(q_eval_next, axis = 1)
            selected_q_next = q_next[batch_index, max_action_next]

            q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next
        ##下一动作是由估计网络得出的
        ####################
        #########DQN###########
        else:
            if self.DD:
                print('I am using DQN')
            q_eval = self.model_eval.predict(batch_memory[:, :self.n_features])
            q_next = self.model_target.predict(batch_memory[:, -self.n_features:])

            q_target = q_eval.copy()

            batch_index = np.arange(self.batch_size, dtype=np.int32)
            eval_act_index = batch_memory[:, self.n_features].astype(int)
            reward = batch_memory[:, self.n_features + 1]

            q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)
            ##执行当前动作的奖励值 = 估计网络的估计奖励值 + 折扣系数 * （现实网络得出的执行下一动作的奖励值）
            ##下一动作是由现实网络得出的
        ##########################
        self.model_eval.fit(batch_memory[:, :self.n_features], q_target, epochs=1, verbose=0)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
