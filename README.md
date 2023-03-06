# Deep Q-Network Cart-Pole (Tensorflow implementation)

In this example, I'll be sharing an implementation of the Deep Q-Network with tensorflow. With this algorithm, we try to approximate the Q-Table(from the Q-Learning algorithm) with a neural network.

<p align="center">
   <img width="600" height="300" src="https://user-images.githubusercontent.com/52584370/221706505-7bc6a9d5-fcce-4c40-b178-059d390061fc.png">
</p>

The base idea, is to obtain the actual state of the environment, then with the `epsilon-greedy` algorithm choose an action. Then, do a step with the action, get new information from the environment(new_observation, reward, and done), and update the network.

<p align="center">
   <img width="600" height="300" src="https://user-images.githubusercontent.com/52584370/221709345-54cdb5ac-7731-4a0d-9f0c-b1dcd1faaa99.png">
</p>


# Implementation

<table border="0" align = "center">
 <tr>
    <td><b style="font-size:30px">No-Trained</b></td>
    <td><b style="font-size:30px">Trained</b></td>
 </tr>
 <tr>
    <td>
      <p align="center">
         <img width="300" height="200" src="https://user-images.githubusercontent.com/52584370/221698390-a0c7a96f-a45b-49aa-98b3-c7dcd7589b8a.gif">
      </p>
    </td>
    <td>
       <p align="center">
         <img width="300" height="200" src="https://user-images.githubusercontent.com/52584370/223244620-09f0d8ef-cafd-4569-8b1e-923cdc38c585.gif">
      </p>
    </td>
 </tr>
</table>


First, we need to define our environment, in this case it is the Cart-pole simulation, and obtain the number of obserbations and the number of possible actions.

```python
  self.env = gym.make("CartPole-v1", render_mode="rgb_array")
  self.Num_actions = self.env.action_space.n
  self.Num_observations = self.env.observation_space.shape[0]
```


In this case, we are using a DNN with 3 layers (this can depend on the project) which will represent de `Q_network`:
```python
  def _create_model(self, verbose = False):
    model = Sequential([
      Dense(24,input_shape = (1,self.Num_observations), activation = 'relu', kernel_initializer = tf.keras.initializers.HeNormal()),
      Dense(12, activation = 'relu', kernel_initializer = tf.keras.initializers.HeNormal()),
      Dense(self.Num_actions, activation = 'linear', kernel_initializer = tf.keras.initializers.HeNormal())
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss = 'mse', metrics=['accuracy'])

    if verbose:
      print(model.summary())
    return model
  
```

We have 2 important things in the previous code `input_shape = (1,self.Num_observations)` which represents the output of the environment(in this case the position and velocity of the cart, and the angle and angular velocity of the pole) and  `self.Num_actions` which are the possible number of actions that our agent can take(in this case 2 possible actions: push left and push right)
This information comes directly from the environment previously created:
```python
  self.Num_actions = self.env.action_space.n
  self.Num_observations = self.env.observation_space.shape[0]
 ```
 
 And we will have another neural network called `Target_network` which will have the same weigths as the `Q_network` in the beggining:
 ```python
  self.Target_network = self._create_model()
  self._set_Target_weights()
 ```
 
 Also, we need to create a memory, which will save all the transitions given by the environment:
 ```python
  self.memory = deque(maxlen=MAX_MEM_LEN)
 ```
 
Following the algorithm, first we need to Initialize the first state:

```python
  state, _  = self.env.reset()
  state = np.array([[state]])
 ```
Then, inside of a while loop we will do the rest of the algorithm.
First we will select an action based on the `epsilon-greedy` algorithm. This aproach, helps us to have an exploration and exploitation timing.
At the begining,  `EPSILON` is equal to 1, and in each step it will decrease until a minimum, helping us to have exploration at first, and further in time exploitation.

```python
  def _get_action(self, state: np.array):
    """Select an action based on epsilon-greedy exploration strategy"""
    if self.Epsilon > random.random():
      return self.env.action_space.sample()
    return np.argmax(self.Q_network.predict(state, verbose = 0))
```

Now, given an action, we need to perform that action in the environment:
```python
  while True:
    #With probability \epsilon select random action 
    action = self._get_action(state)

    #Execute action in emulator and observe reward and next_state
    next_state, reward, done, _, _ =self.env.step(action)
    next_state = np.array([[next_state]])
```

The next step, as we are using a replay memory, is to store that transition. In this case, I delete randomly from the memory (but this came from me, I don't know if it works better or worse this way :alien:)
```python
  #Store transition in memory
  if len(self.memory) < MAX_MEM_LEN:
    self.memory.append([state,action, reward, done, next_state])
  else:
    #Randomly put new data in memory
    erase = random.randint(0, MAX_MEM_LEN - 1)
    self.memory[erase] = [state,action, reward, done, next_state]
```

Now, we enter into the `Experience replay` part, where our agent will learn from past experiences.
```python
def Experience_replay(self):
  if len(self.memory) >= MIN_OBSERVATIONS:

    states, actions, rewards, dones, next_states = zip(*random.sample(self.memory, BATCH_LEN))

    states = tf.reshape(tf.convert_to_tensor(states), [BATCH_LEN,1, self.Num_observations])
    next_states = tf.reshape(tf.convert_to_tensor(next_states), [BATCH_LEN,1, self.Num_observations])

    Q_values = self.Q_network.predict( states , verbose = 0 , batch_size = BATCH_LEN)
    next_Q_values = self.Target_network.predict( next_states , verbose = 0, batch_size = BATCH_LEN)

    for i in range(BATCH_LEN):
      Q_values[i][0][ actions[i] ] = rewards[i] + GAMMA * np.amax( next_Q_values[i] ) if not dones[i] else rewards[i]
            
    self.Q_network.fit(states, Q_values, batch_size=(BATCH_LEN), epochs = 1, verbose = 0)

    del states, actions, rewards, dones, next_states, Q_values, next_Q_values
```

Explaining the previous function step by step, first we need to sample a random minibatch of transitions from the memory:
```python
states, actions, rewards, dones, next_states = zip(*random.sample(self.memory, BATCH_LEN))
```

Then, we need to obtain the `Target` which will help us in the gradient descent:
<p align="center">
   <img width="500" height="70" src="https://user-images.githubusercontent.com/52584370/221712487-827e5215-b84b-4a19-ad7f-6a36d23bbf7c.png">
</p>

From the equation $y_i$ represents the Target from the gradient descent step part. First, we will predict with our Q_network the samples from the minibatch, and we will modify that output based on the case statement. And at the end, that modified output, will help us with the Loss function:
```python
 states = tf.reshape(tf.convert_to_tensor(states), [BATCH_LEN,1, self.Num_observations])
 next_states = tf.reshape(tf.convert_to_tensor(next_states), [BATCH_LEN,1, self.Num_observations])

 Q_values = self.Q_network.predict( states , verbose = 0 , batch_size = BATCH_LEN)
 for i in range(BATCH_LEN):
    Q_values[i][0][ actions[i] ] = rewards[i] + GAMMA * np.amax( next_Q_values[i] ) if not dones[i] else rewards[i]
```

And now, we permorm a gradient descent step:
```python
 self.Q_network.fit(states, Q_values, batch_size=(BATCH_LEN), epochs = 1, verbose = 0)
```

# Results
<p align="center">
   <img width="800" height="300" src="https://user-images.githubusercontent.com/52584370/223244767-83bfb8b0-8494-4626-8ea1-a01577356e5f.png">
</p>

<p align="center">
   <img width="400" height="300" src="https://user-images.githubusercontent.com/52584370/223244620-09f0d8ef-cafd-4569-8b1e-923cdc38c585.gif">
</p>

![2b916fd4-0cc1-4940-9425-e109ec4f358f](g)




