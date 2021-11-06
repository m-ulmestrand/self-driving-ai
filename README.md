# Self driving car with Deep Q-learning
This project explores making a model of a car self-driving. The model is controlled by a neural network which is trained with Q-learning. The environment for making custom tracks as well as collision detection is also built from scratch. Below is an example of the car driving on its own. The green lines are an illustration of how the car measures distance, which is used as features along with a couple other parameters.



https://user-images.githubusercontent.com/54723095/139694716-0a3dcdaa-ae4e-4c1c-bc35-b74df4faec78.mp4



Example of a neural network controlling the car model, trained for around 5 minutes on an NVIDIA GeForce RTX 3050 laptop GPU. That's right, just a few minutes, on a laptop!




https://user-images.githubusercontent.com/54723095/139856396-8dd7224a-b57e-46e2-b873-5fa3d7892108.mp4

The neural network applied to previously unseen test tracks.


https://user-images.githubusercontent.com/54723095/139884525-9245135d-0381-4a14-a78f-5e6c9d83f8ec.mp4

We can even use several cars. At one point, a car overtakes another one.


## Neural network
At the time being, the brain of the car is a simple fully connected neural network.

### Features
The neural network accepts seven features:
1. Far-left distance
2. Left-front distance
3. Front distance
4. Right-front distance
5. Far-right distance
6. Angle of wheels
7. Speed

All of the features are normalised to an absolute value between 0 and 1. I try to make this normalisation a habit since it makes sense to me to have an interval where the weights of the neural network are likely to be initialised in the vicinity of. It may also generalise better, since I normalise by dividing with physical measures and thus make the parameters nondimensional. As such, it does not matter if the measures of the car, track and speed were to be scaled up by some factor, the network still sees the same input. 

### Target network
Deep Q-learning can be very unstable. One way to stabilise the learning procedure is to introduce a target network. The target network is synchronised with the prediction network periodically, and is thus kept constant for prolonged periods. The target network is used to estimate the future Q-values, while the prediction network estimates the current Q-values.

### Output
The network outputs Q-values for:
1. Increasing the angle of the wheels relative to the car
2. Decreasing the angle of the wheels relative to the car
3. Increasing the speed of the car
4. Decreasing the speed of the car

By changing the angle of the wheels, the turning radius is changed as well. There is a smallest allowed turning radius, which decides the angular velocity of the car. The angular velocity is found as the quotient between the speed of the car and the signed turning radius. The turning radius is in turn found as the cotangent of the angle of the wheels, multiplied by the minimum turning radius.

## Dependencies
The neural network is designed with PyTorch (1.8.1+cu102). In addition, NumPy (1.18.5) is used for many operations and information storing. To remove a lot of computational burden, I have used Numba (0.51.2) for several movement handling operations, collision detections, as well as distance measuring etc. For parsing command line arguments, argparse (1.4.0) is used.

# Tutorial for running the scripts
First of, make sure you have all of the necessary packages. Have a virtual environment set up. For example, you can then either use either `pip install numpy` or `conda install numpy` if you're using Conda (I stick to pip). For installing PyTorch, see https://pytorch.org/. The other packages are available on pip (I don't know about conda).
## Drawing tracks



https://user-images.githubusercontent.com/54723095/140609814-b18d80fc-9093-485a-8c1c-8298f616aa3c.mp4

I've stuck to a pretty simple approach of defining the tracks. When you run `python draw_track.py`, you can start drawing a track by holding the letter `A` and moving the mouse to define a set of nodes. When you are close enough to the original node again, the track will form. Defining the borders is not particularly easy to get right, since one border needs to be shorter than the other in a turn. I have a provisional solution for this, but it is not perfect. Making too sharp turns may lead to yanks in the track. Sufficiently smooth curves will lead to a good track.

## Training a neural network
To train a neural network, use the script `train.py`. There are many parameters that can be changed, but I find that the ones defined already in the script work well. To see a short explanation of the parameters, visit the script `racing_agent.py`. `epsilon` controls the initial randomness of the Q-learning algorithm, and thus promotes exploration. It is kept between 1.0 and 0.0. Using `epsilon_start = 1.0`, `epsilon_final = 0.0` and `epsilon_scale = runs` usually works well. These settings mean that `epsilon` linearly decreases from 1.0 to 0.0 from the first generation to the last.

To train a new agent, specify an agent name, such as `agent_dense2`, and run the script with the command `python train.py`. This will train and save an agent in the folder `build`.

Currently, two neural network architectures are implemented: `DenseNetwork` and `RecurrentNetwork`. For `RecurrentNetwork`, you can specify a sequence length for a series of features which the neural network will use as input. However, I find that `DenseNetwork` works the best for this application. By default, the sequence length `seq_length` is kept as 1, which it should be for `DenseNetwork`. To change the number of hidden neurons, you can change the setting `hidden_neurons` in the instance of `RacingAgent`. Relatively few neurons work well, such as `[64, 32, 16]`. This means that hidden layer 1 has 64 neurons, hidden layer 2 has 32 neurons and hidden layer 3 has 16 neurons. The pretrained agent `agent_dense` has `hidden_neurons = [32, 32, 32]`. 

## Testing your trained agent

