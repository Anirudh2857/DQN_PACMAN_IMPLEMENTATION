# Deep Q-Network (DQN) Implementation

This repository contains an implementation of a Deep Q-Network (DQN) for training an agent to play Atari's Ms. Pac-Man. The implementation utilizes PyTorch and Gymnasium to build and train the model.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Installation

Ensure you have Python 3.7+ installed. Then, install the required dependencies:

```sh
pip install gymnasium
pip install "gymnasium[atari, accept-rom-license]"
pip install ale-py
apt-get install -y swig
pip install gymnasium[box2d]
pip install torch torchvision numpy imageio
```

## Project Structure

```
|-- dqn.py        # Contains the neural network, agent class, and training loop
|-- README.md     # This documentation file
```

## Model Architecture

The DQN model consists of:

- Four convolutional layers with batch normalization
- Two fully connected layers
- An output layer for action-value predictions

## Training Process

1. The agent interacts with the `MsPacmanDeterministic-v0` environment.
2. It preprocesses frames and stores experiences in a replay buffer.
3. The model is trained using minibatches of past experiences.
4. The training loop runs until the environment is solved (average score ≥ 500 over 100 episodes).
5. The trained model is saved as `checkpoint.pth`.

## Usage

To train and test the model, run:

```sh
python dqn.py
```

## Results

The training process outputs the average score per 100 episodes. The model is saved once it solves the environment.

To visualize the trained model's performance, run:

```sh
python dqn.py
```

This generates a video of the agent playing the game.

## License

This project is open-source and available under the MIT License.

