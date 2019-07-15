# Udacity-Navigation
Udacity Deep Reinforcement Learning Nanodegree - Navigation project 

## Project details
In this project an agent is trained to navigate and collect bananas.
The state space has 37 dimensions (containing the agent's velocity, along with ray-based perception of objects around the agent's forward direction).
The action space has 4 dimensions : move forward, move backward, turn right and turn left.
The environment is considered solved when an average score of +13 over 100 consecutive episodes is obtained.
 
## Getting started
1. To set up your Python environment correctly follow [this link](https://github.com/udacity/deep-reinforcement-learning#dependencies).

2. To download the Unity Environment follow:
	- [This link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip) for Linux
	- [This link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip) for OSX
	- [This link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip) for Microsoft 32 bits
	- [This link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip) for Microsoft 64 bits
(For Windows users) Check out this link if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

3. Then, place the file in the p1_navigation/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.

4. To run this project you will need the following Python libraries:
	- numpy (tested with 1.16.4)
	- torch (tested with 1.1.0.post2)
	- matplotlib (tested with 3.1.0)


## Instructions
To run the code, follow the *Getting started* instructions, git clone this repository and go to the folder repository. Then just type:
- python navigation.py [arg1] [arg2]
Where:
- arg1 is the path to the Unity environment (Banana.app for instance on OSX)
	- arg1 is 1 for training and 0 for testing