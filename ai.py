# AI for Self Driving Car


# we use AI for deciding which action will our Agent take so that it can drive itsell

# Importing the libraries
 
import numpy as np          			#allows us to work and play with the arrays
import random               			#we will be taking some random samples
import os                   			#useful for loading the saved model or brain as you say
import torch                			#neural network is handled using pytorch as it can handle dynamic graphs
import torch.nn as nn       			# .nn is important module in pytorch
import torch.nn.functional as F         #different functions which we will be using.     
import torch.optim as optim             #for optimizer to perforam stocastic gradient descent 
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the architecture of the Neural Network

class Network(nn.Module): #Network is child class of parent class(nn.Module) it will inherit all its properties.

    #input_size=number of input neurons (There are 5 inputs here) (3 sensors and 2 orientation to keep track of goal) 
    #nb_action= possibilties of action or outputs( 3 possible , left right or straight)
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()     				#trick to use all functions of nn Module
        self.input_size = input_size         
        self.nb_action = nb_action
        nb_hiddenLayer = 30					
        self.fc1 = nn.Linear(input_size, nb_hiddenLayer)    # will connect the input layer w/ hidden layer    
        self.fc2 = nn.Linear(nb_hiddenLayer, nb_action) 	# will connect the hidden layer w/ the output layer
    
    
    
    # Forward propagation function ( Will Activate the neurons and it will return q values depending on the state)
    def forward(self, state):                     #self is written to use the variables of object of Class Network
        x = F.relu(self.fc1(state))               #x represent activated hidden neurons; relu is rectifier function; We are activating hidden neurons.                 
        q_values = self.fc2(x)                    #q_values= output neurons (These are not action) These are output values of neural network
        return q_values

# Implementing Experience Replay

#storing the experience in a array ; last 100 transactions.
#memory will be containing last 100 transaction so it is a simple list
#push function will be used to put values (event) into the memory list and it should not exceed the capacity.
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]            #if memory becomes more , just delete the first element (the oldest experience value).
    
    def sample(self, batch_size):                    #to get random samples of particular size from memory;
        samples = zip(*random.sample(self.memory, batch_size))   #getting random samples of size batch_size and zip
        # returns the batches (action, reward, state)
        return map(lambda x: Variable(torch.cat(x, 0)), samples)        #we cannot return samples directly.
        
        #we will list of batches aligned properly 

# Implementing Deep Q Learning

class Dqn():

    def __init__(self, input_size, nb_action, gamma):                             #gamma parameter is delay coefficient      
        self.gamma = gamma                                                                #initialising all the variables
        self.learning_rate = 0.001 
        self.reward_window = []                                                   #last 100 reward and we want the mean of it should be increasing
        
        #creating neural network for dql model , we call it model:
        
        self.model = Network(input_size, nb_action)                                   
        self.memory = ReplayMemory(100000)                                        #Object of ReplayMemory; Capactiy=100000;
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)             #Adam is optimizer class
                                                                    #giving AI small learning_rate , it will learn slowly from it mistakes.
                                                                    #if i give larger value , it will learn quickly.
        
        '''
         Transition states (last_state, last_action, last_reward)
         last_state = vector of 5 dimensions(3 signals[straight, left, right],
                        orientation, -orientation ), which is the input_size
         in pytorch    it needs to be a Tensor rather than a vector
         .unsqueeze(0) is creating a fake dimension corresponding to the batch
        '''
        
        self.last_state = torch.Tensor(input_size).unsqueeze(0)             #converting vector into Tensor. Fake dimension is the first dimension of last state.         
        
        # action => 0 = go straight, 1 = go right, 2 = go left         (Angles are , 0,-20,20)
        self.last_action = 0
        self.last_reward = 0
    
    
    #Now we need to decide teach AI how to choose right action ;
    '''
        Select Action function
        Here we need to find out the best action to play in order to get the 
        highest score while still exploring other actions, so we'll use SoftMax
        So we need to generate a distribution of probabilities for each of the
        Q-values. Since we have 3 possible actions, we'll have 3 Q-values. 
        The sum of these 3 Q-values is equal to 1 (100%).
        
        SoftMax will atribute the large probability to the highest Q-value
        
        We can configure the temperature parameter to set how will the algorithm
        explore the possible actions.
    '''
    def select_action(self, state):
        temperature_param = 100
        probs = F.softmax(self.model(Variable(state, volatile = True))*temperature_param)    #probabilities. (converting state tensor to torch variable.)
        #volatile true means not including gradient dscent.
        #more temparatue param , more sure that we should take that action; making probabilties higher.
        
        # Now we take a random draw of the distribution        #selecting random action from the probabilities.
        action = probs.multinomial(1)
        return action.data[0,0]
    
    '''
        The Learn function to train the Neural Network
        This method is the transition of the Markov decision process that is the base of 
        Deep Q-Learning
        This method will get these batches from the ReplayMemory
        We need the batch_state and the batch_next_state to compute the loss function
    '''
    
    #training dqn;  forward propa & back propa;  updating wts using stocastic gradient descent.
    #we are giving in one transaction of markov decision process.
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        '''
         We want to get the neuron output of the input state
         self.model(batch_state) returns the output of all the possible actions so
         we need to get only the one that was decided by the network to play, 
         so we use the gather function
        
         outputs = what our neural network predicts
        '''
        #killing fake dimension with squeeze.
        
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        
        # target is our goal, we want that outputs be equals to target
        target = self.gamma*next_outputs + batch_reward
        
        '''
         computing the loss (error of the prediction), calculates the difference between
         our prediction and the target
         
         td = temporal difference
         smooth_l1_loss is one of the best loss function in deep learning
        '''
        td_loss = F.smooth_l1_loss(outputs, target)
        
        # Back propagation to calculate the stochastic gradient descent updating the weights
        self.optimizer.zero_grad()                                   #zero_grad will reinitialize optimizer after transaction
        td_loss.backward(retain_graph = True)
        self.optimizer.step()     #uses optimizer to update the weight.
    
    """"
    Update function - updates all the elements in the transisition as soon as the AI
    reaches a new state (action, state and reward)
    
    This method is called when the car moves (enter in sand, change direction, hits the target)
    and updates the state
    This method makes the connection between the AI and the environment
    """  
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, 
                          new_state, 
                           torch.LongTensor([int(self.last_action)]), 
                          torch.Tensor([self.last_reward])))
        
        # Performs the action (take  random samples on memory)
        action = self.select_action(new_state)  #play new action after reaching new state.
        
        # Now we need to train and AI learn from the information 
        # contained in this samples of 100 random transitions
        nb_transitions = 100
        if len(self.memory.memory) > nb_transitions:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(nb_transitions)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
         # Reward_window has a fixed size
        reward_window_size = 1000
        if len(self.reward_window) > reward_window_size:
            del self.reward_window[0]
        return action
    
    # Calculates the mean of our rewards (Taking sum of all the rewards and dividing by number of rewards) 
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)       #adding plus 1 so that it is never zero.
    
    # Saving our neural network and the optimizer into a file to be able to use later  (# as we only need last weights and optimizer)
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),                    #save function from torch module
                    'optimizer' : self.optimizer.state_dict(),               #.state_dict() is used to save parameters of particular model and optimizer
                   }, 'last_brain.pth')
    
    def load(self):
        # look for the file
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            # We update our existing model/optimizer to the file that is being loaded
            self.model.load_state_dict(checkpoint['state_dict'])               #load_state_dict method is used.
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")

#If we dont want to activate the AI , just make temparature parameter 0.            
#Zip function
# if list=[(1,2,3),(4,5,6)] Then zip(*list)=   [(1,2),(3,4),(5,6)]         