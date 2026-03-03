
import numpy as np
import networkx as nx
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from scipy import linalg as la
from collections import Counter
from scipy.interpolate import griddata
import matplotlib.colors as mcolors
import time as tm
import pandas as pd
from scipy.optimize import minimize
import pickle
import seaborn as sns
#a function to take a m-vector with n<m non-zero probabilities as elements and output an index as in metropolis montecarlo:
def choose(vector):
    m  = len(vector)
    indices = np.where(vector != 0)[0]
    if len(indices) != 0:
        elts = vector[vector != 0]
        cumsums = np.cumsum(elts) 
        t = np.random.random()
        k = indices[0]
        if t < cumsums[0]:
            return k
        else:
            for i in range(len(elts)-1):
                if cumsums[i] <= t < cumsums[i+1]:
                    k = indices[i+1]
        return k
    else:
        return np.random.choice(len(vector))
    
def choose_thresholded(vector):
    m  = len(vector)
    indices = np.where(vector != 0)[0]
    if len(indices) != 0:
        elts = vector[vector != 0]
        cumsums = np.cumsum(elts) 
        t = np.random.random()
        k = indices[0]
        if t < cumsums[0]:
            return k
        else:
            for i in range(len(elts)-1):
                if cumsums[i] <= t < cumsums[i+1]:
                    k = indices[i+1]
        return k
    else:
        # else return None
        return None
    
def nextContext(currentContext, Xi):
    context_vec = encoded_contexts[currentContext]
    next_context_vec = np.dot(Xi, context_vec)
    next_context = np.argmax(next_context_vec)
    if next_context == currentContext:
        new_vec = np.delete(next_context_vec, currentContext,0)
        ind = choose(new_vec)
        return np.where(next_context_vec == new_vec[ind])[0][0]
    else:
        return next_context

def is_reward_in_path(mat,state,states):
    state_dict = {state: i for i, state in enumerate(states)}
    state_number = state_dict[state]
    state_numbers = np.array([states.index(s) for s in states])
    onehot = OneHotEncoder(sparse=False)
    encoded_states = onehot.fit_transform(np.array(state_numbers).reshape(len(states),1))
    state_vec = encoded_states[state_number]
    next_vec = np.dot(state_vec,mat)
    next_next_vec = np.dot(next_vec,mat)
    if next_next_vec[4] == 1:
        bool = True
    else:
        bool = False
    # if bool and all the other indices in next_next_vec are 0, return True
    # if not, return False
    if bool and np.sum(next_next_vec) == 1:
        return bool, True
    elif bool and np.sum(next_next_vec) > 1:
        return bool, False 
    else:
        return False, False
    
# given a list of states - memory, a function to update it with the current one: add the current state to the end of memory and remove the first one
def updateMemory(memory, current_state):
    if len(memory) < 10:
        memory.append(current_state)
    else:
        memory.append(current_state)
        memory = memory[1:]
    return memory

def compare_td_error(list,threshold):
    errornorms = np.linalg.norm(list,axis=1)
    if np.linalg.norm(list[-1]) > threshold* np.mean(errornorms):
        return True
    else:
        return False

# give a next state according to the probabilities in the transition matrix
def give_next_state(state,transition_matrix,states):
    #one hot encode the states
    state_numbers = np.array([states.index(s) for s in states])
    onehot_encoder = OneHotEncoder(sparse_output=False)
    encoded_states = onehot_encoder.fit_transform(np.array(state_numbers).reshape(len(states),1))
    next_vec = transition_matrix.dot(encoded_states[states.index(state)])
    next_index = choose(next_vec)
    return states[next_index]

def give_next_state_thresholded(state,transition_matrix,states):
    #one hot encode the states
    onehot_encoder = OneHotEncoder(sparse_output=False)
    encoded_states = onehot_encoder.fit_transform(np.array(states).reshape(len(states),1))
    next_vec = transition_matrix.dot(encoded_states[states.index(state)])
    next_index = choose_thresholded(next_vec)
    if next_index == None:
        return None
    else:
        return states[next_index]

# given a list of context-SRs, and a current SR (thresholded) compare and find the best one
def findright(list,current):
    counts = np.zeros(len(list))
    for i in range(len(list)):
        counts[i] += (current == list[i]).sum()
    candidate = np.random.choice(np.where(counts==np.max(counts))[0])
    return list[candidate], candidate, counts

def normalize(matrix):
    for i in range(len(matrix)):
        indices = np.where(matrix[i,:] != 0)[0]
        if len(indices) != 0:
            matrix[i,indices] = matrix[i,indices]/np.sum(matrix[i,indices])
        else:
            matrix[i,:] = np.zeros(len(matrix[i,:]))
    return matrix

# a function to compute expectation of next state given current state and a transition matrix or causal graph
def expectation(current_state, next_state, transition, time_step, is_causal, time):
    if is_causal:
        time_constant = 0.1*time_step
        if time > time_step*1.01:
            return (1 - np.exp(-(time_step*1.1)/time_constant)) * transition[current_state, next_state] * np.exp(-(time - time_step*1)/time_constant)
        else:
            return (1 - np.exp(-time/time_constant)) * transition[current_state, next_state]
    else:
        time_constant = 0.9*time_step
        if time > time_step*1.01:
            return (1 - np.exp(-(time_step*1.1)/time_constant)) * transition[current_state, next_state] * np.exp(-(time - time_step*1)/time_constant)
        else:
            return (1 - np.exp(-time/time_constant)) * transition[current_state, next_state]


def choice_2step(states, thresholded_successor, actions):
    state_dict = {state: i for i, state in enumerate(states)}
    reward_in_left, only_reward_in_left = is_reward_in_path(thresholded_successor[:,:,0], 'left',states)
    reward_in_right, only_reward_in_right = is_reward_in_path(thresholded_successor[:,:,1], 'right',states)
    if reward_in_left and reward_in_right:
        if only_reward_in_left and not only_reward_in_right:
            next_state = 'left'
            action_index = 0
        elif only_reward_in_right and not only_reward_in_left:
            next_state = 'right'
            action_index = 1
        else:
            next_state = np.random.choice(['left', 'right'])
            action_index = actions.index(next_state)
    elif reward_in_left and not reward_in_right:
        next_state = 'left'
        action_index = 0
    elif reward_in_right and not reward_in_left:
        next_state = 'right'
        action_index = 1
    else:
        next_state = np.random.choice(['left', 'right'])
        action_index = actions.index(next_state)
    return next_state, action_index

# a function to give a choice probability 

def choiceprob_2step(action_index, epsilon):
    prob =  (1 - epsilon)*action_index + epsilon/2
    rand = np.random.random()
    if rand < prob:
        index = 1
    else:
        index = 0
    return prob, index

def inv_logit(x):
    return np.log(x/(1-x))

# function to compute the negative log-likelihood of the choices given the model and the data
def log_likelihood_choices(results,choices,free_choice):
    log_likelihood = 0
    choices_predicted = results[results['free_choice']==True]['choices']
    choices_actual = choices[np.where(free_choice == True)[0]]
    for i in range(len(choices)):
        choice = choices[i]
        choice_prob = results['choice_probs'][i]
        if choice == 1:
            log_likelihood += np.log(choice_prob)
        else:
            log_likelihood += np.log(1 - choice_prob)
    return  -1 * log_likelihood

def logit(x):
    return np.exp(x)/(1 + np.exp(x))


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def compute_gpe(thresh_successor,state_current,state_next,action,states):
    predicted_next = give_next_state_thresholded(state_current,thresh_successor[:,:,action],states)
    state_current_number = states.index(state_current)
    state_next_number = states.index(state_next)
    predicted_next_number = states.index(predicted_next) if predicted_next != None else None
    if predicted_next == state_next and predicted_next != None:
        gpe = 1
    else:
        gpe = thresh_successor[state_current_number,state_next_number,action] - 1
    return gpe

def transform_params(params = [0,0,0,0,0]):
    list = []
    for i in range(len(params)):
        if i != 3:
            list.append(logit(params[i]))
        else:
            list.append(np.exp(params[i]))
    return list

# function to import parameters from the data
def import_data(sample_session):
    free_choice = sample_session['free_choice']
    transitions = sample_session['transitions']
    choices = sample_session['choices']
    outcomes = sample_session['outcomes']
    second_steps = sample_session['second_steps']
    stims = sample_session['stim']
    n_episodes = len(choices)
    return free_choice, transitions, choices, outcomes, second_steps, stims, n_episodes

# function to create a new dataframe with only the rows where the free choice is 1, and the row after that one
def create_frame(results):
    frame = pd.DataFrame(columns=['choices', 'choice_probs', 'second_steps', 'outcomes', 'transitions', 'free_choice', 'stage', 'mov_average', 'stim', 'stim_type','block'])
    for i in range(len(results)-1):
        if results['free_choice'][i] == 1:
            if results['free_choice'][i+1] == 1:
                frame = pd.concat([frame, results.iloc[i:i+2]], axis = 0)
    frame = frame.reset_index(drop=True)
    return frame

def create_counts(frame,transition_states):
    if transition_states == 1:
        total_counts = {'CR': 0,  'RR': 0, 'CU': 0,'RU': 0}
        stay_probs_predicted = {'CR': 0,  'RR': 0,'CU': 0, 'RU': 0}
        stay_counts_actual = {'CR': 0,  'RR': 0,'CU': 0, 'RU': 0}
        for i in range(0,len(frame),2):
            if frame['transitions'][i] == 1 and frame['outcomes'][i] == 1:
                total_counts['CR'] += 1
                if frame['choices'][i] == frame['choices'][i+1]:
                    stay_counts_actual['CR'] += 1
                if frame['choices'][i] == 1:
                    stay_probs_predicted['CR'] += frame['choice_probs'][i+1]
                elif frame['choices'][i] == 0:
                    stay_probs_predicted['CR'] += 1 - frame['choice_probs'][i+1]
            elif frame['transitions'][i] == 1 and frame['outcomes'][i] == 0:
                total_counts['CU'] += 1
                if frame['choices'][i] == frame['choices'][i+1]:
                    stay_counts_actual['CU'] += 1
                if frame['choices'][i] == 1:
                    stay_probs_predicted['CU'] += frame['choice_probs'][i+1]
                elif frame['choices'][i] == 0:
                    stay_probs_predicted['CU'] += 1 - frame['choice_probs'][i+1]
            elif frame['transitions'][i] == 0 and frame['outcomes'][i] == 1:
                total_counts['RR'] += 1
                if frame['choices'][i] == frame['choices'][i+1]:
                    stay_counts_actual['RR'] += 1
                if frame['choices'][i] == 1:
                    stay_probs_predicted['RR'] += frame['choice_probs'][i+1]
                elif frame['choices'][i] == 0:
                    stay_probs_predicted['RR'] += 1 - frame['choice_probs'][i+1]
            elif frame['transitions'][i] == 0 and frame['outcomes'][i] == 0:
                total_counts['RU'] += 1
                if frame['choices'][i] == frame['choices'][i+1]:
                    stay_counts_actual['RU'] += 1
                if frame['choices'][i] == 1:
                    stay_probs_predicted['RU'] += frame['choice_probs'][i+1]
                elif frame['choices'][i] == 0:
                    stay_probs_predicted['RU'] += 1 - frame['choice_probs'][i+1]
        stay_probs_predicted_std = {'CR': 0,  'RR': 0,'CU': 0, 'RU': 0}
        stay_counts_actual_std = {'CR': 0,  'RR': 0,'CU': 0, 'RU': 0}
        for key in stay_probs_predicted:
            stay_probs_predicted[key] = stay_probs_predicted[key]/total_counts[key]
            stay_counts_actual[key] = stay_counts_actual[key]/total_counts[key]
            stay_probs_predicted_std[key] = np.sqrt(stay_probs_predicted[key]*(1-stay_probs_predicted[key]))/total_counts[key]
            stay_counts_actual_std[key] = np.sqrt(stay_counts_actual[key]*(1-stay_counts_actual[key]))/total_counts[key]
        return stay_probs_predicted, stay_counts_actual, stay_probs_predicted_std, stay_counts_actual_std
    elif transition_states == 0:
        total_counts = {'CR': 0,  'RR': 0,'CU': 0, 'RU': 0}
        stay_probs_predicted = {'CR': 0,  'RR': 0,'CU': 0, 'RU': 0}
        stay_counts_actual = {'CR': 0,  'RR': 0, 'CU': 0,'RU': 0}
        for i in range(0,len(frame),2):
            if frame['transitions'][i] == 0 and frame['outcomes'][i] == 1:
                total_counts['CR'] += 1
                if frame['choices'][i] == frame['choices'][i+1]:
                    stay_counts_actual['CR'] += 1
                if frame['choices'][i] == 1:
                    stay_probs_predicted['CR'] += frame['choice_probs'][i+1]
                elif frame['choices'][i] == 0:
                    stay_probs_predicted['CR'] += 1 - frame['choice_probs'][i+1]
            elif frame['transitions'][i] == 0 and frame['outcomes'][i] == 0:
                total_counts['CU'] += 1
                if frame['choices'][i] == frame['choices'][i+1]:
                    stay_counts_actual['CU'] += 1
                if frame['choices'][i] == 1:
                    stay_probs_predicted['CU'] += frame['choice_probs'][i+1]
                elif frame['choices'][i] == 0:
                    stay_probs_predicted['CU'] += 1 - frame['choice_probs'][i+1]
            elif frame['transitions'][i] == 1 and frame['outcomes'][i] == 1:
                total_counts['RR'] += 1
                if frame['choices'][i] == frame['choices'][i+1]:
                    stay_counts_actual['RR'] += 1
                if frame['choices'][i] == 1:
                    stay_probs_predicted['RR'] += frame['choice_probs'][i+1]
                elif frame['choices'][i] == 0:
                    stay_probs_predicted['RR'] += 1 - frame['choice_probs'][i+1]
            elif frame['transitions'][i] == 1 and frame['outcomes'][i] == 0:
                total_counts['RU'] += 1
                if frame['choices'][i] == frame['choices'][i+1]:
                    stay_counts_actual['RU'] += 1
                if frame['choices'][i] == 1:
                    stay_probs_predicted['RU'] += frame['choice_probs'][i+1]
                elif frame['choices'][i] == 0:
                    stay_probs_predicted['RU'] += 1 - frame['choice_probs'][i+1]
        stay_probs_predicted_std = {'CR': 0,  'RR': 0,'CU': 0,'RU': 0}
        stay_counts_actual_std = {'CR': 0,  'RR': 0,'CU': 0, 'RU': 0}
        for key in stay_probs_predicted:
            stay_probs_predicted[key] = stay_probs_predicted[key]/total_counts[key]
            stay_counts_actual[key] = stay_counts_actual[key]/total_counts[key]
            stay_probs_predicted_std[key] = np.sqrt(stay_probs_predicted[key]*(1-stay_probs_predicted[key])/total_counts[key])
            stay_counts_actual_std[key] = np.sqrt(stay_counts_actual[key]*(1-stay_counts_actual[key])/total_counts[key])
        return stay_probs_predicted, stay_counts_actual, stay_probs_predicted_std, stay_counts_actual_std
    
# function to plot the stay probabilities
def plot_stay_probs(stay_probs_predicted, stay_counts_actual, stay_probs_predicted_std, stay_counts_actual_std):
    stay_probs_predicted = pd.Series(stay_probs_predicted)
    stay_probs_predicted_std = pd.Series(stay_probs_predicted_std)
    stay_counts_actual = pd.Series(stay_counts_actual)
    stay_counts_actual_std = pd.Series(stay_counts_actual_std)
    plt.figure(figsize=(5,5))
    stay_probs_predicted.plot(kind = 'bar', yerr = stay_probs_predicted_std, position=0, width=0.25, label = 'predictions')
    stay_counts_actual.plot(kind = 'bar', yerr = stay_counts_actual_std,color = 'red',position=1,width=0.25, label = 'data')
    #remove the top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    #plt.axhline(y = 0.5, color = 'black', linestyle = ':')
    plt.title('Stay probabilities predicted vs actual')
    plt.legend()

    return stay_probs_predicted, stay_counts_actual
    

def plot_stay_probs_stim(stay_probs_predicted, stay_counts_actual, stay_probs_predicted_std, stay_counts_actual_std):
    stay_probs_predicted = pd.Series(stay_probs_predicted)
    stay_probs_predicted_std = pd.Series(stay_probs_predicted_std)
    stay_counts_actual = pd.Series(stay_counts_actual)
    stay_counts_actual_std = pd.Series(stay_counts_actual_std)
    plt.figure(figsize=(5,5))
    stay_probs_predicted.plot(kind = 'bar', yerr = stay_probs_predicted_std, position=0, width=0.25, label = 'stim')
    stay_counts_actual.plot(kind = 'bar', yerr = stay_counts_actual_std,color = 'red',position=1,width=0.25, label = 'nostim')
    #remove the top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    #plt.axhline(y = 0.5, color = 'black', linestyle = ':')
    plt.title('Stay probabilities stim vs nostim')
    plt.legend()
    
    

    return stay_probs_predicted, stay_counts_actual

# function to plot the choice probs
def plot_choice_probs(results, blocks):
    choice_probs = results['choice_probs']
    choices = results['choices']
    moving_average_choices = moving_average(choices,10)
    block_end_trials = blocks['end_trials']
    choices_new = []
    for i in range(len(choices)):
        if choices[i] == 1:
            choices_new.append(1)
        else:
            choices_new.append(-1)

    # plot the moving average of choice probabilities, and the moving average of choices_new times outcomes
    moving_average_choice_probs = moving_average(choice_probs, 10)
    moving_average_outcomes_side = moving_average(np.multiply(choices_new, outcomes),10)
    plt.figure(figsize=(20,10))
    plt.plot(2*moving_average_choice_probs - 1, label = 'choice probs')
    plt.plot(moving_average_outcomes_side, label = 'outcomes by side')
    #horizontal dashed lines at 0, 0.5, -0.5
    plt.axhline(y = 0, color = 'red', linestyle = ':')
    plt.axhline(y = 0.5, color = 'black', linestyle = ':')
    plt.axhline(y = -0.5, color = 'black', linestyle = ':')
    #vertical lines at the end of each block
    for i in range(len(block_end_trials)):
        plt.axvline(x = block_end_trials[i], color = 'black', linestyle = ':')
    plt.title('Choice probabilities, outcomes by side, and choices')
    plt.legend()
    return moving_average_choice_probs, moving_average_outcomes_side

def get_values_from_graph(reward_list, graph):
    # f_matrix is sum of graph and graph-squared and graph-cubed and so on, 4 powers
    f_matrix = np.zeros((len(reward_list),len(reward_list)))
    for i in range(4):
        f_matrix += np.linalg.matrix_power(graph,i+1)
        # for each non-zero element in f_matrix, set it to 1
    f_matrix[f_matrix != 0] = 1
    # for a given row i in f_matrix, value_list[i] is the sum of rewards of all elements in the row which are 1
    value_list = np.zeros(len(reward_list))
    for i in range(len(reward_list)):
        value_list[i] = np.dot(f_matrix[i,:],reward_list)
    return value_list

def get_values_from_graph(reward_list, graph):
    # f_matrix is sum of graph and graph-squared and graph-cubed and so on, 4 powers
    f_matrix = np.zeros((len(reward_list),len(reward_list)))
    for i in range(4):
        f_matrix += np.linalg.matrix_power(graph,i+1)
        # for each non-zero element in f_matrix, set it to 1
    f_matrix[f_matrix != 0] = 1
    # for a given row i in f_matrix, value_list[i] is the sum of rewards of all elements in the row which are 1
    value_list = np.zeros(len(reward_list))
    for i in range(len(reward_list)):
        value_list[i] = np.dot(f_matrix[i,:],reward_list)
    return value_list

def get_values_from_graph_2(reward_list, graph):
    # f_matrix is sum of graph and graph-squared and graph-cubed and so on, 4 powers
    f_matrix = np.zeros((len(reward_list),len(reward_list)))
    for i in range(4):
        f_matrix += np.linalg.matrix_power(graph,i+1)
        # for each non-zero element in f_matrix, set it to 1
    f_matrix[f_matrix != 0] = 1
    # for a given row i in f_matrix, value_list[i] is the sum of rewards of all elements in the row which are 1
    # add the identity matrix to f_matrix
    f_matrix += np.eye(len(f_matrix))
    value_list = np.zeros(len(reward_list))
    for i in range(len(reward_list)):
        value_list[i] = np.dot(f_matrix[i,:],reward_list)
    return value_list

def get_values_from_graph_3(reward_list, graph):
    # f_matrix is sum of graph and graph-squared and graph-cubed and so on, 4 powers
    f_matrix = np.zeros((len(reward_list),len(reward_list)))
    for i in range(4):
        f_matrix += np.linalg.matrix_power(graph,i+1)
        # for each non-zero element in f_matrix, set it to 1
    #f_matrix[f_matrix != 0] = 1
    # for a given row i in f_matrix, value_list[i] is the sum of rewards of all elements in the row which are 1
    # add the identity matrix to f_matrix
    f_matrix += np.eye(len(f_matrix))
    value_list = np.zeros(len(reward_list))
    for i in range(len(reward_list)):
        value_list[i] = np.dot(f_matrix[i,:],reward_list)
    return value_list


def get_values_from_graph_4(reward_list, graph):
    # f_matrix is sum of graph and graph-squared and graph-cubed and so on, 4 powers
    graph += np.eye(len(graph))
    value_list = np.zeros(len(reward_list))
    for i in range(len(value_list)):
        value_list[i] = np.dot(graph[i,:], reward_list)
    return value_list


def choice_function(T,values):
    value_left = values[0]
    value_right = values[1]
    p_left = 1/(1+np.exp((value_right-value_left)/T))
    return p_left

def choice_function_2(values,variance,T):
    value_left = values[0]
    value_right = values[1]
    # add noise to the values
    numerator = np.random.normal(value_right-value_left,variance)
    p_left = 1/(1+np.exp(numerator/T))
    return p_left

def compare_graphs(graph,theta):
    # subtract theta from graph
    diff_matrix = graph - theta
    changed = False
    # if graph is all zeros, replace it with theta
    #if np.all(graph == 0):
        #graph = theta
    # for each column in diff, if the column has a negative value, replace the column of graph with the corresponding column of theta
    # otherwise, keep the column of graph
    #else:
    for i in range(diff_matrix.shape[1]):
        if np.any(diff_matrix[i,:] < 0):
            graph[i,:] = theta[i,:]
            changed = True
    return graph, changed


    
states = ['right', 'left','up','down','reward','no_reward']
actions = ['right','left']
rewards = [0,0,0,0,1,0]


# a function to generate the stims and free_choice lists - the stim trials are 1s in the stim list, and the free choice trials are 1s in the free_choice list
def generate_stims_freeChoice(n_episodes):
    # start with an array of n_episodes/4 3s
    a = np.full(int(n_episodes/4), 3)
    # add 1 to a random number k of elements in the array, and subtract 1 from some other k elements
    k = np.random.randint(0, int(n_episodes/8))
    # add 1 to k elements
    adds = np.random.choice(int(n_episodes/4), k, replace = False)
    indices = np.arange(int(n_episodes/4))
    #remove the adds from the indices
    removes = np.setdiff1d(indices, adds)
    a[adds] += 1
    # k other indices
    subs = np.random.choice(removes, k, replace = False)
    a[subs] -= 1
    # take all the indices which are in subs out and make a list of indices
    indices = np.setdiff1d(indices, subs)
    # add 1 to k elements
    k = np.random.randint(0, int(len(indices)/2))
    adds = np.random.choice(indices, k, replace = False)
    a[adds] += 1
    # k other indices
    removes = np.setdiff1d(indices, adds)
    subs = np.random.choice(removes, k, replace = False)

    a[subs] -= 1

    # create the stim_type list
    stim_1s = np.ones(int(n_episodes/4))
    # for each element i in stim_1s, add a[i] zeros after it
    stims = []
    for pq in range(len(stim_1s)):
        stims.append(stim_1s[pq])
        for gf in range(a[pq]-1):
            stims.append(0)
    for rtu in range(len(stims)):
        stims[rtu] = int(stims[rtu])
    
    if len(stims) < n_episodes:
        stims = np.append(stims, np.zeros(n_episodes - len(stims)))
    #make a list, free_choice of 1000 zeros
    free_choice = np.zeros(n_episodes)
    # each trial after a stim trial is a free choice trial
    for hty in range(len(stims)):
        if stims[hty] == 1 and hty != n_episodes - 1:
            free_choice[hty+1] = 1

    # of the remaining 750 trials, choose 500 at random to be free choice trials
    remaining_indices = np.where(free_choice == 0)[0]
    free_choice_trials = np.random.choice(remaining_indices, int(n_episodes/2), replace = False)
    free_choice[free_choice_trials] = 1

    # test, see how many trials are 1 in free_choice
    len(np.where(free_choice == 1)[0])
    return stims, free_choice

# set the initial conditions of the SR to be the model-based one for all parameters
# check to see if we can reproduce stimulation effect
states = ['right','left','down','up']
sr_initial = np.zeros((len(states),len(states)))

sr_initial[states.index('right'),states.index('down')] = 1
sr_initial[states.index('left'),states.index('up')] = 1




# new graph compare function such that in graph only has one 1 in each row and column

def compare_graphs_nocon(graph, theta):
    graph_2 = graph.copy()
    for i in range(graph.shape[0]):
        for j in range(graph.shape[1]):
            if theta[i,j] == 1 and graph[i,j] == 0:
                graph_2[i,j] = 1
                #set the rest of the row and column to 0
                for k in range(graph.shape[0]):
                    #if k != j:
                        #graph_2[i,k] = 0
                    if k != i:
                        graph_2[k,j] = 0
    return graph_2



def plot_from_results_2(results, params, filename = ''):
    df = results
    params_df = params
    # Compute whether participant stayed (same choice as last trial)
    df['prev_choice'] = df['choices'].shift(1)
    df['stay'] = (df['choices'] == df['prev_choice']).astype(int)

    # Shift transition and outcome to align with current choice
    df['prev_trans'] = df['transitions'].shift(1)
    df['prev_reward'] = df['outcomes'].shift(1)

    # Drop first trial (no previous data)
    df_clean = df.dropna(subset=['prev_trans', 'prev_reward', 'stay'])

    # Map to condition labels
    def label_condition(row):
        if row['prev_trans'] == 1 and row['prev_reward'] == 1:
            return 'CR'
        elif row['prev_trans'] == 0 and row['prev_reward'] == 1:
            return 'RR'
        elif row['prev_trans'] == 1 and row['prev_reward'] == 0:
            return 'CU'
        else:
            return 'RU'

    df_clean['Condition'] = df_clean.apply(label_condition, axis=1)

    # print the number of trials in each condition
    condition_counts = df_clean['Condition'].value_counts()
    print(condition_counts)

    # Group by condition and compute mean and standard error
    summary = df_clean.groupby('Condition')['stay'].agg(['mean', 'sem']).reset_index()
    summary.columns = ['Condition', 'Stay_Probability', 'SE']

    # Reorder for plotting
    condition_order = ['CR', 'RR', 'CU', 'RU']
    summary['Condition'] = pd.Categorical(summary['Condition'], categories=condition_order, ordered=True)
    summary = summary.sort_values('Condition')
    yerr = summary['SE'].values

    # Plot
    sns.set_style("ticks")
    # use subplots and plot sns.barplot on one side, and to the right, add a box with the parameters
    fig, ax = plt.subplots(1,2, figsize=(10, 5))
    palette = {
        'CR': 'b',  # blue
        'CU': 'b',  # blue
        'RR': 'r',  # orange
        'RU': 'r'   # orange
    }
    ax[0] = sns.barplot(
        data=summary,
        errorbar="se",
        x=summary['Condition'],
        y=summary['Stay_Probability'],
        order=condition_order,
        palette=palette,
        edgecolor='white',
        ax=ax[0],
    )
    # add a box to the right of the plot with params, in ax[1]
    ax[1].axis('off')
    ax[1].add_patch(plt.Rectangle((0.5, 0.5), 0.2, 0.2, fill=True, color='white', alpha=0.5))
    for i in range(len(params_df)):
        ax[1].text(0.5, 0.5 - i * 0.1, params_df.iloc[i]['parameter'] + ': ' + str(params_df.iloc[i]['value']), fontsize=10, ha='center', va='center', transform=ax[1].transAxes)
    # Add error bars
    ax[0].errorbar(summary['Condition'], summary['Stay_Probability'], yerr=yerr, fmt='none', capsize=5, color='black')
    # show the plot

    # Labels
    ax[0].set_title('Human Stay Probabilities by Condition', fontsize=16)
    ax[0].set_ylabel('Stay Probability', fontsize=14)
    ax[0].set_xlabel('')
    ax[0].set_ylim([0.5,1])
    plt.tight_layout()


    # save figure
    if filename != '':
        plt.savefig(filename, format='svg', dpi=300, bbox_inches='tight')
    plt.show()
    
    

    

    return summary

# generate a line here to import all functions from this file
# from helper-functions-SR import *