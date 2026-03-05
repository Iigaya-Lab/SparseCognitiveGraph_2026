import numpy as np
from numba import njit
import torch


@njit
def get_values_from_graph_2_numba(reward_list, graph):
    n = len(reward_list)
    f_matrix = np.zeros((n, n))
    for i in range(4):
        # Numba-compatible matrix power and sum
        g_power = np.eye(n)
        for _ in range(i+1):
            g_power = g_power @ graph
        f_matrix += g_power
    for i in range(n):
        for j in range(n):
            if f_matrix[i, j] != 0:
                f_matrix[i, j] = 1
    for i in range(n):
        f_matrix[i, i] += 1  # add identity
    value_list = np.zeros(n)
    for i in range(n):
        for j in range(n):
            value_list[i] += f_matrix[i, j] * reward_list[j]
    return value_list



@njit
def likelihood_term_nostim_pureSR(x, free_choice, transitions, choices, outcomes, second_steps, stims, n_episodes):
    # x: [lr_base, lr_rew, discount, threshold, temperature, reward_learning_rate, staybias]
    states = ['right','left','up','down','up_reward','down_reward','no_reward']
    learning_rate = 1/(1 + np.exp(-x[0]))
    discount = 1/(1 + np.exp(-x[2]))
    reward_learning_rate = 1/(1 + np.exp(-x[5]))
    staybias = 1/(1 + np.exp(-x[6]))
    temperature = np.exp(x[4])
    if temperature < 1e-6:
        temperature = 1e-6
    given_rewards = np.array([0,0,0,0,1,1,0], dtype=np.float64)
    successor_representation = np.zeros((len(states),len(states)), dtype=np.float64)
    learned_rewards = np.zeros(len(states), dtype=np.float64)
    choices_list = np.empty(n_episodes, dtype=np.int32)
    choice_probs_list = np.empty(n_episodes, dtype=np.float64)
    free_choice_list = np.empty(n_episodes, dtype=np.bool_)
    for i in range(n_episodes):
        free_choice_trial = free_choice[i]
        choice = choices[i]
        outcome = outcomes[i]
        transition = transitions[i]
        second_step = second_steps[i]
        current_state = 0 if choice == 0 else 1  # 0: right, 1: left
        # Use pure SR for value computation (no thresholding)
        values = np.dot(successor_representation, learned_rewards)
        value_left = values[1]
        value_right = values[0]
        vlist = np.array([value_left, value_right])
        p_left = 1/(1+np.exp((value_right-value_left)/temperature))
        p_left = min(max(p_left, 1e-6), 1-1e-6)
        choice_prob = p_left
        if i > 1 and choices[i-1] == 0:
            choice_prob = max(choice_prob - staybias, 1e-6)
        if i > 1 and choices[i-1] == 1:
            choice_prob = min(choice_prob + staybias, 1-1e-6)
        choice_prob = min(max(choice_prob, 1e-6), 1-1e-6)
        next_state = 3 if second_step == 0 else 2  # 3: down, 2: up
        state_number = current_state
        next_state_number = next_state
        td_error = np.eye(len(states))[next_state_number,:] - successor_representation[state_number,:] + discount * successor_representation[next_state_number,:]
        successor_representation[state_number,:] += learning_rate * td_error
        current_state = next_state
        # Always use baseline learning rate for reward update, regardless of outcome
        if outcome == 1 and current_state == 2:
            next_state = 4  # up_reward
        elif outcome == 1 and current_state == 3:
            next_state = 5  # down_reward
        else:
            next_state = 6  # no_reward
        state_number = current_state
        next_state_number = next_state
        learned_rewards[next_state_number] = learned_rewards[next_state_number]*(1- reward_learning_rate) + reward_learning_rate*given_rewards[next_state_number]
        # Use only baseline learning rate for SR update
        td_error = np.eye(len(states))[next_state_number,:] - successor_representation[state_number,:] + discount * successor_representation[next_state_number,:]
        successor_representation[state_number,:] += learning_rate * td_error
        choices_list[i] = choice
        choice_probs_list[i] = choice_prob
        free_choice_list[i] = free_choice_trial
    log_likelihood = 0.0
    for i in range(n_episodes):
        if free_choice_list[i]:
            choice = choices_list[i]
            choice_prob = choice_probs_list[i]
            if choice == 1:
                log_likelihood += np.log(choice_prob)
            else:
                log_likelihood += np.log(1 - choice_prob)
    return -log_likelihood


@njit
def likelihood_term_asymmetric_model_based(x, free_choice, transitions, choices, outcomes, second_steps, stims, n_episodes,transition_type):
    # x: [alpha_pos, alpha_neg, temperature, forget]
    alpha_rew = 1/(1 + np.exp(-x[0]))
    alpha_norew = 1/(1 + np.exp(-x[1]))
    forget = 1/(1 + np.exp(-x[2]))
    temperature = np.exp(x[3])

    if temperature < 1e-6:
        temperature = 1e-6

    # State indices: 2 = up, 3 = down
    V = np.zeros(2)  # V[0]=up, V[1]=down

    choices_list = np.empty(n_episodes, dtype=np.int32)
    choice_probs_list = np.empty(n_episodes, dtype=np.float64)
    free_choice_list = np.empty(n_episodes, dtype=np.bool_)



    # Model-based Q-values for first step using a single transition_type for all trials
    if transition_type[0]: # transition type doesn't change across trials
        prob_up, prob_down = 0.8, 0.2
    else:
        prob_up, prob_down = 0.2, 0.8


    for i in range(n_episodes):
        free_choice_trial = free_choice[i]
        choice = choices[i]
        outcome = outcomes[i]
        second_step = second_steps[i]


        Q_left = prob_up * V[0] + prob_down * V[1]
        Q_right = prob_down * V[0] + prob_up * V[1]

        p_left = 1/(1 + np.exp((Q_right - Q_left)/temperature))
        p_left = min(max(p_left, 1e-6), 1-1e-6)
        choice_prob = p_left

        alpha = alpha_rew if outcome == 1 else alpha_norew
        if second_step == 0:  # down
            V[1] = (1 - alpha) * V[1] + alpha * outcome
            # V[0] = (1 - forget) * V[0] + forget * 0.5  # decay up toward 0.5
            V[0] = (1 - forget) * V[0]    # decay up toward 0
        else:  # up
            V[0] = (1 - alpha) * V[0] + alpha * outcome
            # V[1] = (1 - forget) * V[1] + forget * 0.5  # decay down toward 0.5
            V[1] = (1 - forget) * V[1]   # decay down toward 0

        choices_list[i] = choice
        choice_probs_list[i] = choice_prob
        free_choice_list[i] = free_choice_trial

    log_likelihood = 0.0
    for i in range(n_episodes):
        if free_choice_list[i]:
            choice = choices_list[i]
            choice_prob = choice_probs_list[i]
            if choice == 1:
                log_likelihood += np.log(choice_prob)
            else:
                log_likelihood += np.log(1 - choice_prob)
    return -log_likelihood



from scipy.special import expit


def likelihood_term_mf(x, free_choice, transitions, choices, outcomes, second_steps, stims, n_episodes):

    # free_choice, transitions, choices, outcomes, second_steps, stims, n_episodes = model_utils.import_data(session)
    learning_rate_base_transformed, discount_transformed, reward_learning_rate_transformed, temperature_transformed = x
    learning_rate = 1/(1 + np.exp(-learning_rate_base_transformed))
    discount = 1/(1 + np.exp(-discount_transformed))
    #threshold = 1/(1 + np.exp(-threshold_transformed))
    # learning_rate_rew_stim = 1/(1 + np.exp(-learning_rate_rew_stim_transformed))
    # learning_rate_norew_stim = 1/(1 + np.exp(-learning_rate_norew_stim_transformed))
    # learning_rate_rew_nostim = 1/(1 + np.exp(-learning_rate_rew_nostim_transformed))
    reward_learning_rate = 1/(1 + np.exp(-reward_learning_rate_transformed))
    # staybias = 1/(1 + np.exp(-stay_Bias_transformed))
    #choice_proportion = 1/(1 + np.exp(-choice_proportion))
    states = ['right','left','up','down','up_reward','down_reward','no_reward']
    given_rewards = [0,0,0,0,1,1,0]
    terminal_states = ['up_reward','down_reward','no_reward']
    value_vector = np.zeros(len(states))

    
    #stim_value = 1/(1 + np.exp(-stim_transformed))
    #epsilon = 1/(1 + np.exp(-epsilon_transformed))
    #gpe_rate = 1/(1 + np.exp(-gpe_rate_transformed))
    temperature = np.exp(temperature_transformed)

    # Clamp temperature to avoid division by zero
    if temperature < 1e-6:
        temperature = 1e-6

    #reward_stim_value = 1/(1+np.exp(rew_stim_transformed))
    #gpe_thresh = -1
    #state_dict = {state: i for i, state in enumerate(states)}
    # successor_representation = np.zeros((len(states),len(states)))
    # thresholded_successor = np.zeros((len(states),len(states)))
   # successor_representation = sr_initial.copy()
    #thresholded_successor = sr_initial.copy()
    # thresholded_successor_list = []
    values_list = []
    rewards_list = []
    choices_list = np.zeros(n_episodes, dtype=int)
    free_choice_list = np.zeros(n_episodes, dtype=bool)
    choice_probs_list = np.zeros(n_episodes)
    rewards = np.zeros(len(states))
    rewards_up = []
    rewards_down = []
    learned_rewards = np.zeros(len(states))
    #value = np.zeros(len(states))
    #value_learning_rate = 0.1
    #value_discount = 0.9
    gpe_accumulator = 0
    # graph_list = []
    # graph = np.zeros((len(states),len(states)))
    # results = pd.DataFrame(columns=['choices', 'choice_probs', 'second_steps', 'outcomes', 'transitions', 'free_choice', 'stage', 'mov_average', 'stim', 'stim_type','block'])
    for i in range(n_episodes):
        free_choice_trial = free_choice[i]
        # check if stims is a list or a single string 'NaN'
        if not isinstance(stims,str):
            stim_trial = stims[i] if stims[i] != 'NaN' else 0
        else:
            stim_trial = 0
        choice = choices[i]
        outcome = outcomes[i]
        reward_stim = 1 if outcome == 1 else 0
        transition = transitions[i]
        second_step = second_steps[i]
        if choice == 0:
            current_state = 'right'
            action_index = 0
        else:
            current_state = 'left'
            action_index = 1
        #choice_predicted, action_index_predicted = choice_2step(states, thresholded_successor, actions)
        # values = learned_rewards.copy()
        values_list.append(value_vector.copy())
        value_left = value_vector[1]
        value_right = value_vector[0]
        vlist = [value_left, value_right]
        # choice_prob = model_utils.choice_function(temperature, vlist)

        # estimate p_left
        choice_prob = 1/(1+expit((value_right - value_left) / temperature))
        choice_prob = min(max(choice_prob, 1e-6), 1-1e-6)

        # if i > 1 and choices[i-1] == 0:
        #     choice_prob = choice_prob - staybias if choice_prob - staybias > 0 else 0.001
        # if i > 1 and choices[i-1] == 1:
        #     choice_prob = choice_prob + staybias if choice_prob + staybias < 1 else 0.999
        # get the model's choice
        model_choice = np.random.choice([0,1], p = [choice_prob, 1 - choice_prob])
        # compute moving average of performance
        if second_step == 0:
            next_state = 'down'
        else:
            next_state = 'up'
        # update the SR
        state_number = states.index(current_state)
        current_state_number = states.index(current_state)
        next_state_number = states.index(next_state)
        # reward learning
        forgetting_rate = 0.9
        
        #TD learning for rewards
        learned_rewards[next_state_number] = learned_rewards[next_state_number]*(1- reward_learning_rate) + reward_learning_rate*given_rewards[next_state_number]

        # value learning
        value_error = learned_rewards[current_state_number] - value_vector[current_state_number] + discount * value_vector[next_state_number]
        value_vector[current_state_number] = value_vector[current_state_number] + learning_rate * value_error
        # # get the value of the next state
        # values = get_values_from_graph_2(learned_rewards,graph)
        # if values[next_state_number] > 0.5:
        #     lr = learning_rate_rew_nostim
        # else:
        #     lr = learning_rate
        # other_state_number = states.index('down') if next_state_number == states.index('up') else states.index('up')
        # """ if gpe_accumulator < gpe_thresh:
        #     #set the thresholded SR to 0 for the current state and action
        #     #thresholded_successor[state_number,:,action_index] = np.zeros(len(states))
        #     thresholded_successor[:,:,action_index] = np.zeros((len(states),len(states)))
        #     gpe_accumulator = 0
        #     successor_representation[:,:,action_index] = np.zeros((len(states),len(states))) """
        #     #successor_representation[state_number,:,action_index] = np.zeros(len(states))
        # td_error = np.eye(len(states))[next_state_number,:]  - successor_representation[state_number,:] +  discount * successor_representation[next_state_number,:]
        # successor_representation[state_number,:] = successor_representation[state_number,:] + learning_rate * td_error


        # columnwise normalize the SR
        #for i in range(len(actions)):
           # successor_representation[:,:,i] = successor_representation[:,:,i]/np.sum(successor_representation[:,:,i], axis = 0)
        # threshold the SR
        # thresholded_successor[state_number,:] = successor_representation[state_number,:].copy()
        # thresholded_successor[thresholded_successor < threshold] = 0
        # thresholded_successor[thresholded_successor >= threshold] = 1
        # graph = thresholded_successor.copy()
        
        current_state = next_state
        if outcome == 1 and current_state == 'up':
            next_state = 'up_reward'
            rewards_up.append(1)
            # if stim_trial == 1:
            #     trial_learning_rate = learning_rate_rew_stim
            # else:
            #     trial_learning_rate = learning_rate_rew_nostim
        elif outcome == 1 and current_state == 'down':
            next_state = 'down_reward'
            rewards_down.append(1)
            # if stim_trial == 1:
            #     trial_learning_rate = learning_rate_rew_stim
            # else:
            #     trial_learning_rate = learning_rate_rew_nostim
        else:
            next_state = 'no_reward'
            # if stim_trial == 1:
            #     trial_learning_rate = learning_rate_norew_stim
            # else:
            #     trial_learning_rate = learning_rate
        other_states = [state for state in terminal_states if state != next_state]
        state_number = states.index(current_state)
        current_state_number = states.index(current_state)
        next_state_number = states.index(next_state)
        # reward learning
        forgetting_rate = 0.9
        learned_rewards[next_state_number] = learned_rewards[next_state_number]*(1- reward_learning_rate) + reward_learning_rate*given_rewards[next_state_number]
       # value learning
        value_error = learned_rewards[current_state_number] - value_vector[current_state_number] + discount * value_vector[next_state_number]
        value_vector[current_state_number] = value_vector[current_state_number] + learning_rate * value_error

        #value of next state (terminal) is just learned rewards
        value_vector[next_state_number] = learned_rewards[next_state_number]
        # get the value of the next state
        values = learned_rewards.copy()
        # if values[next_state_number] > 0.5 and stim_trial == 1:
        #     lr = learning_rate_rew_stim
        # elif values[next_state_number] > 0.5 and stim_trial == 0:
        #     lr = learning_rate_rew_nostim
        # elif values[next_state_number] <= 0.5 and stim_trial == 1:
        #     lr = learning_rate_norew_stim
        # elif values[next_state_number] <= 0.5 and stim_trial == 0:
        #     lr = learning_rate
        # for state in other_states:
        #     other_state_number = states.index(state)
        #     rewards[other_state_number] = rewards[other_state_number]*(reward_learning_rate)

        # td_error = np.eye(len(states))[next_state_number,:] - successor_representation[state_number,:] + discount * successor_representation[next_state_number,:]
        # successor_representation[state_number,:] = successor_representation[state_number,:] + lr * td_error
        # thresholded_successor[state_number,:] = successor_representation[state_number,:].copy()
        # thresholded_successor[thresholded_successor < threshold] = 0
        # thresholded_successor[thresholded_successor >= threshold] = 1
        # graph = thresholded_successor.copy()

        # results = pd.concat([results, pd.DataFrame({'choices': choice, 'choice_probs': choice_prob, 'second_steps': second_step, 'outcomes': outcome, 'transitions': transition, 'free_choice': free_choice_trial, 'stage': 4.7, 'mov_average': 0, 'stim': stim_trial, 'stim_type': 'outcome_cue', 'block': 0}, index=[0])],axis=0, ignore_index=True)
        # thresholded_successor_list.append(thresholded_successor)
        # graph_list.append(graph.copy())
        # rewards_list.append(rewards.copy())
        choices_list[i] = choice
        choice_probs_list[i] = choice_prob
        free_choice_list[i] = free_choice_trial
    # estimate likelihood for this subject
    
    log_likelihood = 0

    conditionSelect = np.where((free_choice_list == True))
    # choices_list = np.array(choices_list)
    # choice_probs_list = np.array(choice_probs_list)

    ## use to fit using free trials only
    choices_actual = choices_list[conditionSelect[0]]
    results_free = choice_probs_list[conditionSelect[0]]

    ## use to fit using free-free trials only
    # free_choice_prev = free_choice_list.shift(1)
    # choices_actual = choices[np.where((free_choice == True) & (free_choice_prev == True) )[0]]
    # results_free = results[(results['free_choice']==True) & (free_choice_prev == True)]['choice_probs'].values

    for i in range(len(choices_actual)):
        choice = choices_actual[i]
        choice_prob = results_free[i]
        if choice == 1:
            log_likelihood += np.log(choice_prob)
        else:
            log_likelihood += np.log(1 - choice_prob)
    # print('log_likelihood:', log_likelihood)
    
    return -log_likelihood


@njit
def likelihood_term_basic_model_based(x, free_choice, transitions, choices, outcomes, second_steps, stims, n_episodes,transition_type):
    # x: [alpha_pos, alpha_neg, temperature, forget]
    alpha = 1/(1 + np.exp(-x[0]))
    # alpha_norew = 1/(1 + np.exp(-x[1]))
    # forget = 1/(1 + np.exp(-x[2]))
    temperature = np.exp(x[1])

    if temperature < 1e-6:
        temperature = 1e-6

    # State indices: 2 = up, 3 = down
    V = np.zeros(2)  # V[0]=up, V[1]=down

    choices_list = np.empty(n_episodes, dtype=np.int32)
    choice_probs_list = np.empty(n_episodes, dtype=np.float64)
    free_choice_list = np.empty(n_episodes, dtype=np.bool_)



    # Model-based Q-values for first step using a single transition_type for all trials
    if transition_type[0]: # transition type doesn't change across trials
        prob_up, prob_down = 0.8, 0.2
    else:
        prob_up, prob_down = 0.2, 0.8


    for i in range(n_episodes):
        free_choice_trial = free_choice[i]
        choice = choices[i]
        outcome = outcomes[i]
        second_step = second_steps[i]


        Q_left = prob_up * V[0] + prob_down * V[1]
        Q_right = prob_down * V[0] + prob_up * V[1]

        p_left = 1/(1 + np.exp((Q_right - Q_left)/temperature))
        p_left = min(max(p_left, 1e-6), 1-1e-6)
        choice_prob = p_left

        # alpha = alpha_rew if outcome == 1 else alpha_norew
        if second_step == 0:  # down
            V[1] = (1 - alpha) * V[1] + alpha * outcome
            # V[0] = (1 - forget) * V[0] + forget * 0.5  # decay up toward 0.5
            # V[0] = (1 - forget) * V[0]    # decay up toward 0
        else:  # up
            V[0] = (1 - alpha) * V[0] + alpha * outcome
            # V[1] = (1 - forget) * V[1] + forget * 0.5  # decay down toward 0.5
            # V[1] = (1 - forget) * V[1]   # decay down toward 0

        choices_list[i] = choice
        choice_probs_list[i] = choice_prob
        free_choice_list[i] = free_choice_trial

    log_likelihood = 0.0
    for i in range(n_episodes):
        if free_choice_list[i]:
            choice = choices_list[i]
            choice_prob = choice_probs_list[i]
            if choice == 1:
                log_likelihood += np.log(choice_prob)
            else:
                log_likelihood += np.log(1 - choice_prob)
    return -log_likelihood



@njit
def likelihood_term_transition_model_based(x, free_choice, transitions, choices, outcomes, second_steps, stims, n_episodes,transition_type):
    # x: [alpha, temperature]
    alpha = 1/(1 + np.exp(-x[0]))
    temperature = np.exp(x[1])

    if temperature < 1e-6:
        temperature = 1e-6

    # State indices: 2 = up, 3 = down
    V = np.zeros(2)  # V[0]=up, V[1]=down

    # Transition counts matrix for learning prob_up and prob_down
    # Row 0 = left (state 0), Row 1 = right (state 1)
    # Col 0 = up (state 2), Col 1 = down (state 3)
    T_counts = np.zeros((2, 2))  # T_counts[start_state, destination]

    choices_list = np.empty(n_episodes, dtype=np.int32)
    choice_probs_list = np.empty(n_episodes, dtype=np.float64)
    free_choice_list = np.empty(n_episodes, dtype=np.bool_)

    for i in range(n_episodes):
        free_choice_trial = free_choice[i]
        choice = choices[i]
        outcome = outcomes[i]
        second_step = second_steps[i]
        transition = transitions[i]

        # Learn transition structure: count transitions
        # transition maps first step choice to second step state
        # choice: 0=right, 1=left
        # second_step: 0=down, 1=up
        # if transition == 1:  # common transition
        #     T_counts[choice, second_step] += 1
        # else:  # rare transition
        #     T_counts[choice, 1 - second_step] += 1  # opposite destination

        # Calculate learned prob_up and prob_down for each first step state
        # For left (choice=1): prob_up = P(up | left), prob_down = P(down | left)
        # For right (choice=0): prob_up = P(up | right), prob_down = P(down | right)
        
        # Add small pseudocount to avoid division by zero
        left_total = np.sum(T_counts[1, :]) + 1e-8
        right_total = np.sum(T_counts[0, :]) + 1e-8
        
        prob_up_left = T_counts[1, 1] / left_total
        prob_down_left = T_counts[1, 0] / left_total
        prob_up_right = T_counts[0, 1] / right_total
        prob_down_right = T_counts[0, 0] / right_total

        # Use learned transition probabilities for decision
        Q_left = prob_up_left * V[0] + prob_down_left * V[1]
        Q_right = prob_up_right * V[0] + prob_down_right * V[1]

        p_left = 1/(1 + np.exp((Q_right - Q_left)/temperature))
        p_left = min(max(p_left, 1e-6), 1-1e-6)
        choice_prob = p_left

        # Update state values based on outcome
        if second_step == 0:  # down
            V[1] = (1 - alpha) * V[1] + alpha * outcome
        else:  # up
            V[0] = (1 - alpha) * V[0] + alpha * outcome

        choices_list[i] = choice
        choice_probs_list[i] = choice_prob
        free_choice_list[i] = free_choice_trial

    log_likelihood = 0.0
    for i in range(n_episodes):
        if free_choice_list[i]:
            choice = choices_list[i]
            choice_prob = choice_probs_list[i]
            if choice == 1:
                log_likelihood += np.log(choice_prob)
            else:
                log_likelihood += np.log(1 - choice_prob)
    return -log_likelihood



@njit
# stay bias parameter inside the logisitic
def likelihood_term_free_free_noStim_2lr_stay_forget(x, free_choice, transitions, choices, outcomes, second_steps, stims, n_episodes): 

    
    # Use len() to get a proper integer for array initialization
    n_trials = len(free_choice)
    
    # x: [lr_base, lr_rew, discount, threshold, temperature, reward_learning_rate, staybias]
    states = ['right','left','up','down','up_reward','down_reward','no_reward']
    learning_rate = 1/(1 + np.exp(-x[0]))
    discount = 1/(1 + np.exp(-x[2]))
    threshold = 1/(1 + np.exp(-x[3]))
    reward_learning_rate = 1/(1 + np.exp(-x[5]))
    staybias = np.tanh(x[6])
    # forget = 1/(1 + np.exp(-x[7]))

    temperature = np.exp(x[4])
    if temperature < 1e-6:
        temperature = 1e-6
    given_rewards = np.array([0,0,0,0,1,1,0], dtype=np.float64)
    successor_representation = np.zeros((len(states),len(states)), dtype=np.float64)
    learned_rewards = np.zeros(len(states), dtype=np.float64)
    graph = np.zeros((len(states),len(states)), dtype=np.float64)
    choices_list = np.empty(n_trials, dtype=np.int32)
    choice_probs_list = np.empty(n_trials, dtype=np.float64)
    free_choice_list = np.empty(n_trials, dtype=np.int8)  # Use int8 instead of bool for numba


    lapse_rate = 0.00000000000000000001


    for i in range(n_trials):
        free_choice_trial = free_choice[i]
        choice = choices[i]
        outcome = outcomes[i]
        # transition = transitions[i]
        second_step = second_steps[i]
        current_state = 0 if choice == 0 else 1  # 0: right, 1: left
        # Use inlined get_values_from_graph_2 logic

        values = get_values_from_graph_2_numba(learned_rewards, graph)
        value_left = values[1]
        value_right = values[0]

        # Stay bias term: positive if previous choice was right (0), negative if left (1)
        # Only applies when i > 0
        prev_choice_idx = max(0, i - 1)
        prev_choice = choices[prev_choice_idx]
        has_prev = 1.0 * (i > 0)  # 1.0 if there's a previous trial, else 0.0

        stay_bias_term = has_prev * staybias * (1.0 - 2.0 * prev_choice)

        # Include stay bias in the value difference before the logistic
        
        value_diff = value_right - value_left + stay_bias_term 

        p_left = 1.0 / (1.0 + np.exp(value_diff / temperature))

        p_left = (1 - lapse_rate) * p_left + lapse_rate * 0.5

        choice_prob = p_left

        next_state = 3 if second_step == 0 else 2  # 3: down, 2: up
        state_number = current_state
        next_state_number = next_state
        td_error = np.eye(len(states))[next_state_number,:] - successor_representation[state_number,:] + discount * successor_representation[next_state_number,:]
        successor_representation[state_number,:] += learning_rate * td_error
        thresholded_successor = (successor_representation >= threshold).astype(np.float64)
        graph = thresholded_successor.copy()
        current_state = next_state
        # Only two cases for reward update: base or base+add_rew
        if outcome == 1:
            lr = x[0] + x[1]  # lr_base + add_rew
        else:
            lr = x[0]         # lr_base
        learning_rate_trial = 1/(1 + np.exp(-lr))

        # Determine next state based on outcome and second step
        if outcome == 1 and current_state == 2:
            next_state = 4  # up_reward
            # other_state = np.array([5], dtype=np.int64)


        elif outcome == 1 and current_state == 3:
            next_state = 5  # down_reward
            # other_state = np.array([4], dtype=np.int64)
        else:
            next_state = 6  # no_reward
            # other_state = np.array([4,5], dtype=np.int64)

        state_number = current_state
        next_state_number = next_state


        # Apply forgetting to all non-current reward states
        # for s in range(len(states)):
        #     if s in other_state:
        learned_rewards = learned_rewards * (1 - reward_learning_rate)

        # learned_rewards[next_state_number] = learned_rewards[next_state_number]*(1- reward_learning_rate) + reward_learning_rate*given_rewards[next_state_number]
        learned_rewards[next_state_number] = learned_rewards[next_state_number] + reward_learning_rate*given_rewards[next_state_number]


        td_error = np.eye(len(states))[next_state_number,:] - successor_representation[state_number,:] + discount * successor_representation[next_state_number,:]
        successor_representation[state_number,:] += learning_rate_trial * td_error
        thresholded_successor = (successor_representation >= threshold).astype(np.float64)
        graph = thresholded_successor.copy()
        choices_list[i] = choice
        choice_probs_list[i] = choice_prob
        free_choice_list[i] = int(free_choice_trial)  # Convert to int
        
    log_likelihood = 0.0
    for i in range(n_trials-1):# to avoid index error
        if free_choice_list[i] and free_choice_list[i+1]:
            choice = choices_list[i+1]
            choice_prob = choice_probs_list[i+1]
            if choice == 1:
                log_likelihood += np.log(choice_prob)
            else:
                log_likelihood += np.log(1 - choice_prob)

    
    # print('IT IS RUNNING LIKELIHOOD TERMS!!!!!1')

    return -log_likelihood




@njit
# stay bias parameter inside the logisitic
def likelihood_term_free_free_noStim_1lr_stay_forget(x, free_choice, transitions, choices, outcomes, second_steps, stims, n_episodes): 
    
    # Use len() to get a proper integer for array initialization
    n_trials = len(free_choice)
    
    # x: [lr_base, lr_rew, discount, threshold, temperature, reward_learning_rate, staybias]
    states = ['right','left','up','down','up_reward','down_reward','no_reward']
    
    learning_rate = 1/(1 + np.exp(-x[0]))
    discount = 1/(1 + np.exp(-x[1]))
    threshold = 1/(1 + np.exp(-x[2]))
    reward_learning_rate = 1/(1 + np.exp(-x[4]))
    staybias = np.tanh(x[5])
    # forget = 1/(1 + np.exp(-x[7]))

    temperature = np.exp(x[3])
    
    if temperature < 1e-6:
        temperature = 1e-6
    given_rewards = np.array([0,0,0,0,1,1,0], dtype=np.float64)
    successor_representation = np.zeros((len(states),len(states)), dtype=np.float64)
    learned_rewards = np.zeros(len(states), dtype=np.float64)
    graph = np.zeros((len(states),len(states)), dtype=np.float64)
    choices_list = np.empty(n_trials, dtype=np.int32)
    choice_probs_list = np.empty(n_trials, dtype=np.float64)
    free_choice_list = np.empty(n_trials, dtype=np.int8)  # Use int8 instead of bool for numba


    lapse_rate = 0.00000000000000000001


    for i in range(n_trials):
        free_choice_trial = free_choice[i]
        choice = choices[i]
        outcome = outcomes[i]
        # transition = transitions[i]
        second_step = second_steps[i]
        current_state = 0 if choice == 0 else 1  # 0: right, 1: left
        # Use inlined get_values_from_graph_2 logic

        values = get_values_from_graph_2_numba(learned_rewards, graph)
        value_left = values[1]
        value_right = values[0]

        # Stay bias term: positive if previous choice was right (0), negative if left (1)
        # Only applies when i > 0
        prev_choice_idx = max(0, i - 1)
        prev_choice = choices[prev_choice_idx]
        has_prev = 1.0 * (i > 0)  # 1.0 if there's a previous trial, else 0.0

        stay_bias_term = has_prev * staybias * (1.0 - 2.0 * prev_choice)

        # Include stay bias in the value difference before the logistic
        
        value_diff = value_right - value_left + stay_bias_term 

        p_left = 1.0 / (1.0 + np.exp(value_diff / temperature))

        p_left = (1 - lapse_rate) * p_left + lapse_rate * 0.5

        choice_prob = p_left

        next_state = 3 if second_step == 0 else 2  # 3: down, 2: up
        state_number = current_state
        next_state_number = next_state
        td_error = np.eye(len(states))[next_state_number,:] - successor_representation[state_number,:] + discount * successor_representation[next_state_number,:]
        successor_representation[state_number,:] += learning_rate * td_error
        thresholded_successor = (successor_representation >= threshold).astype(np.float64)
        graph = thresholded_successor.copy()
        current_state = next_state
        # Only two cases for reward update: base or base+add_rew
        # if outcome == 1:
        #     lr = x[0] + x[1]  # lr_base + add_rew
        # else:
        #     lr = x[0]         # lr_base
        learning_rate_trial = learning_rate

        # Determine next state based on outcome and second step
        if outcome == 1 and current_state == 2:
            next_state = 4  # up_reward
            # other_state = np.array([5], dtype=np.int64)


        elif outcome == 1 and current_state == 3:
            next_state = 5  # down_reward
            # other_state = np.array([4], dtype=np.int64)
        else:
            next_state = 6  # no_reward
            # other_state = np.array([4,5], dtype=np.int64)

        state_number = current_state
        next_state_number = next_state


        # Apply forgetting to all non-current reward states
        # for s in range(len(states)):
        #     if s in other_state:
        learned_rewards = learned_rewards * (1 - reward_learning_rate)

        # learned_rewards[next_state_number] = learned_rewards[next_state_number]*(1- reward_learning_rate) + reward_learning_rate*given_rewards[next_state_number]
        learned_rewards[next_state_number] = learned_rewards[next_state_number] + reward_learning_rate*given_rewards[next_state_number]


        td_error = np.eye(len(states))[next_state_number,:] - successor_representation[state_number,:] + discount * successor_representation[next_state_number,:]
        successor_representation[state_number,:] += learning_rate_trial * td_error
        thresholded_successor = (successor_representation >= threshold).astype(np.float64)
        graph = thresholded_successor.copy()
        choices_list[i] = choice
        choice_probs_list[i] = choice_prob
        free_choice_list[i] = int(free_choice_trial)  # Convert to int
        
    log_likelihood = 0.0
    for i in range(n_trials-1):# to avoid index error
        if free_choice_list[i] and free_choice_list[i+1]:
            choice = choices_list[i+1]
            choice_prob = choice_probs_list[i+1]
            if choice == 1:
                log_likelihood += np.log(choice_prob)
            else:
                log_likelihood += np.log(1 - choice_prob)

    
    # print('IT IS RUNNING LIKELIHOOD TERMS!!!!!1')

    return -log_likelihood