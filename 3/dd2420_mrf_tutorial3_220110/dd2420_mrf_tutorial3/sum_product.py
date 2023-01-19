# Author Marcus Klasson (mklas@kth.se)
# PGM tutorial on MRFs
# Loopy belief propagation in image denoising

import numpy as np 

def compute_labeling(beliefs):
    labels = np.argmax(beliefs, axis=-1)
    return labels
    
def compute_beliefs(data_cost, msg_up, msg_down, msg_left, msg_right):
    _, _, n_levels = data_cost.shape
    beliefs = np.zeros(data_cost.shape)

    # Get incoming messages by rolling messages from previous iteration and opposite direction
    incoming_msg_from_up = np.roll(msg_down, 1, axis=0)
    incoming_msg_from_down = np.roll(msg_up, -1, axis=0)
    incoming_msg_from_left = np.roll(msg_right, 1, axis=1)
    incoming_msg_from_right = np.roll(msg_left, -1, axis=1)
    
    for i in range(n_levels):
        data_cost_for_label = np.exp(-data_cost[:, :, i])
        msg_prod_for_label = (incoming_msg_from_up[:, :, i] * incoming_msg_from_down[:, :, i] * 
                                incoming_msg_from_left[:, :, i] * incoming_msg_from_right[:, :, i])
        beliefs[:, :, i] = data_cost_for_label * msg_prod_for_label
    return beliefs 

def update_messages(msg_up_pre, msg_down_pre, msg_left_pre, msg_right_pre, data_cost, lamb):
    _, _, n_levels = data_cost.shape

    msg_up = np.zeros(msg_up_pre.shape)
    msg_down = np.zeros(msg_up_pre.shape)
    msg_left = np.zeros(msg_up_pre.shape)
    msg_right = np.zeros(msg_up_pre.shape)

    # Get incoming messages by rolling messages from previous iteration and opposite direction
    incoming_msg_from_up = np.roll(msg_down_pre, 1, axis=0)
    incoming_msg_from_down = np.roll(msg_up_pre, -1, axis=0)
    incoming_msg_from_left = np.roll(msg_right_pre, 1, axis=1)
    incoming_msg_from_right = np.roll(msg_left_pre, -1, axis=1)

    for i in range(n_levels):
        
        for j in range(n_levels):
            data_cost_for_label = np.exp(-data_cost[:, :, j])
            smoothness_cost_for_label = np.exp(-lamb * (i != j))

            msg_prod_up = incoming_msg_from_down[:, :, j] * incoming_msg_from_left[:, :, j] * incoming_msg_from_right[:, :, j] 
            msg_prod_down = incoming_msg_from_up[:, :, j] * incoming_msg_from_left[:, :, j] * incoming_msg_from_right[:, :, j] 
            msg_prod_left = incoming_msg_from_right[:, :, j] * incoming_msg_from_up[:, :, j] * incoming_msg_from_down[:, :, j] 
            msg_prod_right = incoming_msg_from_left[:, :, j] * incoming_msg_from_up[:, :, j] * incoming_msg_from_down[:, :, j] 

            msg_up[:, :, i] += smoothness_cost_for_label * (data_cost_for_label * msg_prod_up)
            msg_down[:, :, i] += smoothness_cost_for_label * (data_cost_for_label * msg_prod_down)
            msg_left[:, :, i] += smoothness_cost_for_label * (data_cost_for_label * msg_prod_left)
            msg_right[:, :, i] += smoothness_cost_for_label * (data_cost_for_label * msg_prod_right)
    return msg_up, msg_down, msg_left, msg_right

def normalize_messages(msg_up, msg_down, msg_left, msg_right):
    msg_up = msg_up / np.sum(msg_up, axis=-1, keepdims=True)
    msg_down = msg_down / np.sum(msg_down, axis=-1, keepdims=True) 
    msg_left = msg_left / np.sum(msg_left, axis=-1, keepdims=True) 
    msg_right = msg_right / np.sum(msg_right, axis=-1, keepdims=True) 
    return msg_up, msg_down, msg_left, msg_right