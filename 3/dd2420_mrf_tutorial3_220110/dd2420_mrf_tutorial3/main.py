# Author Marcus Klasson (mklas@kth.se)
# PGM tutorial on MRFs
# Loopy belief propagation in image denoising

import os 
import time 
import numpy as np 
import matplotlib.pyplot as plt 

from sum_product import update_messages, normalize_messages, compute_beliefs, compute_labeling
from utils import save_np_array_as_png

def compute_energy(data_cost, labels, lamb):
    # Compute label cost
    data_cost_for_labels = np.take_along_axis(data_cost, np.expand_dims(labels, axis=-1), axis=-1)
    energy = np.sum(data_cost_for_labels)

    # Compute smoothness cost between neighbors
    smoothness_cost_up = lamb*((labels - np.roll(labels, 1, axis=0)) != 0)
    smoothness_cost_down = lamb*((labels - np.roll(labels, -1, axis=0)) != 0)
    smoothness_cost_left = lamb*((labels - np.roll(labels, 1, axis=1)) != 0)
    smoothness_cost_right = lamb*((labels - np.roll(labels, -1, axis=1)) != 0)

    # Ignore edge costs at boundaries
    smoothness_cost_up[0, :] = 0.0
    smoothness_cost_down[-1, :] = 0.0
    smoothness_cost_left[:, 0] = 0.0
    smoothness_cost_right[:, -1] = 0.0

    # Add smoothness costs to energy
    energy += np.sum(smoothness_cost_up)
    energy += np.sum(smoothness_cost_down) 
    energy += np.sum(smoothness_cost_left) 
    energy += np.sum(smoothness_cost_right)
    return energy

def compute_node_potentials_with_observations(img, tau, n_levels=2):
    h, w = img.shape
    node_potentials = np.zeros([h, w, n_levels])
    for i in range(n_levels):
        node_potentials[:, :, i] = tau * np.abs(img - i) 
    return node_potentials


def run_lbp_denoising(img, tau, lamb, iters=2):
    height, width = img.shape
    energies = []

    obs_node_potentials = compute_node_potentials_with_observations(img, tau)

    # Initialize messages
    msg_up = np.ones([height, width, 2])
    msg_down = np.ones([height, width, 2])
    msg_left = np.ones([height, width, 2])
    msg_right = np.ones([height, width, 2])

    t_start = time.time()

    for i in range(iters):
        # Update messages
        msg_up, msg_down, msg_left, msg_right = update_messages(msg_up, 
                                                                msg_down, 
                                                                msg_left, 
                                                                msg_right, 
                                                                obs_node_potentials, lamb)
        # Normalize messages
        msg_up, msg_down, msg_left, msg_right = normalize_messages(msg_up, 
                                                        msg_down, 
                                                        msg_left, 
                                                        msg_right)
        # Compute beliefs
        beliefs = compute_beliefs(obs_node_potentials, msg_up, msg_down, msg_left, msg_right)
        # Compute labeling of pixels
        labels = compute_labeling(beliefs)
        # Compute energy cost
        energy = compute_energy(obs_node_potentials, labels, lamb)
        energies.append(energy)
        print('Iter %d - Energy: %3.2f - Time elapsed: %3.2f' %(i+1, energy, time.time() - t_start))
        print()
    print('LBP is done. \n')
    return labels, energies

def add_noise_to_image(img, theta=0.1):
    # flip pixel values by probability theta
    m = np.random.binomial(n=1, p=theta, size=img.shape).astype(np.float32)
    out = np.abs(img - m) 
    return out

def show_true_and_noisy_images(img_true, img_noisy):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img_true, cmap='gray')
    ax[0].set_title('True image')
    ax[1].imshow(img_noisy, cmap='gray')
    ax[1].set_title('Noisy image')
    plt.show()

def main():
    # Create directory for saving results
    if not os.path.exists('./results'):
        os.makedirs('./results')

    # Set random seed
    np.random.seed(42)

    # Set parameters
    tau = 2.0 # node coefficent
    lamb = 2.0 # smoothness coefficient
    theta = 0.2 # noise probability for flipping pixel value
    iters = 20 # number of iterations
    
    # Load image
    img = np.load('./images/mrf.npy')
    img_true = (img > 0.5).astype(np.float32) # binarize image

    # Add noise to true image
    img_noisy = add_noise_to_image(img_true, theta=theta)

    results = []

    for t in np.arange(1,10.1,0.2):
        for l in np.arange(1,10.1,0.2):

            # Run LBP with sum-product algorithm
            img_recon, energies = run_lbp_denoising(img_noisy, t, l, iters)
            mse = ((img_true - img_recon)**2).mean(axis=None)
            print(f"MSE {mse}")
            results.append((t,l,mse,energies,img_recon,len(energies)))

    results = sorted(results, key=lambda tup: tup[2])
    img_recon = results[0][4]
    energies = results[0][3]
    tau = results[0][0]
    lamb = results[0][1]
    mse = results[0][2]

    # Save and plot results
    save_np_array_as_png('./results/img_original', img_true)
    save_np_array_as_png('./results/img_noisy', img_noisy)
    save_np_array_as_png('./results/img_recovered', img_recon)

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].imshow(img_true, cmap='gray')
    ax[0, 0].set_title('True image')
    ax[0, 1].imshow(img_noisy, cmap='gray')
    ax[0, 1].set_title('Noisy image')
    ax[1, 0].imshow(img_recon, cmap='gray')
    ax[1, 0].set_title('Recon. image')
    ax[1, 1].plot(np.array(energies), '-o')
    ax[1, 1].set_xlabel('Iteration')
    ax[1, 1].set_ylabel('$E(x, y)$')
    ax[1, 1].set_title('Energy')
    plt.suptitle(f"Tau {tau} ,Lambda {lamb}, Mse {mse}")
    plt.show()


if __name__ == '__main__':
    main()