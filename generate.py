import torch
import torchvision
import os
import argparse


from model import Generator, Discriminator
from utils import load_model
from latent_space_OT import make_image
from variables import *
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate Normalizing Flow.')
    parser.add_argument("--batch_size", type=int, default=2048,
                      help="The batch size to use for training.")
    parser.add_argument("--n_samples", type=int, default=10000,)
    args = parser.parse_args()




    print('Model Loading...')
    # Model Pipeline
    mnist_dim = 784

    G = Generator(g_output_dim = mnist_dim).to(device)
    G.device = device
    G = load_model(G, 'checkpoints', mode = 'G')
    D = Discriminator(mnist_dim).to(device)
    D = load_model(D, 'checkpoints', mode = 'D')
    D.device = device

    if device == 'cuda':
        G = torch.nn.DataParallel(G).to(device)
        G = torch.nn.DataParallel(G).to(device)
        D = torch.nn.DataParallel(D).to(device)
        D = torch.nn.DataParallel(D).to(device)
    else :
        G = G.to(device)
        D = D.to(device)
    print('Model loaded.')
    print('Start Generating')
    os.makedirs('samples', exist_ok=True)


    print('Start Generating')
    os.makedirs('samples_no_OT', exist_ok=True)
    n_samples = 0
    with torch.no_grad():
        while n_samples<10000:
            z = torch.randn(args.batch_size, 100).to(device)
            x = G(z)
            x = x.reshape(args.batch_size, 28, 28)
            for k in range(x.shape[0]):
                if n_samples<10000:
                    torchvision.utils.save_image(x[k:k+1], os.path.join('samples_no_OT', f'{n_samples}.png'))         
                    n_samples += 1

    n_samples = args.n_samples
    img = make_image(G = G, D= D, batchsize = n_samples, N_update=100, ot=True, mode='dot', k=1, lr=0.005, optmode='sgd')
    for k in range(n_samples):
        torchvision.utils.save_image(img[k, :, :], os.path.join('samples', f'{k}.png'))
