import torch
import torchvision
import os

from variables import *

def D_train(x, G, D, D_optimizer, criterion):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real, y_real = x, torch.ones(x.shape[0], 1)
    x_real, y_real = x_real.to(device), y_real.to(device)
    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output


    # train discriminator on facke
    z = torch.randn(x.shape[0], 100).to(device)
    x_fake, y_fake = G(z), torch.zeros(x.shape[0], 1).to(device)

    D_output =  D(x_fake)

    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    with torch.no_grad():
        D_accuracy_on_real = (D_real_score>0.5).sum()/x.shape[0]
        D_accuracy_on_fake = (D_fake_score<0.5).sum()/x.shape[0]


    return  D_loss.data.item(), D_real_loss.data.item(), D_fake_loss.data.item(),D_accuracy_on_real,D_accuracy_on_fake


def G_train(x, G, D, G_optimizer, criterion):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(x.shape[0], 100).to(device)
    y = torch.ones(x.shape[0], 1).to(device)

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()


def DW_train(x, G, D, D_optimizer, weight_clip=0.01,n_critic=5):
    """Train the critic (discriminator) using Wasserstein loss"""
    #=======================Train the critic=======================#
    for i in range(n_critic):
        D.zero_grad()
        
        # Train critic on real
        x_real = x.to(device)
        D_real = D(x_real)
        D_real_loss = -torch.mean(D_real)  # Want to maximize D(real)
        
        # Train critic on fake
        z = torch.randn(x.shape[0], 100).to(device)
        x_fake = G(z).detach()  # Detach to avoid computing gradients for generator
        D_fake = D(x_fake)
        D_fake_loss = torch.mean(D_fake)  # Want to minimize D(fake)
        
        # Combined loss (negative since we're minimizing)
        D_loss = D_fake_loss + D_real_loss
        
        # Gradient backprop & optimize ONLY D's parameters
        D_loss.backward()
        D_optimizer.step()
        
        # Weight clipping (essential for WGAN)
        for p in D.parameters():
            p.data.clamp_(-weight_clip, weight_clip)
    
    # Calculate Wasserstein estimate
    with torch.no_grad():
        Wasserstein_D = -D_real_loss - D_fake_loss
        # For compatibility with original metrics
        D_accuracy_on_real = (D_real > 0).float().mean()
        D_accuracy_on_fake = (D_fake < 0).float().mean()
    
    return (D_loss.item(), 
            -D_real_loss.item(), 
            D_fake_loss.item(),
            D_accuracy_on_real,
            D_accuracy_on_fake)


def GW_train(x, G, D, G_optimizer):
    """Train the generator using Wasserstein loss"""
    #=======================Train the generator=======================#
    G.zero_grad()
    
    z = torch.randn(x.shape[0], 100).to(device)
    fake_images = G(z)
    D_output = D(fake_images)
    
    # Generator wants to maximize D(G(z))
    G_loss = -torch.mean(D_output)
    
    # Gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
    
    return G_loss.item()


def save_models(G, D, folder):
    torch.save(G.state_dict(), os.path.join(folder,'G.pth'))
    torch.save(D.state_dict(), os.path.join(folder,'D.pth'))


def load_model(G, folder, mode = 'G'):
    ckpt = torch.load(os.path.join(folder, mode +'.pth'),map_location=torch.device(device))
    G.load_state_dict({k.replace('module.', ''): v for k, v in ckpt.items()})
    return G

def save_real_samples(args, train_loader, force_save=False):
    """Function to save real samples of MNIST, used to calculate FID"""

    real_images_dir = 'samples/real_samples'
    try:
        os.makedirs(real_images_dir, exist_ok=False)
        for batch_idx, (x, _) in enumerate(train_loader):
            if x.shape[0] != args.batch_size:
                image = x.reshape(x.shape[0],28,28)
            else:
                image = x.reshape(args.batch_size, 28, 28)
            for k in range(x.shape[0]):
                filename = os.path.join(real_images_dir, f'real_image_{batch_idx * args.batch_size + k}.png')
                torchvision.utils.save_image(image[k:k+1], filename)

    except:
        if force_save:
            print('Real samples already exist. Overwriting...')
            os.rmdir(real_images_dir)
            os.makedirs(real_images_dir, exist_ok=False)
            for batch_idx, (x, _) in enumerate(train_loader):
                if x.shape[0] != args.batch_size:
                    image = x.reshape(x.shape[0],28,28)
                else:
                    image = x.reshape(args.batch_size, 28, 28)
                for k in range(x.shape[0]):
                    filename = os.path.join(real_images_dir, f'real_image_{batch_idx * args.batch_size + k}.png')
                    torchvision.utils.save_image(image[k:k+1], filename)
        else:
            print('Real samples already exist. Skipping...')



def generate_fake_samples(args, generator, num_samples, device):
        """Function to generate fake samples using the generator"""

        n_samples = 0
        try:
            os.makedirs('samples/fake_samples', exist_ok=False)

        except:
            print('Fake samples directory already exist. Overwriting...')
            os.system("rm -r samples/fake_samples")
            os.makedirs('samples/fake_samples', exist_ok=False)

        with torch.no_grad():
            while n_samples<num_samples:
                z = torch.randn(args.batch_size, 100).to(device)
                x = generator(z)
                x = x.reshape(args.batch_size, 28, 28)
                for k in range(x.shape[0]):
                    if n_samples<num_samples:
                        torchvision.utils.save_image(x[k:k+1], os.path.join('samples/fake_samples', f'{n_samples}.png'))
                        n_samples += 1
