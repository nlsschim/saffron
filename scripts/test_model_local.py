import torch
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tqdm

# Add src to path so we can import saffron
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from saffron.io import data_io
from saffron.data.datasets import MicrogliaDataset, generate_dataloaders
from saffron.data import data_processing
from saffron.models.torch_models import MicrogliaCNN
from torchvision.transforms import v2

def device_check(req_dev):
	"""
	Decides the device being used for training.
	Returns:
		req_dev: [None, "cpu", "cuda", "mps"]
	"""
	device = None

	if req_dev:
		# Attempt manual device selection
		if req_dev == "mps" and torch.backends.mps.is_available():
			device = torch.device("mps")  # Apple Silicon GPU
		elif req_dev == "cuda" and torch.cuda.is_available():
			device = torch.device("cuda")  # NVIDIA GPU
		else:
			device = torch.device("cpu")  # Defaults to CPU
	else:
		# Automatic device detection
		if torch.backends.mps.is_available():
			# Check and use Apple Silicon GPU
			# https://pytorch.org/docs/stable/notes/mps.html
			device = torch.device("mps")
		elif torch.cuda.is_available():
			# The provided code for CUDA
			device = torch.device("cuda")
		else:
			# Default to CPU if no accelerator available
			device = torch.device("cpu")

	return device


def make_dataloaders():

    transforms = v2.Compose([
        #EnsureRGB(),
        #v2.ToDtype(torch.float32, scale=True),
        v2.Resize(size=(512, 512)),
        #v2.RandomHorizontalFlip(p=0.5),
        #v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # ImageNet-style
		])
    
    images_path = Path("/gscratch/cheme/nlsschim/data/microglia_data/ml_images/split_tifs/")
    # images = data_io.load_images_from_directory(images_path)

    microglia_dataset = MicrogliaDataset(images_path,
                                         train=True,
                                         labels=['HC', 'OGD', 'ROT'],
                                         transform=transforms)
    print(microglia_dataset.length)

    data_train, data_val = generate_dataloaders(microglia_dataset, num_workers=1)
    return data_train, data_val


def train(model, weights, epochs, data, device, loss_func, optimizer):
    train_loss_epoch = []
    val_loss_epoch = []
    val_accuracy_epoch = []

	# Stage model on whatever device we are using
    model.to(device)

	# Split out the training and validation DataLoaders
    data_train, data_val = data

    # Prettier tqdm progress bar
    pbar_epoch = tqdm.tqdm(iterable=range(epochs), colour="green", desc="Epoch")
    for epoch in pbar_epoch:
        train_loss_batch = []
        val_loss_batch = []
        val_accuracy_batch = []

        # batches (data.dataset.length)
        pbar_batch = tqdm.tqdm(total=len(data_train), colour="blue", desc="Batch", leave=False)
        for _, (images, labels) in enumerate(data_train):
            images = images.to(device)
            labels = labels.to(device)

            train_outputs = model(images)
            loss = loss_func(train_outputs, labels)
            train_loss_batch.append(loss.item()) # batch loss
            
            # Batch-level backpropagation
            optimizer.zero_grad() # Zero gradients from previous epoch
            loss.backward() # Calculate gradient
            optimizer.step() # Update weights

            #if batch % 35 == 0 and batch != 0:
            # Compute validation accuracy
            val_pred = []
            val_lbls = []
            val_loss = []
            with torch.no_grad():
                for _, (images, labels) in enumerate(data_val):
                    images = images.to(device)
                    labels = labels.to(device)

                    val_outputs = model(images)
                    val_loss_iter = loss_func(val_outputs, labels)
                    val_loss.append(val_loss_iter.item())
                    val_pred += model(images).cpu().tolist()
                    val_lbls += labels.cpu().tolist()

            # numpy object for vectorization
            val_pred = np.array(val_pred).argmax(axis=1)
            val_lbls = np.array(val_lbls)

            # Get validation accuracy
            val_accuracy_batch.append((val_pred == val_lbls).mean())
            val_loss_batch.append(np.array(val_loss).mean())

            pbar_batch.set_postfix({
                    "Loss": train_loss_batch[-1],
                    "Acc": val_accuracy_batch[-1],
                })

            pbar_batch.update(1)

            # Save weights at the end of each batch
            if weights:
                # logging.info(f"Checkpoint model to {weights}")
                torch.save(model.state_dict(), weights)

        pbar_batch.close()

        # Save batch stats at the epoch level
        val_loss_epoch.append(np.mean(val_loss_batch))
        val_accuracy_epoch.append(np.mean(val_accuracy_batch))
        train_loss_epoch.append(np.mean(train_loss_batch))

        # Display loss and accuracy in tqdm progress bar
        pbar_epoch.set_postfix({
                "Loss": np.mean(val_loss_epoch[epoch]), 
                "Acc": np.mean(val_accuracy_epoch[epoch])
            })

        # Plot training curves
        epochs = range(1, len(val_accuracy_epoch)+1)
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, val_accuracy_epoch, marker='o', color='blue', label='Validation Accuracy')
        plt.title('Validation Accuracy per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy (%)')
        plt.grid(True)
        plt.legend()
        plt.savefig('fig/validation.png')
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, val_loss_epoch, marker='o', color='blue', label='Validation Loss')
        plt.plot(epochs, train_loss_epoch, marker='o', color='red', label='Training Loss')
        plt.title('Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig('fig/training.png')
        plt.close()

    return val_accuracy_epoch, val_loss_epoch, train_loss_epoch


if __name__ == "__main__":



    data_train, data_val = make_dataloaders()
    model = MicrogliaCNN()

    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    val_accuracy, val_loss, train_loss = train(
        model=model,
        weights=None,
        epochs=20,
        device=torch.device("cuda"),
        data=[data_train, data_val],
        loss_func=loss_func,
        optimizer=optimizer
    )
