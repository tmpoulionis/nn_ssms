import os
import torch
import torchaudio
import torchaudio.datasets as datasets
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

class SCDataset(datasets.SPEECHCOMMANDS):
    def __init__(self, root, subset, mel_transform=True, n_mels=64, n_fft=400, hop_length=150, filter_labels=None):
        # The torchaudio class handles downloading and file mapping
        super().__init__(root, url="speech_commands_v0.02", download=True, subset=subset)
        
        if mel_transform:
            self.transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length
            )
        else:
            self.transform = None
            
        # IMPORTANT: Get all labels by accessing parent class directly
        # This avoids calling our __getitem__ before filtered_indices is set
        all_labels = sorted(list(set(
            super(SCDataset, self).__getitem__(i)[2] 
            for i in range(super(SCDataset, self).__len__())
        )))
        
        if filter_labels is not None:
            self.labels = [label for label in all_labels if label in filter_labels]
            self.data_indices = [
                i for i in range(super(SCDataset, self).__len__())
                if super(SCDataset, self).__getitem__(i)[2] in filter_labels
            ]
        else:
            self.labels = all_labels
            self.data_indices = list(range(len(self)))
            
        self.label_to_index = {
            label: i for i, label in enumerate(self.labels)
        }
        
    def __getitem__(self, index):
        # Map to actual index if filtered
        new_index = self.data_indices[index]
        waveform, sample_rate, label, speaker_id, utterance_number = super().__getitem__(new_index)
        
        if self.transform is not None:
            waveform = self.transform(waveform) # (1, n_mels, time)
            waveform = waveform.squeeze(0) # (n_mels, time)
        
        label_index = self.label_to_index[label]
        
        return waveform, label_index
    
    def __len__(self):
        return len(self.data_indices)
    
    def get_labels(self):
        return self.labels
    
    @property
    def num_classes(self):
        return len(self.labels)


def get_dataloaders(
    dataset_name: str,
    root: str='./data', 
    batch_size: int=32, 
    mel_transform=True, 
    n_mels=64, 
    n_fft=400, 
    hop_length=150,
    num_workers: int=4
    ):
    """
    Create dataloaders for speech classification tasks.
    
    Args:
        dataset_name: Name of dataset
        root: Root directory for data
        batch_size: Batch size
        mel_transform: Whether to apply mel spectrogram
        n_mels: Number of mel bands
        n_fft: FFT size
        hop_length: Hop length for spectrogram
        max_length: Maximum sequence length (None for variable)
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer
    
    Returns:
        Dictionary containing dataloaders and dataset info
    """
    if not os.path.exists(root):
        os.makedirs(root)
    
    if dataset_name.lower() == "speechcommands":
        print(f"\t Loading {dataset_name.lower()} dataset...")
        
        training_data = SCDataset(
            root=root, subset="training", mel_transform=mel_transform,
            n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )
        validation_data = SCDataset(
            root=root, subset="validation", mel_transform=mel_transform,
            n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )
        test_data = SCDataset(
            root=root, subset="testing", mel_transform=mel_transform,
            n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )
    
    elif dataset_name.lower() == "sc09":
        print(f"\t Loading {dataset_name.lower()} dataset...")
        
        filter_labels = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

        training_data = SCDataset(
            root=root, subset="training", mel_transform=mel_transform,
            n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
            filter_labels=filter_labels
        )
        validation_data = SCDataset(
            root=root, subset="validation", mel_transform=mel_transform,
            n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
            filter_labels=filter_labels
        )
        test_data = SCDataset(
            root=root, subset="testing", mel_transform=mel_transform,
            n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
            filter_labels=filter_labels
        )
        
    else:
        raise ValueError(f"Dataset '{dataset_name}', is not supported.")
    
    print(f"\t Creating DataLoaders...")
    train_loader = DataLoader(training_data, batch_size=batch_size, collate_fn=custom_mamba_collate, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(validation_data, batch_size=batch_size, collate_fn=custom_mamba_collate, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=custom_mamba_collate, shuffle=False, num_workers=num_workers)
    print(f"\t Dataloaders created.")
    
    data_iter = iter(train_loader)
    sample, _ = next(data_iter)
    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "num_classes": training_data.num_classes,
        "labels": training_data.get_labels(),
        "input_shape": sample.shape,
        "feature_dim": sample.shape[-1],
        "sequence_length": sample.shape[1]
    }

def custom_mamba_collate(batch):
    """
    Pads sequences in a batch to the maximum length.
    Expects batch elements to be (data, label).
    """
    data_tensors =[item[0].T for item in batch] # Keep transposed since the model expects (B, L, D)
    labels = torch.as_tensor([item[1] for item in batch])
    
    padded_data = pad_sequence(data_tensors, batch_first=True)
    
    return padded_data, labels