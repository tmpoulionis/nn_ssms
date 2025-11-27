import os
import torch
import torchaudio
import torchaudio.datasets as datasets
from torch.nn.utils.rnn import pad_sequence

class SCDataset(datasets.SPEECHCOMMANDS):
    """
    Custom PyTorch Dataset for the Speech Commands V0.02 dataset.
    This class extends torchaudio.datasets.SPEECHCOMMANDS to:
        - Apply MelSpectrogram transformation to the raw audio waveform if enabled.
        - Allows filtering the dataset to include only a specific set of labels (e.g., 'sc09').
        - Maps string labels to integer indices (0 to num_classes-1).
    """
    
    def __init__(self, subset, root='./data/speechcommands', mel_transform=True, n_mels=64, n_fft=400, hop_length=150, filter_labels=None):
        
        if not os.path.exists(root):
            os.makedirs(root, exist_ok=True)
        
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
            self.data_indices = list(range(super(SCDataset, self).__len__()))
            
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
        norm_waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8) # Normalize
        
        return norm_waveform, label_index
    
    def __len__(self):
        return len(self.data_indices)
    
    @property
    def num_classes(self):
        return len(self.labels)


def sc_custom_collate(batch):
    """
    Pads sequences in a batch to the maximum length.
    Expects batch elements to be (data, label).
    """
    data_tensors =[item[0].T for item in batch] # Keep transposed since the model expects (B, L, D)
    labels = torch.as_tensor([item[1] for item in batch])
    
    padded_data = pad_sequence(data_tensors, batch_first=True)
    
    return padded_data, labels