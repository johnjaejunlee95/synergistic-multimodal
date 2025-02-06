import torch
import numpy as np
import torch.nn.functional as F

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True 
    
       
        
def collate_fn_padding(batch):
    audios, language_data, labels = zip(*batch)
    
    # Find the max length of audio in the batch
    max_audio_len = max([audio.shape[1] for audio in audios])

    # Pad all audio sequences to the length of the longest one and calculate the new padding ratio
    padded_audios = []
    padding_ratios = []
    
    for audio in audios:
        original_len = audio.shape[1]
        if original_len < max_audio_len:
            padded_audio = F.pad(audio, (0, max_audio_len - original_len))
        else:
            padded_audio = audio[:, :max_audio_len]
            original_len = max_audio_len
        
        # padded_audio = padded_audio[:, :48000]
        padded_audios.append(padded_audio)
        
        # Calculate the padding ratio relative to the longest sequence
        padding_ratio = original_len / max_audio_len
        padding_ratios.append(padding_ratio)

    # Convert to tensors
    padded_audios = torch.stack(padded_audios).contiguous().view(len(audios), -1)
    padding_ratios = torch.tensor(padding_ratios)
    labels = torch.tensor(labels)

    return (padded_audios, padding_ratios), language_data, labels


def collate_fn_truncate_language(batch):
    audios, language_data, labels = zip(*batch)
    
    min_audio_len = min([audio.shape[1] for audio in audios])
    truncated_audios = []
    
    for audio in audios:
        original_len = audio.shape[1]
        if original_len > min_audio_len:
            padded_audio = audio[:, :min_audio_len]
        else:
            padded_audio = audio
        
        truncated_audios.append(padded_audio)

    # Convert to tensors
    truncated_audios = torch.stack(truncated_audios).contiguous().view(len(audios), -1)
    labels = torch.tensor(labels)

    return truncated_audios, language_data, labels


def collate_fn_truncate_image(batch):
    audios, image, labels = zip(*batch)
    
    min_audio_len = min([audio.shape[1] for audio in audios])
    truncated_audios = []
    
    for audio in audios:
        original_len = audio.shape[1]
        if original_len > min_audio_len:
            padded_audio = audio[:, :min_audio_len]
        else:
            padded_audio = audio
        
        truncated_audios.append(padded_audio)

    # Convert to tensors
    truncated_audios = torch.stack(truncated_audios).contiguous().view(len(audios), -1)
    labels = torch.tensor(labels)
    image = torch.stack(image)
    return truncated_audios, image, labels


def length_to_mask(length, max_len=None, dtype=None, device=None):
    
    assert len(length.shape) == 1

    if max_len is None:
        max_len = length.max().long().item()  # using arange to generate mask
    mask = torch.arange(
        max_len, device=length.device, dtype=length.dtype
    ).expand(len(length), max_len) < length.unsqueeze(1)

    if dtype is None:
        dtype = length.dtype

    if device is None:
        device = length.device

    mask = torch.as_tensor(mask, dtype=dtype, device=device)
    return mask

def make_padding_masks(src, wav_len=None, pad_idx=0):

    src_key_padding_mask = None
    if wav_len is not None:
        abs_len = torch.round(wav_len * src.shape[1])
        src_key_padding_mask = length_to_mask(abs_len).bool()

    return src_key_padding_mask