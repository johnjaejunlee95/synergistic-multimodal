import numpy as np
import os 
import tarfile


def avmnist_load_data(dir_root, is_spectrogram=False, is_PCA=True):    
    
    # Extract raw audio tar.gz files if not already extracted --> might take a while to extract (15min)
    audio_paths = os.path.join(dir_root, 'audio')
    audio_tar_path = os.path.join(dir_root, 'avmnist_audio_raw.tar.gz')
    
    if not os.path.exists(audio_paths) or not any(file.endswith('.wav') for file in os.listdir(audio_paths)):
        print('Extracting audio files')
        with tarfile.open(audio_tar_path, 'r:gz') as tar:
            tar.extractall(path=dir_root)
    else:
        print('Audio files already extracted')
        
    audio_data_path = [os.path.join(audio_paths, audio_file) for audio_file in os.listdir(audio_paths)]
        
    
    if is_PCA:
        print('Loading PCA-Projected MNIST')
        train_image = np.load(os.path.join(dir_root, 'image_pca','train_data.npy'))
        test_image = np.load(os.path.join(dir_root, 'image_pca','test_data.npy'))
    else:
        print('Loading Original MNIST')
        train_image = np.load(os.path.join(dir_root, 'image','train_data.npy'))
        test_image = np.load(os.path.join(dir_root, 'image','test_data.npy'))
    
    train_label = np.load(os.path.join(dir_root,'train_labels.npy'))
    test_label = np.load(os.path.join(dir_root, 'test_labels.npy')) 
    
    train_audio = np.array(audio_data_path[:len(train_image)])
    test_audio = audio_data_path[len(train_image):]
    if is_spectrogram:
        print('Loading Spectrogram data')
        train_audio = np.load(os.path.join(dir_root, 'spectrogram', 'train_data.npy'))
        test_audio = np.load(os.path.join(dir_root, 'spectrogram', 'test_data.npy'))
    
    # Shuffle the data
    indices = np.arange(len(train_image))
    np.random.shuffle(indices)
    train_image = train_image[indices]
    train_audio = train_audio[indices]
    if not is_spectrogram:
        train_audio = train_audio[indices].tolist()
    train_label = train_label[indices]
    
    train_len = int(len(train_image)* 0.9)
    
    train_images = train_image[:train_len]
    train_audios = train_audio[:train_len]
    train_labels = train_label[:train_len]
    
    val_image = train_image[train_len:]
    val_audio = train_audio[train_len:]
    val_label = train_label[train_len:]
    
    
    
  
    return (train_images, train_audios, train_labels), (val_image, val_audio, val_label), (test_image, test_audio, test_label)