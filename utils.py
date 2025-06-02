import librosa
import numpy as np
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    
    # Match n_mfcc and pad/truncate length to 6960
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_flat = mfccs.flatten()

    # Ensure the feature vector is exactly 6960 in length
    if len(mfccs_flat) < 6960:
        # Pad with zeros
        mfccs_flat = np.pad(mfccs_flat, (0, 6960 - len(mfccs_flat)), mode='constant')
    else:
        # Truncate to match
        mfccs_flat = mfccs_flat[:6960]

    return mfccs_flat
