import librosa
from librosa.feature import melspectrogram
from librosa.core.spectrum import _spectrogram
import librosa.filters as filters

import lws

import numpy as np


def tensor2im(input_image, imtype=np.uint8):
    image_numpy = input_image[0].data.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def mel_spectrogram(y, sr=44100, n_fft=2048, crop_to_multiple_of_4=True):
    mel_basis = filters.mel(sr, n_fft,)
    
    S_stft = librosa.stft(y, n_fft=n_fft)
    L = S_stft.shape[1]
    if crop_to_multiple_of_4 and  L % 4 != 0:
    	# The width of the spectrogram needs to be a multiple of 4 for a proper application of the cyclegan model.
    	S_stft = S_stft[:, :L-L%4]
    S = np.abs(S_stft)**2
    S_mel = np.dot(mel_basis, S)
    S_mel = librosa.power_to_db(S_mel)
    S_mel = S_mel
    return S_mel, S_stft
    
    
def mel_to_stft(S, sr=44100, n_fft=2048):
    mel_basis = filters.mel(sr, n_fft,)
    
    S_ = S
    S_ = librosa.db_to_power(S_)
    S_ = np.dot(mel_basis.T, S_)
    S_ = np.sqrt(S_)
    return S_


max_ = 34.0
min_ = -47.0
def png_to_spectrogram(S):
    S = S / 255.0
    S = S*(max_ - min_) + min_
    return S


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def tensor_to_spectrogram_and_audio(tensor, original_audio=None):
    
    img = rgb2gray(tensor2im(tensor))
    S_img = mel_to_stft(png_to_spectrogram(img))
    
    # Calculate the phase of the original audio
    S_mel, S_stft = mel_spectrogram(original_audio, 44100, crop_to_multiple_of_4=True)
    phase = S_stft/np.abs(S_stft)
    S_stft = np.abs(mel_to_stft(S_mel)) * phase
    S_img = np.abs(S_img) * phase

    # Audio 1: reconstruct phase with RTISI-LA, initial is original audio's phase
    n_fft = 2048
    lws_processor = lws.lws(n_fft, n_fft // 4, mode="music", online_iterations=25, batch_iterations=100)
    S_lws = lws_processor.run_lws(S_img.T.astype('complex128')).T
    y_lws = librosa.istft(S_lws)
    
    # Audio 2: transformed audio, with original phase
    y_no_lws = librosa.istft(S_img)
    
    # Audio 3: original audio (not transformed by cyclegan), but sent through mel scale compression
    y_lws_orig = librosa.istft(lws_processor.run_lws(np.abs(S_stft).T.astype('complex128')).T)
    
    return y_lws, S_lws, y_no_lws, S_img, y_lws_orig, np.abs(S_stft)
