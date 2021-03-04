# This script shows how the dataset was created for the drum and bass style transfer CycleGAN model
# presented in the paper
# "Vande Veire, Len, De Bie, Tijl, and Dambre, Joni. "A CycleGAN for style transfer between drum & bass subgenres", ML4MD Workshop at ICML 2019, Long Beach, CA, USA, 2019. https://biblio.ugent.be/publication/8619952
#
# (c) Len Vande Veire

import autodj
import autodj.dj.annotators.wrappers as annot 
from autodj.dj.songcollection import SongCollection 
from autodj.dj.timestretching import *

import argparse
import numpy as np
import os
import pyaudio  # audio playback
import sys

from essentia.standard import AudioOnsetsMarker
import librosa
import soundfile as sf
import numpy as np
import os
import sys

from PIL import Image
import librosa.filters as filters
import IPython

import matplotlib.colors as color
import matplotlib.pyplot as plt

import essentia
from essentia.standard import NSGConstantQ, NSGIConstantQ


def create_and_save_wav_extracts(songcollection, out_dir, len_in_downbeats=4, n_seg_per_track=3, overwrite=False):
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for song in songcollection.get_annotated():

        song.open()
        song.openAudio()
        print("{},\"{}\",{}".format(song.dir_, song.title, song.tempo))
        
        out_dir_song = os.path.join(
                    out_dir,
                    os.path.split(song.dir_)[-1],
        )
        if not os.path.exists(out_dir_song):
            os.makedirs(out_dir_song)

        high_segments = [i for i,j in zip(song.segment_indices, song.segment_types) if j == 'H']
        try:
            for j in range(n_seg_per_track):
                filename = os.path.join(
                    out_dir_song,
                    '{}_{}.wav'.format(song.title, j),
                )
        
                if not overwrite and os.path.exists(filename):
                    print('File already exists, skipping!')
                    break
                
                seg_idx = high_segments[j] + j*4 # add multiple of 4 to prevent too similar samples
                start_idx = int(song.downbeats[seg_idx]*44100) 
                end_idx = int(song.downbeats[seg_idx + len_in_downbeats]*44100)
                audio = song.audio[start_idx:end_idx]

                f = song.tempo / 175.0
                audio = librosa.util.fix_length(time_stretch_and_pitch_shift(audio, f,), 240000)

                sf.write(filename, audio, 44100,) # 'stereo_file.wav', np.random.randn(10, 2), 44100, 'PCM_24'
        except IndexError:
            print('\tNo more than {} high segments here...'.format(j))
        song.close()

            
def mel_spectrogram(y, sr=44100, n_fft=2048, n_mels=256,):
    mel_basis = filters.mel(sr, n_fft, n_mels=n_mels)
    
    S_stft = librosa.stft(y, n_fft=n_fft)
    S = np.abs(S_stft)**2    
    
    S_mel = np.dot(mel_basis, S)
    S_mel = librosa.power_to_db(S_mel)
    
    return S_mel, S_stft


def save_matrix_to_grayscale(filename, amplitude,):
    
    img = np.empty(amplitude.shape)
    amplitude_normalized = (amplitude - amplitude.min()) / (amplitude.max() - amplitude.min())
    img[:,:] = amplitude_normalized
    
    img = np.pad(
        img, 
        [(0, (4 - img.shape[0] % 4) % 4),
         (0, (4 - img.shape[1] % 4) % 4),],
        'constant', constant_values = (0,0))
    
    im = Image.fromarray((255 * img).astype(np.uint8)).convert('L')
    im.save(filename)
    print(filename, img.shape)


def calc_and_save_mel_spectrogram(filename, dir_out, n_mels=128, overwrite=False):
    filename_ = os.path.splitext(filename)[0]+'.png'
    filename_ = os.path.join(dir_out, os.path.basename(filename_))
    if not overwrite and os.path.exists(filename_):
        print(f'PNG file {filename} already exists, skipping!')
        return
    y, sr = librosa.load(filename, sr=44100)
    S_mel, S_stft = mel_spectrogram(y, sr=sr, n_mels=n_mels)
    save_matrix_to_grayscale(filename_, S_mel)
    print(filename_)
    
    return y, S_mel, S_stft

    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Generate a dataset of black and white PNGs to use to train your own CycleGAN style transfer demo.')
    parser.add_argument('--in-dir',
                        help='Input directory. Contains multiple sub-folders, which in turn contain tracks from a specific sub-genre of drum and bass.'
                       )
    parser.add_argument('--out-dir', default='./output-dataset/',
                        help='Directory in which the dataset will be stored.'
                             'If non-empty and overwrite is allowed, use the -f flag.'
                       )
    parser.add_argument('--force-overwrite', action='store_true',
                        help='Allow overwriting contents of the output directory if the output directory already exists.')
    args = parser.parse_args()
    
    if os.path.exists(args.out_dir) and not args.force_overwrite:
        print(f'Output directory {args.out_dir} already exists.'
                'If files should be overwritten, then re-execute this script with the -f or --force-overwrite option.')
    
    # Create a SongCollection to keep track of all the songs that should be processed
    annotation_modules = [
        annot.BeatAnnotationWrapper(),
        annot.OnsetCurveAnnotationWrapper(),
        annot.DownbeatAnnotationWrapper(),
        annot.StructuralSegmentationWrapper(),
        annot.ReplayGainWrapper(),
        annot.KeyEstimatorWrapper(),  # Not required for this demo per se, but the current version of the autoDJ code that is used for structural segmentation also requires this in its initialization
    ]
    sc = SongCollection(annotation_modules)
    
    # Load all songs in the provided directories into the SongCollection
    for dir_ in os.listdir(args.in_dir):
        sc.load_directory(os.path.join(args.in_dir, dir_))
        print('Loaded directory {}'.format(dir_))
    sc.annotate()
    
    # Extract .wav exerpts from each track in the provided directories,
    #  time-stretching them and clipping them to an equal length.
    out_dir_wav = os.path.join(args.out_dir, 'wav')
    create_and_save_wav_extracts(sc, out_dir_wav,
                                 len_in_downbeats = 4,   # The length of each extracted segment in downbeats
                                 n_seg_per_track = 3, # The number of segments extracted per track
                                 overwrite=args.force_overwrite,
                                )
    
    # For each extract, calculate the corresponding Mel-spectrogram and save as a grayscale image
    out_dir_png = os.path.join(args.out_dir, 'img')
    wav_directories = [os.path.join(out_dir_wav, x) for x in os.listdir(out_dir_wav)]
    png_directories = [os.path.join(out_dir_png, x) for x in os.listdir(out_dir_wav)]
    for d_wav, d_png in zip(wav_directories, png_directories):
        
        # Create PNG output dir
        if not os.path.exists(d_png):
            os.makedirs(d_png)
            
        # Calculate PNG for each .wav exerpt in this directory
        for f in os.listdir(d_wav):
            if f.endswith('.wav'):
                calc_and_save_mel_spectrogram(os.path.join(d_wav, f), d_png, overwrite=args.force_overwrite)
    