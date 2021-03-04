# Liquid/Dancefloor Drum and Bass Style Transfer using CycleGAN

This repository contains some of the code that I used in the application of the [CycleGAN model (in PyTorch)](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) to music style transfer for drum and bass.
More specifically, it contains a script to create a dataset in the same way as done in the paper,
and a Jupyter Notebook containing a pretrained model and a demo.
Code for training can be found in the [PyTorch CycleGAN and pix2pix repository](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

You can find our paper on this dnb style transfer system here:

_[Vande Veire, L., De Bie, T., & Dambre, J. (2019). A CycleGAN for style transfer between drum and bass subgenres. In ML4MD at ICML2019, Proceedings. Long Beach, CA.](https://biblio.ugent.be/publication/8619952))_



## Dataset generation

The ``dnb-cyclegan-create-dataset.py`` script allows you to create your own dataset for training a CycleGAN model.
The script expects as its input a directory, containing two subdirectories, one for each source genre, each containing a few full drum and bass tracks in that genre.
It extracts short snippets from each track, converts them to the required spectrogram representation, and saves the resulting snippets and spectrograms in a separate output directory.

Note that the segmentation of the tracks into segments happens automatically and might be imperfect (even though it should work for the majority of drum and bass tracks, see [our paper on the autoDJ system](https://asmp-eurasipjournals.springeropen.com/articles/10.1186/s13636-018-0134-8) for which this code was initially developed).
The script also expects the tracks to be _drum and bass_ music, so using it for other music genres will require some modifications (you could also adapt the script and extract your own segments from the songs manually).
Check whether the extracted segments align with the music as expected before training the model!

From the paper: 

_The spectrograms for training the CycleGAN model are
extracted from 40 liquid and 40 dancefloor drum & bass
songs. From each song, 3 segments of 4 downbeats (approximately 5.5 seconds)
are selected, resulting in a total of 240
segments for training. The segments are downbeat-aligned
and selected from the ‘main’ part of the song (similar to the
chorus for vocal-based music). The tempo of each song is
stretched to 175 BPM using WSOLA time stretching to ensure a 
consistent length for training. Note that after training, the model
can be applied to extracts in any tempo._



## Training

The code for training your own CycleGAN model in PyTorch can be found here: [link](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).  
The command that I used for training is the following: `python train.py --dataroot ./datasets/dnb --name dnb --model cycle_gan --display_port 6006 --resize_or_crop none --input_nc 1 --output_nc 1`


## Demo

The demo (implemented as a Jupyter Notebook) allows you to apply a pretrained model to your own liquid or dancefloor drum and bass songs.

The required Python packages are in the `requirements.txt` file, install using `pip install -r requirements.txt`.
Then launch the Jupyter notebook environment using `jupyter notebook` in the demo directory, open the `Liquid and dancefloor style transfer.ipynb` Notebook, run all the cells, and enjoy! B)


## Examples

Examples of what the output of the demo sounds like, can be found on this website: [link](https://users.ugent.be/~levdveir/2019ML4MD/).
