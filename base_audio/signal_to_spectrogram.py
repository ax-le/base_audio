# -*- coding: utf-8 -*-
"""
Created on July 2024, based on old code.

@author: a23marmo

Computing spectrogram in different feature description.

Note that Mel (and variants of Mel) spectrograms follow the particular definition of [1].

Spectrogram computation is done with the toolbox librosa [2].

References
----------
[1] Grill, T., & Schlüter, J. (2015, October). 
Music Boundary Detection Using Neural Networks on Combined Features and Two-Level Annotations. 
In ISMIR (pp. 531-537).

[2] McFee, B., et al. (2024). librosa/librosa: 0.10.2.post1 (0.10.2.post1). 
Zenodo. https://doi.org/10.5281/zenodo.11192913
"""

import numpy as np
import librosa.core
import librosa.feature
import librosa.effects
from math import inf
import base_audio.errors as err
import warnings

mel_power = 2 # Power of the mel spectrogram

class FeatureObject():
    """
    FeatureObject class, which computes different types of spectrograms.
    All these spectrograms are computed with the toolbox librosa [2].
    """

    def __init__(self, sr, feature, hop_length, n_fft=2048, fmin = 0, fmax=None, mel_grill = True, n_mels=80):
        """
        Constructor of the FeatureObject class.

        Parameters
        ----------
        sr : float
            Sampling rate of the signal, (typically 44100Hz).
        feature : String
            The types of spectrograms to compute:
                - "pcp" : Pitch Class Profile
                - "cqt" : Constant-Q Transform
                - "mel" : Mel spectrogram
                - "log_mel" : Log Mel spectrogram
                - "nn_log_mel" : Nonnegative Log Mel spectrogram (i.e. log (mel + 1))
                - "padded_log_mel" : Padded Log Mel spectrogram (i.e. log mel + min(log mel))
                - "minmax_log_mel" : Min-Max Log Mel spectrogram (i.e. (log mel + min(log mel)) / max(log mel)
                - "stft" : Short-Time Fourier Transform
                - "stft_complex" : Complex Short-Time Fourier Transform
        hop_length : integer
            The desired hop_length, which is the step between two frames (ie the time "discretization" step)
            It is expressed in terms of number of samples, which are defined by the sampling rate.
        n_fft : integer, optional
            The number of samples to use in each frame. 
            The default is 2048.
        fmin, fmax : integer, optional
            The minimal (resp. maximal) frequence to consider, used for denoising.
            The default is 98 (resp. None, i.e. no maximal frequency).
        mel_grill : boolean, optional
            If True, the mel spectrogram is computed with the same parameters as the ones of [1].
            The default is True.
        n_mels : integer, optional
            Number of mel bands to consider.
            The default is 80.
        """
        self.sr = sr
        self.feature = feature.lower()
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.fmin = fmin
        self.fmax = fmax
        self.mel_grill = mel_grill
        self.n_mels = n_mels

        match self.feature:
            case "pcp":
                self.frequency_dimension = 12
            case "cqt":
                self.frequency_dimension = 84
            case "mel" | "log_mel" | "nn_log_mel" | "padded_log_mel" | "minmax_log_mel":
                self.frequency_dimension = self.n_mels
            case "stft" | "stft_complex":
                self.frequency_dimension = self.n_fft // 2 + 1
            case _:
                raise err.InvalidArgumentValueException(f"Unknown signal representation: {self.feature}.")

    # TODO: add MFCC, maybe tonnetz
    def get_spectrogram(self, signal):
        """
        Returns a spectrogram, from the signal of a song.
        Different types of spectrogram can be computed, which are specified by the argument "feature" of the object.
        All these spectrograms are computed with the toolbox librosa [2].
        
        Parameters
        ----------
        signal : numpy array
            Signal of the song.
        Other parameters : see the FeatureObject class.

        Returns
        -------
        numpy array
            Spectrogram of the signal.
        """
        match self.feature:
            case "pcp":
                return self._compute_pcp(signal)
        
            case "cqt":
                return self._compute_cqt(signal)
        
            # For Mel spectrograms, by default we use the same parameters as the ones of [1].
            case "mel":
                return self._compute_mel_spectrogram(signal)
        
            case "log_mel" | "nn_log_mel" | "padded_log_mel" | "minmax_log_mel":
                mel_spectrogram = self._compute_mel_spectrogram(signal)
                return get_log_mel_from_mel(mel_spectrogram, self.feature)
            
            case "stft":
                return self._compute_stft(signal, complex = False)
            case "stft_complex":
                return self._compute_stft(signal, complex = True)
        
            case _:
                raise err.InvalidArgumentValueException(f"Unknown signal representation: {self.feature}.")
        
    def _compute_pcp(self, signal):
        norm=inf # Columns normalization
        win_len_smooth=82 # Size of the smoothign window
        n_octaves=6
        bins_per_chroma = 3
        bins_per_octave=bins_per_chroma * 12
        fmin=self.fmin
        return librosa.feature.chroma_cens(y=signal,sr=self.sr,hop_length=self.hop_length,
                                    fmin=fmin, n_chroma=12, n_octaves=n_octaves, bins_per_octave=bins_per_octave,
                                    norm=norm, win_len_smooth=win_len_smooth)

    def _compute_cqt(self, signal):
        constant_q_transf = librosa.cqt(y=signal, sr = self.sr, hop_length = self.hop_length)
        return np.abs(constant_q_transf)

    def _compute_mel_spectrogram(self, signal):
        if self.mel_grill:
            mel = librosa.feature.melspectrogram(y=signal, sr = self.sr, n_fft=2048, hop_length = self.hop_length, n_mels=80, fmin=80.0, fmax=16000, power=2)
        else:
            mel = librosa.feature.melspectrogram(y=signal, sr = self.sr, n_fft=self.n_fft, hop_length = self.hop_length, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax, power=mel_power)
        return np.abs(mel)
        
    def _compute_stft(self, signal, complex):
        stft = librosa.stft(y=signal, hop_length=self.hop_length,n_fft=self.n_fft)
        if complex:
            mag, phase = librosa.magphase(stft, power = 1)
            return mag, phase
        else:
            return np.abs(stft)
        
    def get_stft_from_mel(self, mel_spectrogram, feature=None):
        if feature is None: # Recursive function, so it takes the feature as an argument
            feature = self.feature # Default case takes the object feature as the feature to compute

        match feature: 
            case "mel":
                if self.mel_grill:
                    return librosa.feature.inverse.mel_to_stft(M=mel_spectrogram, sr=self.sr, n_fft=2048, power=2, fmin=80.0, fmax=16000) # Fixed as such
                else:
                    return librosa.feature.inverse.mel_to_stft(M=mel_spectrogram, sr=self.sr, n_fft=self.n_fft, power=mel_power, fmin=self.fmin, fmax=self.fmax)
        
            case "log_mel":
                mel = librosa.db_to_power(S_db=mel_spectrogram, ref=1)
                return self.get_stft_from_mel(mel, "mel")

            case "nn_log_mel":
                mel = librosa.db_to_power(S_db=mel_spectrogram, ref=1) - np.ones(mel_spectrogram.shape)
                return self.get_stft_from_mel(mel, "mel")

            case _:
                raise err.InvalidArgumentValueException("Unknown feature representation.")
    
    def load_file_compute_spectrogram(self, file_path):
        """
        A wrapper to load a file and compute the spectrogram.
        """
        signal, original_sr = librosa.load(file_path, sr=44100, mono=True)
        if original_sr != self.sr:
            signal = librosa.resample(signal, original_sr, self.sr)
        return self.get_spectrogram(signal)

def get_log_mel_from_mel(mel_spectrogram, feature):
    """
    Computes a variant of a Mel spectrogram (typically Log Mel).

    Parameters
    ----------
    mel_spectrogram : numpy array
        Mel spectrogram of the signal.
    feature : string
        Desired feature name (must be a variant of a Mel spectrogram).

    Raises
    ------
    err.InvalidArgumentValueException
        Raised in case of unknown feature name.

    Returns
    -------
    numpy array
        Variant of the Mel spectrogram of the signal.

    """
    match feature:
        case "log_mel":
            return librosa.power_to_db(np.abs(mel_spectrogram), ref=1)
    
        case "nn_log_mel":
            mel_plus_one = np.abs(mel_spectrogram) + np.ones(mel_spectrogram.shape)
            nn_log_mel = librosa.power_to_db(mel_plus_one, ref=1)
            return nn_log_mel
    
        case "padded_log_mel":
            log_mel = get_log_mel_from_mel(mel_spectrogram, "log_mel")
            return log_mel - np.amin(log_mel) * np.ones(log_mel.shape)
        
        case "minmax_log_mel":        
            padded_log_mel = get_log_mel_from_mel(mel_spectrogram, "padded_log_mel")
            return np.divide(padded_log_mel, np.amax(padded_log_mel))
    
        case _:
            raise err.InvalidArgumentValueException("Unknown feature representation.")

def get_spectrogram(signal, sr, feature, hop_length, fmin = 98):
    """
    Old behavior: calling a function. Deprecated, but still available in order to avoid breaking the code.
    """
    warnings.warn("This function is deprecated. Please use the FeatureObject class instead.")
    feature_object = FeatureObject(sr, feature, hop_length, n_fft=2048, fmin = fmin, fmax=None, mel_grill = True, n_mels=80)
    return feature_object.get_spectrogram(signal)