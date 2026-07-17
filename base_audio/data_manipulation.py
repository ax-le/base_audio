# -*- coding: utf-8 -*-
"""
Created on July 2026

@author: a23marmo

Generic conversions and helpers for time/frequency-indexed audio data (spectrograms, activation
matrices, segmentations...), shared across projects relying on base_audio's FeatureObject
(see signal_to_spectrogram.py).
"""

import numpy as np
import math

# =========================================================================
# Time conversions (frames and time)
# =========================================================================

def _sample_index_to_time_in_seconds(sample_index, sr):
    """
    Convert a raw audio sample index to time in seconds, based on the sample rate (sr).
    This is a low-level, feature-independent conversion: it knows nothing about
    hop_length or feature frames, only about the sample rate of the underlying signal.
    Works with scalars only.

    Parameters
    ----------
    sample_index : int or float
        Raw sample index
    sr : int
        Sample rate in Hz

    Returns
    -------
    time_in_seconds : float
        Time in seconds
    """
    return float(sample_index) / float(sr)

def _time_in_seconds_to_sample_index(time_in_seconds, sr):
    """
    Convert time in seconds to the nearest raw audio sample index, based on the sample rate (sr).
    This is a low-level, feature-independent conversion: it knows nothing about
    hop_length or feature frames, only about the sample rate of the underlying signal.
    Works with scalars only. Inverse of _sample_index_to_time_in_seconds.

    Parameters
    ----------
    time_in_seconds : float
        Time value in seconds
    sr : int
        Sample rate in Hz

    Returns
    -------
    sample_index : int
        Sample index (rounded to nearest integer)
    """
    return int(np.round(time_in_seconds * sr))

def _samples_per_feature_frame(feature_object):
    """
    Compute the number of raw audio samples spanned by a single feature frame
    (i.e. by one column of the feature matrix), based on the feature type.

    This is the only place where the feature-dependent frame duration is computed,
    so that frame_to_second and second_to_frame stay simple and consistent.

    - For LTSA features (identified by 'ltsa' in the feature name), a frame aggregates
    an integer number of STFT hops. This number is derived from ltsa_time_per_frame
    (the requested LTSA frame duration), the sample rate (sr) and the hop_length.
    If ltsa_time_per_frame is smaller than the actual STFT hop duration, a single
    STFT hop is used instead (a frame cannot be smaller than one hop_length).
    - For non-LTSA features, a frame is simply one STFT hop, i.e. hop_length samples.
    """
    feature_name = feature_object.feature.lower() if isinstance(feature_object.feature, str) else ""

    if feature_name in {"ltsa", "ltsa_pcen"}:
        if feature_object.ltsa_time_per_frame is None:
            raise ValueError("ltsa_time_per_frame must be provided for LTSA features")
        # LTSA frames are built by aggregating an integer number of STFT columns.
        ltsa_samples_per_frame = int(float(feature_object.ltsa_time_per_frame) * float(feature_object.sr))
        cols_per_ltsa_frame = max(1, ltsa_samples_per_frame // int(feature_object.hop_length))
        return cols_per_ltsa_frame * int(feature_object.hop_length)
    else:
        # Non-LTSA features use standard STFT frame timing.
        return int(feature_object.hop_length)

def frame_to_second(frame_index, feature_object):
    """
    Convert a feature frame index to time in seconds based on the feature type.
    Works with scalar frame indices only.

    A feature frame spans _samples_per_feature_frame(feature_object) raw audio samples,
    so the conversion goes through the low-level sample-to-time conversion.
    See _samples_per_feature_frame for how this frame duration depends on the feature type.

    Parameters
    ----------
    frame_index : int
        Feature frame index
    feature_object : FeatureObject
        Feature object containing sr, hop_length, feature type, and optionally ltsa_time_per_frame

    Returns
    -------
    time_in_seconds : float
        Time in seconds
    """
    samples_per_frame = _samples_per_feature_frame(feature_object)
    sample_index = frame_index * samples_per_frame
    return _sample_index_to_time_in_seconds(sample_index, feature_object.sr)

def second_to_frame(time_event, feature_object):
    """
    Convert time in seconds to a feature frame index. Inverse of frame_to_second.
    Works with scalar time values only.
    See frame_to_second and _samples_per_feature_frame for the details of the
    conversion, which depends on the feature type.

    Parameters
    ----------
    time_event : float
        Time in seconds
    feature_object : FeatureObject
        Feature object containing sr, hop_length, feature type, and optionally ltsa_time_per_frame

    Returns
    -------
    frame_index : int
        Feature frame index
    """
    samples_per_frame = _samples_per_feature_frame(feature_object)
    sample_index = _time_in_seconds_to_sample_index(time_event, feature_object.sr)
    return int(np.round(sample_index / samples_per_frame))


# =========================================================================
# FREQUENCY BINS
# =========================================================================

def frequency_bins_to_hz(feature_object):
    """Convert frequency bin indices to hertz based on the feature configuration."""
    match feature_object.feature:
        case "stft" | "stft_complex" | "pcen" | "ltsa" | "ltsa_pcen":
            freq_bins = np.arange(feature_object.frequency_dimension)
            freq_hz = (freq_bins * feature_object.sr) / feature_object.n_fft
            return freq_hz

        case _:
            raise ValueError(f"Unsupported feature type: {feature_object.feature}")

def hz_to_frequency_bin(freq, feature_object):
    """Convert a frequency in hertz to the closest frequency bin index. Inverse of frequency_bins_to_hz."""
    match feature_object.feature:
        case "stft" | "stft_complex" | "pcen" | "ltsa" | "ltsa_pcen":
            freq_bin = int(np.rint(freq * feature_object.n_fft / feature_object.sr))
            if freq_bin < 0 or freq_bin >= feature_object.frequency_dimension:
                raise ValueError(f"Frequency {freq} Hz is out of bounds for sampling rate {feature_object.sr} Hz.")
            return freq_bin

        case _:
            raise ValueError(f"Unsupported feature type: {feature_object.feature}")

def freq_to_midi(frequency):
    """
    Returns the frequency (Hz) in the MIDI scale

    Parameters
    ----------
    frequency: float
        Frequency in Hertz

    Returns
    -------
    midi_f0: integer
        Frequency in MIDI scale
    """
    return int(round(69+ 12 * math.log(frequency/440,2)))

def midi_to_freq(midi_freq):
    """
    Returns the MIDI frequency in Hertz

    Parameters
    ----------
    midi_freq: integer
        Frequency in MIDI scale

    Returns
    -------
    frequency: float
        Frequency in Hertz
    """
    return 440 * 2**((midi_freq - 69)/12)
