"""
    audio_test.py

    This programs collects audio data from an I2S mic on the Raspberry Pi 
    and saves it in a WAV file.

    Author: Mahesh Venkitachalam
    Website: electronut.in

"""

import pyaudio
import numpy as np

import wave
import argparse

from tflite_runtime.interpreter import Interpreter

import scipy.io.wavfile as wavfile
from scipy.io.wavfile import write

import librosa
import librosa.feature
import librosa.display

import cv2

import clarity
from clarity.evaluator import haspi
from clarity.utils.audiogram import Audiogram

import time

def snr_db(clean, test):
    """Compute SNR in dB."""
    noise = clean - test
    return 10 * np.log10(np.mean(clean**2) / (np.mean(noise**2) + 1e-12))

def denoise_full_spectrogram(noisy_spec_db, interpreter, inputs, outputs, alpha=1.0):
    """Denoise a full spectrogram in overlapping 128-frame chunks."""
    hop_frames = 64
    n_frames = noisy_spec_db.shape[1]
    clean_spec_db = np.zeros_like(noisy_spec_db)
    weight = np.zeros_like(noisy_spec_db)

    for start in range(0, n_frames, hop_frames):
        end = start + 128
        patch = noisy_spec_db[:, start:end]

        if patch.shape[1] < 128:
            pad_width = 128 - patch.shape[1]
            patch = np.pad(patch, ((0, 0), (0, pad_width)),
                           mode='constant', constant_values=-46)

        patch_scaled = (patch + 46) / 50
        patch_scaled = patch_scaled.reshape(1, 128, 128, 1).astype(np.float32)

        interpreter.set_tensor(inputs[0]['index'], patch_scaled)
        interpreter.invoke()
        output_scaled = interpreter.get_tensor(outputs[0]['index'])

        predicted_noise_db = output_scaled.squeeze() * 82 + 6
        clean_patch_db = patch - alpha * predicted_noise_db

        if end > n_frames:
            clean_patch_db = clean_patch_db[:, :n_frames - start]

        clean_spec_db[:, start:start+clean_patch_db.shape[1]] += clean_patch_db
        weight[:, start:start+clean_patch_db.shape[1]] += 1

    weight[weight == 0] = 1
    clean_spec_db /= weight
    return clean_spec_db

def clean_existing_data():
    model_path = "/home/coris/model_unet_best_9_10.tflite"
    input_audio = "/home/coris/cleaned_A.wav"
    clean_ref_audio = "/home/coris/cleaned_A.wav"

    interpreter = Interpreter(model_path)
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    print(f"Model loaded from: {model_path}")

    y, sr = librosa.load(input_audio, sr=8000)
    
    # start the latency timer!
    start_time = time.time()
    
    S_complex = librosa.stft(y, n_fft=255, hop_length=63)
    S_mag, S_phase = librosa.magphase(S_complex)
    noisy_spec_db = librosa.amplitude_to_db(S_mag, ref=np.max)

    clean_spec_db = denoise_full_spectrogram(noisy_spec_db, interpreter, inputs, outputs, alpha=1.0)

    clean_spec_amp = librosa.db_to_amplitude(clean_spec_db)
    clean_stft = clean_spec_amp * np.exp(1j * np.angle(S_phase))
    reconstructed = librosa.istft(clean_stft, hop_length=63)

    max_val = np.max(np.abs(reconstructed))
    if max_val > 0:
        reconstructed = reconstructed / max_val
    
    # end the latency timer!
    end_time = time.time()
    latency_sec = end_time - start_time
    print(f"Denoising latency: {latency_sec:.3f} seconds")

    write("cleaned_test_11_sept.wav", sr, (reconstructed * 32767).astype(np.int16))
    print("Saved cleaned audio to cleaned_test_11_sept.wav")

    try:
        clean_ref_waveform, _ = librosa.load(clean_ref_audio, sr=sr)

        # 🔹 Sanitize
        clean_ref_waveform = np.asarray(clean_ref_waveform, dtype=np.float64).flatten()
        reconstructed = np.asarray(reconstructed, dtype=np.float64).flatten()
        sr = int(sr)

        min_len = min(len(clean_ref_waveform), len(reconstructed))
        clean_ref_waveform = clean_ref_waveform[:min_len]
        reconstructed = reconstructed[:min_len]

        audiogram = Audiogram(
            frequencies=[250, 500, 1000, 2000, 4000, 6000],
            levels=[0, 0, 0, 0, 0, 0]
        )

        # 🔹 Debug info
        print("\n--- DEBUG: Metric Inputs ---")
        print(f"clean_ref_waveform: shape={clean_ref_waveform.shape}, dtype={clean_ref_waveform.dtype}, first5={clean_ref_waveform[:5]}")
        print(f"reconstructed:      shape={reconstructed.shape}, dtype={reconstructed.dtype}, first5={reconstructed[:5]}")
        print(f"sr: type={type(sr)}, value={sr}")
        print(f"audiogram: {audiogram}")
        print("----------------------------\n")

        # Metrics
        snr_val = snr_db(reconstructed, clean_ref_waveform)
        haspi_val, _ = haspi.haspi_v2(clean_ref_waveform, float(sr), reconstructed, float(sr), audiogram)

        print(f"📊 Audio Quality Metrics:")
        print(f"   SNR   = {snr_val:.2f} dB")
        print(f"   HASPI = {haspi_val:.3f}")

    except FileNotFoundError:
        print("⚠️ Clean reference file not found — skipping SNR/HASPI metrics.")
    except Exception as e:
        print(f"⚠️ Could not compute HASPI/SNR: {e}")

# get pyaudio input device
def list_input_devices(p):
    nDevices = p.get_device_count()
    print('Found %d devices:' % nDevices)
    for i in range(nDevices):
        deviceInfo = p.get_device_info_by_index(i)
        print(deviceInfo)
        devName = deviceInfo['name']
        print(devName)

def main():

    # create parser
    descStr = """
    This program collects audio data from an I2S mic and saves to a WAV file.
    """
    parser = argparse.ArgumentParser(description=descStr)

    # add expected arguments
    parser.add_argument('--output', dest='wavfile_name', required=False)
    parser.add_argument('--nsec', dest='nsec', required=False)
    parser.add_argument('--list', action='store_true', required=False)
    parser.add_argument('--ai', dest='ai', required=False)

    # parse args
    args = parser.parse_args()

    # set defaults
    wavfile_name = 'out.wav'
    nsec = 1
    # set args
    if args.ai:
        clean_existing_data()
        exit(0)
    if args.wavfile_name:
        wavfile_name = args.wavfile_name
    if args.nsec:
        nsec = int(args.nsec)
    if args.list:
        # list devices
        print("Listing devices...")
        # initialize pyaudio
        p = pyaudio.PyAudio()
        list_input_devices(p)
        p.terminate()
        print("done.")
        exit(0)

    CHUNK = 4096
    FORMAT = pyaudio.paInt32
    CHANNELS = 1
    RATE = 48000
    RECORD_SECONDS = nsec
    WAVE_OUTPUT_FILENAME = wavfile_name
    NFRAMES = int((RATE * RECORD_SECONDS) / CHUNK)

    # initialize pyaudio
    p = pyaudio.PyAudio()

    print('opening stream...') # format = FORMAT
    stream = p.open(format = FORMAT,
                    channels = CHANNELS,
                    rate = RATE,
                    input = True,
                    frames_per_buffer = CHUNK,
                    input_device_index = 1)

    frames = []

    # discard first 1 second
    for i in range(0, NFRAMES):
        data = stream.read(CHUNK)

    print("Collecting data for %d seconds in %s..." % (nsec, wavfile_name))
    print("start recording!")

    for i in range(0, NFRAMES):
        data = stream.read(CHUNK)
        #print(data)
        frames.append(data)

    # TODO: run the sound we just got through the AI model and save the new version as <audio name>_clean.wav

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print("done.")

# main method
if __name__ == '__main__':
    main()
