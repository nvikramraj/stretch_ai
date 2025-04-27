import sounddevice as sd
import numpy as np
import whisper
import torch
import time



class speech_to_text():

    def __init__(self):

        # Load Whisper model
        self.model = whisper.load_model("base")

        # Settings
        self.sample_rate = 16000
        self.chunk_duration = 0.5  # seconds
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        self.silence_threshold = 0.1  # adjust depending on your mic
        self.silence_duration = 2.0  # seconds
        self.max_recording = 10  # max recording time in seconds
        self.recorded_audio = []
        self.silence_start = None
        self.start_time = time.time()
        pass

        

    def is_silent(self,chunk, threshold):
        return np.sqrt(np.mean(chunk**2)) < threshold
    
    def record_audio(self):

        print("Recording")
        self.recorded_audio = []
        self.silence_start = None
        self.start_time = time.time()
        # Stream audio in chunks
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32', blocksize=self.chunk_size) as stream:
            while True:
                chunk, _ = stream.read(self.chunk_size)
                chunk = np.squeeze(chunk)
                self.recorded_audio.append(chunk)

                silent = self.is_silent(chunk, self.silence_threshold)

                if silent:
                    if self.silence_start is None:
                        self.silence_start = time.time()
                    elif time.time() - self.silence_start > self.silence_duration:
                        print("Silence detected. Stopping...")
                        break
                else:
                    self.silence_start = None

                if time.time() - self.start_time > self.max_recording:
                    print("Max time reached. Stopping...")
                    break

            # Concatenate audio and preprocess
            audio = np.concatenate(self.recorded_audio)
        
        return audio

    def speech_to_text(self, audio):
        # Whisper expects 1D mono float32 audio at 16kHz
        audio_tensor = torch.from_numpy(audio)
        audio_tensor = whisper.pad_or_trim(audio_tensor)
        mel = whisper.log_mel_spectrogram(audio_tensor).to(self.model.device)
        options = whisper.DecodingOptions(language="en", fp16=False)
        result = whisper.decode(self.model, mel, options)

        print("Transcribed text:")
        print(result.text)
        return result.text

