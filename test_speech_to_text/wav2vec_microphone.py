# Cf. https://huggingface.co/facebook/wav2vec2-large-xlsr-53-french
# Cf. https://thepythoncode.com/article/speech-recognition-using-huggingface-transformers-in-python
import logging
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import pyaudio
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("wav2vec_microphone.main()")

    model_name = "facebook/wav2vec2-large-xlsr-53-french"
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
    processor = Wav2Vec2Processor.from_pretrained(model_name)

    sampling_rate = 16000
    record_time_in_seconds = 5.0
    number_of_samples = round(record_time_in_seconds * sampling_rate)
    mic_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16,
                            channels=1,
                            rate=sampling_rate,
                            input=True,
                            frames_per_buffer=number_of_samples)

    # Get audio
    logging.info("Speak!")
    speech_arr = np.frombuffer(mic_stream.read(number_of_samples), dtype=np.int16)
    print(speech_arr)
    logging.info(f"type(speech_arr) = {type(speech_arr)}; speech.shape = {speech_arr.shape}")

    speech_tsr = torch.from_numpy(speech_arr)
    logging.info(f"speech_tsr.shape = {speech_tsr.shape}")

    # Tokenize our tensor
    input_values = processor(speech_tsr, return_tensord="pt", sampling_rate=sampling_rate)["input_values"]
    logging.info(f"type(input_values) = {type(input_values)}")
    logging.info(f"len(input_values) = {len(input_values)}")
    logging.info(f"type(input_values[0]) = {type(input_values[0])}")
    logging.info(f"input_values[0].shape = {input_values[0].shape}")
    input_tsr = torch.from_numpy(input_values[0]).to(device).unsqueeze(0)
    logging.info(f"input_tsr.shape = {input_tsr.shape}")  # (1, N)

    # Perform inference
    logits = model(input_tsr)["logits"]
    logging.info(f"logits.shape = {logits.shape}")  # (1, 5123, 49)
    # Use argmax to get the predicted IDs
    predicted_ids = torch.argmax(logits, dim=-1)
    logging.info(f"type(predicted_ids) = {type(predicted_ids)}; predicted_ids.shape = {predicted_ids.shape}")

    # Decode the IDs to text
    transcription = processor.decode(predicted_ids[0]).lower()
    logging.info(f"transcription = {transcription}")

if __name__ == '__main__':
    main()