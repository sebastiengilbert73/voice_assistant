# Cf. https://huggingface.co/docs/transformers/main/en/model_doc/seamless_m4t_v2
from transformers import AutoProcessor, SeamlessM4Tv2Model
import torchaudio
import logging
import pyaudio
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

def main():
    logging.info("seamless-m4t-v2.main()")

    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

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
    #print(speech_arr)
    logging.info(f"type(speech_arr) = {type(speech_arr)}; speech.shape = {speech_arr.shape}")

    speech_tsr = torch.from_numpy(speech_arr)
    logging.info(f"speech_tsr.shape = {speech_tsr.shape}")

    audio_inputs = processor(audios=speech_tsr, return_tensors="pt")
    output_tokens = model.generate(**audio_inputs, tgt_lang="fra", generate_speech=False)

    logging.info(f"type(audio_array_from_audio) = {type(output_tokens)}")
    translated_text_from_audio = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)

    logging.info(f"translated_text_from_audio = {translated_text_from_audio}")

if __name__ == '__main__':
    main()