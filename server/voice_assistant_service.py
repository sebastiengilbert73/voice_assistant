# Note: To add voices for pyttsx3 in Windows 11, follow the instructions in:
# https://puneet166.medium.com/how-to-added-more-speakers-and-voices-in-pyttsx3-offline-text-to-speech-812c83d14c13
import ast
import xml.etree.ElementTree as ET
import logging
import pyttsx3
from types import SimpleNamespace
import os
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoProcessor, SeamlessM4Tv2Model
import pyaudio
import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

def load_config(filepath="./voice_assistant_service_config.xml"):
    config = SimpleNamespace()
    tree = ET.parse(filepath)
    root_elm = tree.getroot()
    for root_child_elm in root_elm:
        if root_child_elm.tag == 'chat_service_url':
            config.chat_service_url = root_child_elm.text
        elif root_child_elm.tag == 'voice_id':
            config.voice_id = root_child_elm.text
        elif root_child_elm.tag == 'language_code':
            config.language_code = root_child_elm.text
        elif root_child_elm.tag == 'welcome_message':
            config.welcome_message = root_child_elm.text
        elif root_child_elm.tag == 'goodbye_message':
            config.goodbye_message = root_child_elm.text
        elif root_child_elm.tag == 'speech_to_text_model':
            config.speech_to_text_model = root_child_elm.text
        elif root_child_elm.tag == 'sampling_rate':
            config.sampling_rate = int(root_child_elm.text)
        elif root_child_elm.tag == 'record_time_in_seconds':
            config.record_time_in_seconds = float(root_child_elm.text)
        elif root_child_elm.tag == 'prompt_seed':
            config.prompt_seed = root_child_elm.text
        elif root_child_elm.tag == 'start_of_response_marker':
            config.start_of_response_marker = root_child_elm.text
        elif root_child_elm.tag == 'end_of_conversation_text':
            config.end_of_conversation_text = root_child_elm.text
        elif root_child_elm.tag == 'speech_minimum_std_dev':
            config.speech_minimum_std_dev = float(root_child_elm.text)
        elif root_child_elm.tag == 'chunk_length_in_seconds':
            config.chunk_length_in_seconds = float(root_child_elm.text)
        elif root_child_elm.tag == 'gibberish_prefix_list':
            config.gibberish_prefix_list = ast.literal_eval(root_child_elm.text)
        else:
            raise NotImplementedError(f"voice_assistant_service.load_config(): Not implemented element <{root_child_elm.tag}>")
    return config

def welcome(engine, config):
    play_message(config.welcome_message, engine, config)

def goodbye(engine, config):
    play_message(config.goodbye_message, engine, config)

def play_message(message, engine, config):
    # Set the voice
    voices = engine.getProperty('voices')
    chosen_voice_ids = []
    available_voice_ids = []
    for voice in voices:
        # logging.info(f"voice_assistant_service.main(): voice.id: {voice.id}")
        if config.voice_id.upper() in voice.id.upper():
            chosen_voice_ids.append(voice.id)

    if len(chosen_voice_ids) == 0:
        raise NotImplementedError(
            f"voice_assistant_service.play_message(): The voice id '{config.voice_id}' could not be found in the avalaible voice ids: {available_voice_ids}")
    if len(chosen_voice_ids) > 1:
        raise ValueError(
            f"voice_assistant_service.play_message(): More than one voice id could match '{config.voice_id}': {chosen_voice_ids}")
    voice_id = chosen_voice_ids[0]
    logging.info(f"voice_id = {voice_id}")

    engine.setProperty('voice', voice_id)
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    engine.say(message)
    engine.runAndWait()

def main():
    logging.info("voice_assistant_service.main()")

    config_filepath = "./voice_assistant_service_config.xml"
    if not os.path.exists(config_filepath):
        raise FileNotFoundError(f"voice_assistant_service.main(): Could not find config file '{config_filepath}'")
    config = load_config(config_filepath)

    device = "cuda" if torch.cuda.is_available() else 'cpu'

    # Setup the text-to-speech engine
    engine = pyttsx3.init()

    # Setup the speech-to-text processor
    stt_processor = None
    stt_model = None
    if config.speech_to_text_model == 'facebook/wav2vec2-large-xlsr-53-french':
        stt_processor = Wav2Vec2Processor.from_pretrained(config.speech_to_text_model)
        stt_model = Wav2Vec2ForCTC.from_pretrained(config.speech_to_text_model).to(device)
    elif config.speech_to_text_model == "facebook/seamless-m4t-v2-large":
        stt_processor = AutoProcessor.from_pretrained(config.speech_to_text_model)
        stt_model = SeamlessM4Tv2Model.from_pretrained(config.speech_to_text_model).to(device)
    else:
        raise NotImplementedError(f"voice_assistant_service.main(): Not implemented speech-to-text model '{config.speech_to_text_model}'")

    # Setup the microphone
    number_of_samples = round(config.record_time_in_seconds * config.sampling_rate)
    mic_stream = pyaudio.PyAudio().open(format=pyaudio.paInt16,
                                        channels=1,
                                        rate=config.sampling_rate,
                                        input=True,
                                        frames_per_buffer=number_of_samples)

    welcome(engine, config)

    end_of_conversation = False
    while not end_of_conversation:
        transcription = get_sentence(
            mic_stream, stt_processor, stt_model, device, config.sampling_rate,
            config
        )
        if transcription.lower().replace('.', '').replace('!', '') == config.end_of_conversation_text.lower():
            logging.info(f"voice_assistant_service.main(): End of conversation")
            end_of_conversation = True
        else:
            sentence_is_gibberish = False
            if transcription[0] == '[':
                sentence_is_gibberish = True
            for prefix in config.gibberish_prefix_list:
                if transcription.lower().startswith(prefix):
                    sentence_is_gibberish = True
            if len(transcription) > 15 and not sentence_is_gibberish:
                response = send_request_to_chat_service(config, transcription)
                logging.info(f"voice_assistant_service.main(): response = {response}")
                play_message(response, engine, config)
    goodbye(engine, config)

def get_sentence(mic_stream, stt_processor, stt_model, device, sampling_rate, config):
    chunks = []
    speech_has_started = False
    chunk_length = round(config.chunk_length_in_seconds * sampling_rate)
    #logging.info(f"voice_assistant_service.get_sentence(): chunk_length = {chunk_length}")
    while not speech_has_started:
        current_chunk_arr = np.frombuffer(mic_stream.read(chunk_length), dtype=np.int16)
        std = np.std(current_chunk_arr)
        if std >= config.speech_minimum_std_dev:
            speech_has_started = True
            chunks.append(current_chunk_arr)

    speech_has_stopped = False
    while not speech_has_stopped:
        current_chunk_arr = np.frombuffer(mic_stream.read(chunk_length), dtype=np.int16)
        std = np.std(current_chunk_arr)
        if std < config.speech_minimum_std_dev:
            chunks.append(current_chunk_arr)
            speech_has_stopped = True
        else:
            chunks.append(current_chunk_arr)

    logging.info(f"voice_assistant_service.get_sentence(): len(chunks) = {len(chunks)}")
    speech_arr = np.concatenate(chunks)
    speech_tsr = torch.from_numpy(speech_arr)

    transcription = None
    if config.speech_to_text_model == "facebook/wav2vec2-large-xlsr-53-french":
        input_values = stt_processor(speech_tsr, return_tensord="pt", sampling_rate=sampling_rate)["input_values"]
        input_tsr = torch.from_numpy(input_values[0]).to(device).unsqueeze(0)
        # Perform speech-to-text inference
        logits = stt_model(input_tsr)["logits"]
        # Use argmax to get the predicted IDs
        predicted_ids = torch.argmax(logits, dim=-1)
        # Decode the IDs to text
        transcription = stt_processor.decode(predicted_ids[0]).lower()
    elif config.speech_to_text_model == "facebook/seamless-m4t-v2-large":  # https://huggingface.co/docs/transformers/main/en/model_doc/seamless_m4t_v2
        audio_inputs = stt_processor(audios=speech_tsr, return_tensors="pt").to(device)
        output_tokens = stt_model.generate(**audio_inputs, tgt_lang=config.language_code, generate_speech=False)
        transcription = stt_processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
    else:
        raise NotImplementedError(f"voice_assistant_service.get_sentence(): Not implemented speech-to-text model '{config.speech_to_text_model}'")
    logging.info(f"voice_assistant_service.get_sentence(): transcription = {transcription}")
    return transcription


def play_audio(msg_arr):  # Cf. https://lliçons.jutge.org/upc-python-cookbook/signal-processing/audio-image.html
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=len(msg_arr.shape),
                     rate=16000, output=True)
    stream.write(msg_arr.tobytes())
    stream.stop_stream()
    stream.close()
    pa.terminate()

def send_request_to_chat_service(config, message):
    prompt = f"{config.prompt_seed}\n{message}"
    data_dict = {"prompt": prompt}
    session = requests.Session()
    result = session.post(
        config.chat_service_url,
        json=data_dict
    )
    if result.status_code != 200:
        logging.error(f"voice_assistant_service.send_request_to_chat_service(): The result status code ({result.status_code}) is not 200")

    # Find the beginning of the useful message
    response = trim_response(result.text, config.start_of_response_marker)

    return response

def trim_response(response, start_of_response_marker):
    start_of_marker = response.find(start_of_response_marker)
    if start_of_marker == -1:  # The marker was not found
        return response
    start_of_message = start_of_marker + len(start_of_response_marker)
    return response[start_of_message:]

if __name__ == '__main__':
    main()