# Note: To add voices for pyttsx3 in Windows 11, follow the instructions in:
# https://puneet166.medium.com/how-to-added-more-speakers-and-voices-in-pyttsx3-offline-text-to-speech-812c83d14c13
import xml.etree.ElementTree as ET
import logging
import pyttsx3
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import os

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
        elif root_child_elm.tag == 'welcome_message':
            config.welcome_message = root_child_elm.text
        else:
            raise NotImplementedError(f"voice_assistant_service.load_config(): Not implemented element <{root_child_elm.tag}>")
    return config

def main():
    logging.info("voice_assistant_service.main()")

    config_filepath = "./voice_assistant_service_config.xml"
    if not os.path.exists(config_filepath):
        raise FileNotFoundError(f"voice_assistant_service.main(): Could not find config file '{config_filepath}'")
    config = load_config(config_filepath)

    engine = pyttsx3.init()
    # Set the voice
    voices = engine.getProperty('voices')
    chosen_voice_ids = []
    available_voice_ids = []
    for voice in voices:
        #logging.info(f"voice_assistant_service.main(): voice.id: {voice.id}")
        if config.voice_id.upper() in voice.id.upper():
            chosen_voice_ids.append(voice.id)

    if len(chosen_voice_ids) == 0:
        raise NotImplementedError(f"The voice id '{config.voice_id}' could not be found in the avalaible voice ids: {available_voice_ids}")
    if len(chosen_voice_ids) > 1:
        raise ValueError(f"More than one voice id could match '{config.voice_id}': {chosen_voice_ids}")
    voice_id = chosen_voice_ids[0]
    logging.info(f"voice_id = {voice_id}")

    engine.setProperty('voice', voice_id)
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    engine.say(config.welcome_message)
    engine.runAndWait()


if __name__ == '__main__':
    main()