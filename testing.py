from google.cloud import texttospeech

client = texttospeech.TextToSpeechClient()

text = "Hello! This is a test of Google Text to Speech. Have a great day! 1 2 3 4 5."
synthesis_input = texttospeech.SynthesisInput(text=text)

voice = texttospeech.VoiceSelectionParams(
    language_code="en-US",
)

audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.OGG_OPUS
)

response = client.synthesize_speech(
    input=synthesis_input,
    voice=voice,
    audio_config=audio_config,
)

with open("answer.ogg", "wb") as out:
    out.write(response.audio_content)

print("Saved answer.ogg")




























 



















