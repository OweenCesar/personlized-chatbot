import json
import requests

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


# Audio to speech 
import os
import subprocess
from google.cloud import texttospeech



OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
LLM_MODEL = "llama3.2:3b"          # <-- matches your ollama list
INDEX_DIR = "faiss_index"


def load_vector_store():
    embeddings = OllamaEmbeddings(
        model="all-minilm:l6-v2",
        base_url="http://localhost:11434",
    )
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)


def filter_think_stream(part: str, state: dict) -> str:
    """
    Removes <think>...</think> even if tags are split across chunks.
    state = {"in_think": False}
    """
    out = []
    i = 0
    while i < len(part):
        if not state["in_think"]:
            start = part.find("<think>", i)
            if start == -1:
                out.append(part[i:])
                break
            out.append(part[i:start])
            state["in_think"] = True
            i = start + len("<think>")
        else:
            end = part.find("</think>", i)
            if end == -1:
                break
            state["in_think"] = False
            i = end + len("</think>")
    return "".join(out)


def ask_llama(question: str, context: str) -> str:
    system = (
        "You are a simulated candidate speaking in first person, based ONLY on the resume context provided.\n"
        "Rules:\n"
        "- Use only facts from the resume context.\n"
        "- If the context does not contain the answer, say: \"I don't have that information in my resume.\"\n"
        "- Keep it formal and recruiter-friendly.\n"
        "- Do not say you are an AI model.\n"
        "- Do not output <think> or any reasoning.\n"
        "If the user is just greeting, respond with a brief greeting and ask what they'd like to know.\n"
    )

    r = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": LLM_MODEL,
            "stream": True,
            "options": {
                "num_predict": 220,
                "temperature": 0.2,
                "num_ctx": 4096,   # helps when context is longer
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": f"RESUME CONTEXT:\n{context}\n\nQUESTION:\n{question}"},
            ],
        },
        stream=True,
        timeout=300,
    )
    r.raise_for_status()

    print("\nCandidate: ", end="", flush=True)
    state = {"in_think": False}
    full_clean = ""

    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        chunk = json.loads(line)
        part = chunk.get("message", {}).get("content", "")

        if part:
            clean = filter_think_stream(part, state)  # <-- APPLY FILTER
            if clean:
                full_clean += clean
                print(clean, end="", flush=True)

        if chunk.get("done"):
            break

    print("\n")
    return full_clean.strip()


def build_context(docs):
    parts = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        page = meta.get("page", "?")
        parts.append(f"[Source {i} | page {page}]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def tts_google_ogg_opus(text: str, out_path: str = "answer.ogg", lang: str = "en-US"):
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code=lang,
        # You can optionally set name="en-US-Neural2-J" etc later
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.OGG_OPUS
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config,
    )

    with open(out_path, "wb") as f:
        f.write(response.audio_content)

    return out_path



def main():
    vs = load_vector_store()
    print("✅ Resume chat ready. Type a question. Type 'exit' to quit.\n")

    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break

        docs = vs.similarity_search(q, k=4)
        context = build_context(docs)

        answer = ask_llama(q, context)

        # TTS + play
        out_file = tts_google_ogg_opus(answer, out_path="answer.ogg", lang="en-US")
        print(f"🔊 Saved voice: {out_file}")

        # Play it (requires ffmpeg installed)
        subprocess.run(["ffplay", "-nodisp", "-autoexit", out_file], check=False)

        print("\n---\n")



if __name__ == "__main__":
    main()
