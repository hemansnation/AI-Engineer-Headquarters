def ask_llm(messages: list[dict], model: str) -> str:
    try:
        import ollama
        response = ollama.chat(model=model, messages=messages, options={'temperature': 0})
    except Exception as error:
        raise RuntimeError(
            f'Could not reach Ollama. Start it with ollama serve, then run ollama pull {model}'
        ) from error
    return response['message']['content'].strip()