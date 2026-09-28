import sys

from llm import MODEL, ask
from tracker import running_totals

HINTS = {
    400: "Bad Request: The server could not understand the request due to invalid syntax.",
    401: "Unauthorized: The client must authenticate itself to get the requested response.",
    403: "Forbidden: The client does not have access rights to the content.",
    404: "Not Found: The server can not find the requested resource.",
    429: "Too Many Requests: The user has sent too many requests in a given amount of time.",
}

def explain_error(error):
    status = getattr(error, "status_code", None)
    if status is None:
        return f"No response from gemini"
    if status >= 500:
        hint = "gemini had a problem, please try again later."
    else:
        hint = HINTS.get(status, "An error occurred.")
    return f"Error {status}: {hint}"

def main():
    prompt = " ".join(sys.argv[1:]) or "explain what an API is in one sentence"

    try:
        interaction = ask(prompt)
    except Exception as error:
        print(explain_error(error))
        sys.exit(1)

    usage = interaction.usage
    print(interaction.output_text)
    print("-"*50)
    print(f"Model: {MODEL}")
    print(f"input: {usage.total_input_tokens} tokens")
    print(f"output: {usage.total_output_tokens} tokens")
    print(f"thinking: {usage.total_thought_tokens or 0} tokens")

    totals = running_totals()
    print("-"*50)
    print(f"total calls: {totals['calls']} total tokens: {totals['tokens']}")
    print(f"on the paid tier this would have cost: ${totals['paid_equivalent_usd']:.4f}")


if __name__ == "__main__":
    main()
