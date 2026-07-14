"""
Terminal client for the /chat endpoint -- lets you demo streaming without
building a frontend. Start the API first:

    uvicorn app.main:app --reload

Then in another terminal:

    python scripts/chat_cli.py
"""
import httpx

API_URL = "http://localhost:8000/chat"


def main() -> None:
    session_id = "cli"
    print("AI Engineering Assistant - Phase 1 (plain chat). Ctrl+C to exit.\n")
    while True:
        try:
            message = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not message:
            continue
        print("Assistant: ", end="", flush=True)
        with httpx.stream(
            "POST", API_URL, json={"session_id": session_id, "message": message}, timeout=None
        ) as r:
            for chunk in r.iter_text():
                print(chunk, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    main()
