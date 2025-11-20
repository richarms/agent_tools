import argparse
import os
from agent_tools import config, session, agent

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive agent REPL.")
    parser.add_argument(
        "--backend",
        choices=("gemini", "openai"),
        default=os.getenv("AGENT_BACKEND", "gemini"),
        help="Choose which LLM backend to use (default: gemini).",
    )
    parser.add_argument(
        "--gemini-model",
        default=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        help="Gemini model to use (default: gemini-2.5-flash).",
    )
    parser.add_argument(
        "--openai-model",
        default=os.getenv("OPENAI_MODEL", config.DEFAULT_MODEL),
        help=f"OpenAI Responses API model to use (default: {config.DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--target-mc",
        default=os.getenv("TARGET_MC", config.DEFAULT_TARGET_MC),
        help="Base MC URL used for Grafana/Kibana/Elasticsearch defaults.",
    )
    parser.add_argument(
        "--resume",
        metavar="SESSION_ID",
        help="Resume a saved session using the hashed reference printed on exit.",
    )
    # --model is deprecated/ambiguous now, but kept for backward compatibility mapping to openai_model if used?
    # Or we can remove it. Let's keep it but make it override the specific model config if provided.
    parser.add_argument(
        "--model",
        help="Override model name for the selected backend.",
    )
    return parser.parse_args()

def repl(backend: agent.BaseBackend) -> None:
    while True:
        line = input("\033[1mAsk anything: \033[0m")
        result, report = backend.process(line)
        print(f"%%; {result}\n")
        if report:
            print(report)
            print()

def main() -> None:
    args = parse_args()
    config.configure_target_mc(args.target_mc)
    
    # Handle generic --model override
    gemini_model = args.model if (args.model and args.backend == "gemini") else args.gemini_model
    openai_model = args.model if (args.model and args.backend == "openai") else args.openai_model
    
    backend = agent.create_backend(args.backend, gemini_model, openai_model)
    
    if args.resume:
        loaded_context = session.load_session(args.resume)
        if loaded_context:
            # If the loaded context doesn't have a system prompt at start, add it.
            if not loaded_context or loaded_context[0].get("role") != "system":
                loaded_context.insert(0, {"role": "system", "content": config.SYSTEM_PROMPT})
            
            if isinstance(backend, agent.OpenAIBackend):
                backend.set_context(loaded_context)
            elif isinstance(backend, agent.GeminiBackend):
                backend.set_context(loaded_context)
                print("Warning: Resuming session in Gemini backend only restores the save history, context might not be fully active in the model's memory.")
            
            print(f"Resumed session '{args.resume}'.")
        else:
            print(f"Could not find session '{args.resume}', starting a new one.")
            
    try:
        repl(backend)
    except (EOFError, KeyboardInterrupt):
        print("Bye.")
        summary = agent.token_tracker.session_summary()
        if summary:
            print(summary)
    finally:
        # Save session
        saved_id = session.save_session_context(backend.context)
        if saved_id:
            print(f"Session context saved. Reference: {saved_id}")
            print(f"Resume later with: python -m agent_tools --resume {saved_id}")
        else:
            print("Unable to save session context (temporary storage unavailable).")

if __name__ == "__main__":
    main()