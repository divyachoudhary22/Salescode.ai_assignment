# LiveKit Voice Interruption Handling (Filler-Aware)

## Files
- `interrupt_filter.py` – core filter
- `agent_main.py` – minimal wiring example
- `test_interrupt_filter.py` – unit tests

## Quick Start
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pytest
python agent_main.py               # run the tiny demo
pytest -q                          # run unit tests
```

## Env (optional)
```bash
export IGNORED_WORDS="uh,umm,hmm,haan,achha"
export INTERRUPT_INTENTS="stop,wait,hold on,no not that,cancel,ruko,band karo"
```
