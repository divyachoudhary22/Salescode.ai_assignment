# agent_main.py (example glue)
import asyncio
from interrupt_filter import InterruptFilter, ASRSegment

def logger(msg, kv):
    print(f"[{msg}] {kv}")

filter = InterruptFilter(logger=logger)

async def on_tts_start():
    await filter.set_tts_speaking(True)

async def on_tts_end():
    await filter.set_tts_speaking(False)

async def on_asr_segment(text: str, confidence: float, is_final: bool):
    decision = await filter.decide(ASRSegment(text=text, confidence=confidence, is_final=is_final))

    if decision.should_interrupt:
        logger("action", {"do": "STOP_TTS", "reason": decision.reason})
        # your_tts_controller.stop_now()

    if decision.should_consume:
        logger("drop", {"text": text, "reason": decision.reason})
        return

    logger("pass_downstream", {"text": text, "reason": decision.reason})
    # dialog_manager.feed(text)

async def demo():
    await on_tts_start()
    await on_asr_segment("umm", 0.95, True)             # filler -> consumed
    await on_asr_segment("umm okay stop", 0.92, True)   # command -> interrupt
    await on_tts_end()
    await on_asr_segment("hmm yeah", 0.6, False)        # quiet -> passthrough

if __name__ == "__main__":
    asyncio.run(demo())
