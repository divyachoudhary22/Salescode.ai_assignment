# test_interrupt_filter.py
import pytest
import asyncio
from interrupt_filter import InterruptFilter, ASRSegment

@pytest.mark.asyncio
async def test_ignore_filler_while_speaking():
    f = InterruptFilter()
    await f.set_tts_speaking(True)
    d = await f.decide(ASRSegment(text="umm", confidence=0.90, is_final=True))
    assert d.should_interrupt is False
    assert d.should_consume is True
    assert d.reason == "filler_only"

@pytest.mark.asyncio
async def test_interrupt_on_command_while_speaking():
    f = InterruptFilter()
    await f.set_tts_speaking(True)
    d = await f.decide(ASRSegment(text="umm okay stop", confidence=0.92, is_final=True))
    assert d.should_interrupt is True
    assert d.should_consume is False
    assert d.reason == "command_intent"

@pytest.mark.asyncio
async def test_passthrough_when_quiet():
    f = InterruptFilter()
    await f.set_tts_speaking(False)
    d = await f.decide(ASRSegment(text="umm", confidence=0.50, is_final=False))
    assert d.should_interrupt is False
    assert d.should_consume is False
    assert d.reason == "agent_quiet_passthrough"

@pytest.mark.asyncio
async def test_low_confidence_ignored_while_speaking():
    f = InterruptFilter(confidence_threshold=0.6)
    await f.set_tts_speaking(True)
    d = await f.decide(ASRSegment(text="noise hmm", confidence=0.4, is_final=False))
    assert d.should_interrupt is False
    assert d.should_consume is True
    assert d.reason == "low_confidence"

@pytest.mark.asyncio
async def test_meaningful_overlap_interrupts():
    f = InterruptFilter()
    await f.set_tts_speaking(True)
    d = await f.decide(ASRSegment(text="no, choose the second one", confidence=0.93, is_final=True))
    assert d.should_interrupt is True
    assert d.should_consume is False
    assert d.reason == "meaningful_overlap"
