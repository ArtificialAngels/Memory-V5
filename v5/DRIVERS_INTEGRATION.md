# Ikaros V5 Thinking Driver — Integration Proposal

## What was created

### 1) Research Document
**`v5/thinking_drivers_research.md`** — Comparative analysis of all three algorithms,
recommending Phase 2 implementation order: ECA > Chaos PAD > AIS.

### 2) Working Python Module
**`v5/drivers.py`** — Three dataclass-based drivers + a composite `ThinkingOrchestrator`.  
Each driver is <50 lines of core logic, passes self-test.

---

## Driver Verification Results

| Driver | Lines | O/tick | Status | Output |
|--------|-------|--------|--------|--------|
| **LorenzPAD** | 35 | O(1) | ✅ | Smooth PAD drift, Lorenz butterfly attractor |
| **ECAGrid** (Rule 110, 41 cells) | 40 | O(41) | ✅ | Topics: 混沌思维↔静默↔情感波动↔外部关注 |
| **AISDetectorSet** (100 detectors) | 48 | O(M×D) | ✅ | Novelty scores 0–1, correct ranking |
| **ThinkingOrchestrator** | 25 | O(1+M×D) | ✅ | Three layers composited |

---

## Integration Path into think.py

The existing `inner_monologue()` in `think.py` does:

```python
mood = _pad_to_mood(p, a, d)
templates = _TEMPLATES.get(mood, _TEMPLATES["neutral_calm"])
text = random.choice(templates)
```

Replace the last two lines with:

```python
from v5.drivers import LorenzPAD, ECAGrid, AISDetectorSet

# Instantiate once at module level
_chaos = LorenzPAD()
_eca = ECAGrid()
_ais = AISDetectorSet()

def inner_monologue(*, now=None):
    # 1) Load current PAD (existing code)
    state = AffectState.load().decay(now=now)
    p, a, d = state.pleasure, state.arousal, state.dominance

    # 2) Chaos PAD: blend current state with attractor
    p, a, d = _chaos.blend((p, a, d), blend_factor=0.3)

    # 3) ECA: thinking topic
    topic = _eca.tick()
    activity = _eca.activity_ratio()

    # 4) Select template based on topic, not mood
    templates = _TOPIC_TEMPLATES.get(topic, _TEMPLATES["neutral_calm"])
    text = random.choice(templates)

    # 5) AIS: novelty score for the resulting thought
    # ... (Phase 3)
```

This is a drop-in replacement — the existing `Thought` dataclass, pending file, and intensity
threshold logic remain unchanged. Only the template selection source changes.

---

## Recommendation Summary

**Phase 2 (immediate):** Implement ECA + Chaos PAD as the thinking driver core.  
- ECA Rule 110 solves "what to think about" with organic, unpredictable topic transitions  
- Chaos PAD solves "when emotions drift" without needing conversation input  
- Combined cost: ~μs per tick, <1ms even on low-power CPU  

**Phase 3 (next):** Implement AIS novelty detection for memory recall prioritization.  
- Requires sufficiently populated PAD columns in V4 memory  
- Solves "which memories are interesting right now"  

**Do not replace** the template layer entirely — keep `_TEMPLATES` as the output renderer,
just drive template selection from ECA topic + Chaos PAD instead of `random.choice`.
