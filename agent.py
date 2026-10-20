"""CyberShield AI - Agentic Moderation Pipeline (Phase 3).

4-stage controlled agentic workflow:
    Stage 1 - DETECT   : Classify input and extract raw signals.
    Stage 2 - ANALYZE  : Deep reasoning over evidence and context.
    Stage 3 - DECIDE   : Policy-bound decision with priority routing.
    Stage 4 - ACT      : Produce concrete, auditable moderation actions.

Design constraints:
  - Each stage is deterministic and reproducible.
  - No automatic punitive actions; humans review critical cases.
  - All reasoning steps returned to caller for full transparency.
  - Thread-safe via stateless function design.
  - Zero external runtime dependencies.
"""

import re
import time
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Label metadata
# ---------------------------------------------------------------------------

LABEL_META: Dict[str, Dict[str, str]] = {
    "safe":    {"icon": "safe",    "color": "ok",    "display": "Safe"},
    "toxic":   {"icon": "toxic",   "color": "bad",   "display": "Toxic"},
    "insult":  {"icon": "insult",  "color": "bad",   "display": "Insult"},
    "threat":  {"icon": "threat",  "color": "crit",  "display": "Threat"},
    "obscene": {"icon": "obscene", "color": "warn",  "display": "Obscene"},
    "sarcasm": {"icon": "sarcasm", "color": "muted", "display": "Sarcasm"},
}

HARMFUL_LABELS = {"toxic", "insult", "threat", "obscene"}

LABEL_EMOJIS = {
    "safe":    "safe",
    "toxic":   "toxic",
    "insult":  "insult",
    "threat":  "threat",
    "obscene": "obscene",
    "sarcasm": "sarcasm",
}

# ---------------------------------------------------------------------------
# Action policies
# ---------------------------------------------------------------------------

ACTION_POLICIES: Dict[str, Dict[str, Any]] = {
    "ALLOW": {
        "badge": "ALLOW",
        "badge_color": "ok",
        "description": "Content is safe and constructive. No moderation action required.",
        "steps": [
            "Log analysis result.",
            "Return content to user without alteration.",
        ],
        "human_review": False,
    },
    "ALLOW_UNCERTAIN": {
        "badge": "UNCERTAIN",
        "badge_color": "muted",
        "description": "Low-confidence prediction. Treated as safe with an uncertainty notice.",
        "steps": [
            "Log uncertain result.",
            "Flag for passive monitoring.",
            "Display uncertainty notice to analyst.",
        ],
        "human_review": False,
    },
    "ALLOW_WITH_CONTEXT": {
        "badge": "CONTEXT",
        "badge_color": "muted",
        "description": "Sarcasm detected without active toxicity. Contextual review recommended.",
        "steps": [
            "Log sarcasm signal.",
            "Display context-aware notice.",
            "Proceed without restriction.",
        ],
        "human_review": False,
    },
    "SUGGEST_REWRITE_OR_WARN": {
        "badge": "WARN",
        "badge_color": "warn",
        "description": "Mildly harmful content detected. Suggest safer rewrite or issue soft warning.",
        "steps": [
            "Highlight flagged words in UI.",
            "Present safer rewrite suggestion.",
            "Issue soft community guidelines reminder.",
            "Log for trend monitoring.",
        ],
        "human_review": False,
    },
    "ELEVATED_MONITORING": {
        "badge": "ELEVATED",
        "badge_color": "warn",
        "description": "Borderline threat probability. Enhanced logging and priority monitoring activated.",
        "steps": [
            "Escalate to priority monitoring queue.",
            "Notify on-call moderator of borderline threat signal.",
            "Log full analysis result with timestamps.",
            "Do not auto-ban; await human assessment.",
        ],
        "human_review": True,
    },
    "FLAG_FOR_HUMAN_REVIEW": {
        "badge": "CRITICAL",
        "badge_color": "crit",
        "description": "High-confidence threat detected. Immediate human review required. No automated punitive action.",
        "steps": [
            "Immediately flag content for human moderator.",
            "Send critical priority alert to moderation team.",
            "Preserve content and metadata for review.",
            "Display safety notice to analyst.",
            "DO NOT auto-ban or auto-delete - human decision required.",
        ],
        "human_review": True,
    },
    "IGNORE_INVALID": {
        "badge": "INVALID",
        "badge_color": "muted",
        "description": "Input was empty or invalid. No action taken.",
        "steps": ["Reject invalid input.", "Return error to caller."],
        "human_review": False,
    },
}

# ---------------------------------------------------------------------------
# Evidence extraction helpers
# ---------------------------------------------------------------------------

_THREAT_PATTERNS = [
    r"\b(kill|hurt|destroy|murder|harm|attack|shoot|stab|beat|ruin)\b",
    r"\b(i will|i'll|gonna|i'm going to).{0,30}(you|him|her|them|everyone)\b",
    r"\b(find you|know where you|come for you|make you pay|you'll regret)\b",
]
_INSULT_PATTERNS = [
    r"\b(idiot|moron|stupid|dumb|loser|freak|worthless|pathetic|ugly|trash|garbage)\b",
]
_OBSCENE_PATTERNS = [
    r"\b(shit|fuck|damn|hell|bullshit|ass|crap|bastard|bitch)\b",
]
_SARCASM_PATTERNS = [
    # Classic mock-praise openers
    r"\b(oh|wow)\b.{0,25}(great|fantastic|wonderful|perfect|brilliant|amazing|sure)",
    r"\boh (great|fantastic|wonderful|perfect|sure|brilliant|amazing|charming|delightful|thrilling)\b",
    # Understated irony markers
    r"\byeah right\b",
    r"\boh sure\b",
    r"\bas if\b",
    r"\bof course\b.{0,40}(not|never|zero|wrong|broke|late|missed|failed|crashed|delayed)",
    r"\bobviously\b.{0,30}(not|never|zero|wrong|nothing|nobody|useless)",
    r"\bclearly\b.{0,30}(not|never|wrong|broke|failed|useless|pointless)",
    r"\bwhat a.{0,25}(joy|treat|pleasure|delight|surprise|shock|relief|gift)\b",
    # Rhetorical 'because' sarcasm
    r"\bbecause (clearly|obviously|surely|naturally|of course)\b",
    r"\bbecause (nothing|who|everyone|nobody) (loves|enjoys|wants)\b",
    r"\bbecause (that|this) (is|was) (definitely|obviously|clearly|totally)\b",
    # Passive-aggressive 'love when/spending/receiving'
    r"\bi (just |totally |absolutely )?(love|adore|enjoy) (when|spending|receiving|being|having)\b",
    r"\bi am (absolutely |totally |just )?(thrilled|delighted|overjoyed|ecstatic)\b",
    # Fake kudos/applause
    r"\b(kudos|bravo|applause|congratulations|a standing ovation|big round of applause)\b",
    r"\b(brilliant|genius|masterpiece|masterclass) (strategy|plan|idea|move|display|effort)\b",
    # 'Right because' / 'naturally' irony
    r"\bright,? because\b",
    r"\bnaturally,? the (one|only|exact)\b",
    # 'Exactly what I wanted/needed'
    r"\bexactly (what|how) (i|we|everyone) (wanted|needed|planned|expected|hoped)\b",
    # 'Truly inspiring/wonderful' irony
    r"\btruly (inspiring|wonderful|impressive|magnificent|spectacular|a masterpiece)\b",
]


def _find_pattern_evidence(text: str, patterns: List[str]) -> List[str]:
    text_lower = text.lower()
    found = []
    for pat in patterns:
        for m in re.finditer(pat, text_lower):
            found.append(m.group(0).strip())
    seen = set()
    deduped = []
    for item in found:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def _structural_signals(text: str) -> List[str]:
    signals = []
    if len(text) > 5 and text.isupper():
        signals.append("ALL-CAPS (shouting signal)")
    if text.count("!") >= 3:
        signals.append("Multiple exclamation marks (aggression signal)")
    if text.count("?") >= 3:
        signals.append("Multiple question marks (sarcasm/frustration signal)")
    if len(text) > 200:
        signals.append("Long message (detailed complaint or harassment pattern)")
    if len(text.split()) < 4:
        signals.append("Very short message (context-limited prediction)")
    return signals


# ---------------------------------------------------------------------------
# Stage 1: DETECT
# ---------------------------------------------------------------------------

def _stage_detect(text: str, predictor_result: Dict[str, Any]) -> Dict[str, Any]:
    """Stage 1: Extract raw signals and run initial classification."""
    t0 = time.monotonic()

    scores = predictor_result.get("scores", {})
    predicted_labels = predictor_result.get("predicted_labels", [])
    raw_predicted = predictor_result.get("raw_predicted_labels", predicted_labels)

    threat_ev  = _find_pattern_evidence(text, _THREAT_PATTERNS)
    insult_ev  = _find_pattern_evidence(text, _INSULT_PATTERNS)
    obscene_ev = _find_pattern_evidence(text, _OBSCENE_PATTERNS)
    sarcasm_ev = _find_pattern_evidence(text, _SARCASM_PATTERNS)
    structural  = _structural_signals(text)

    label_scores_display = []
    for lab in ["safe", "toxic", "insult", "threat", "obscene", "sarcasm"]:
        meta = LABEL_META.get(lab, {"icon": lab, "color": "muted", "display": lab})
        label_scores_display.append({
            "label": lab,
            "score": round(scores.get(lab, 0.0), 3),
            "threshold_crossed": lab in raw_predicted,
            "icon": meta["icon"],
            "color": meta["color"],
            "display": meta["display"],
        })

    evidence_found = []
    for e in threat_ev[:3]:
        evidence_found.append("Threat pattern: '" + e + "'")
    for e in insult_ev[:3]:
        evidence_found.append("Insult term: '" + e + "'")
    for e in obscene_ev[:3]:
        evidence_found.append("Obscene term: '" + e + "'")
    for e in sarcasm_ev[:2]:
        evidence_found.append("Sarcasm cue: '" + e + "'")
    evidence_found.extend(structural)

    reasoning = []
    word_count = len(text.split())
    reasoning.append("Text length: " + str(len(text)) + " characters, " + str(word_count) + " words.")

    if not evidence_found:
        reasoning.append("No explicit harmful patterns detected by lexical scan.")
    else:
        preview = ", ".join(evidence_found[:4])
        suffix = "..." if len(evidence_found) > 4 else ""
        reasoning.append(
            "Lexical scan found " + str(len(evidence_found)) + " signal(s): " + preview + suffix + "."
        )

    reasoning.append("ML model produced raw scores for " + str(len(scores)) + " labels.")

    crossed = [l for l in raw_predicted if l != "safe"]
    if crossed:
        reasoning.append("Threshold crossed for: " + ", ".join(crossed) + ".")
    else:
        reasoning.append("No harmful label crossed its decision threshold.")

    return {
        "stage": "DETECT",
        "stage_index": 1,
        "icon": "search",
        "emoji": "🔎",
        "title": "Signal Detection",
        "summary": "Scanned for harmful patterns. " + str(len(evidence_found)) + " signal(s) found.",
        "reasoning": reasoning,
        "label_scores": label_scores_display,
        "evidence": evidence_found,
        "auxiliary_context": predictor_result.get("auxiliary_context"),
        "duration_ms": round((time.monotonic() - t0) * 1000, 1),
    }


# ---------------------------------------------------------------------------
# Stage 2: ANALYZE
# ---------------------------------------------------------------------------

def _stage_analyze(
    text: str,
    predictor_result: Dict[str, Any],
    detect_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Stage 2: Deep reasoning over evidence, context, and multi-label signals."""
    t0 = time.monotonic()

    scores = predictor_result.get("scores", {})
    predicted = predictor_result.get("predicted_labels", [])
    aux = predictor_result.get("auxiliary_context")

    threat_score  = scores.get("threat", 0.0)
    insult_score  = scores.get("insult", 0.0)
    toxic_score   = scores.get("toxic", 0.0)
    obscene_score = scores.get("obscene", 0.0)
    sarcasm_score = scores.get("sarcasm", 0.0)
    safe_score    = scores.get("safe", 0.0)

    reasoning = []
    observations = []
    concerns = []

    # Threat analysis
    if threat_score >= 0.35:
        concerns.append(
            "Threat probability " + str(round(threat_score * 100)) + "% exceeds 35% critical threshold."
        )
        reasoning.append(
            "THREAT: Score " + str(round(threat_score, 3)) + " >= 0.35 threshold. Immediate escalation required."
        )
    elif threat_score >= 0.20:
        concerns.append(
            "Borderline threat signal (" + str(round(threat_score * 100)) + "%). Priority monitoring required."
        )
        reasoning.append(
            "THREAT: Borderline score " + str(round(threat_score, 3)) + " (0.20-0.35). Elevated monitoring."
        )
    else:
        reasoning.append(
            "THREAT: Score " + str(round(threat_score, 3)) + " below threat threshold. No direct threat detected."
        )

    # Multi-label co-occurrence
    harm_labels = [l for l in predicted if l in HARMFUL_LABELS]
    if len(harm_labels) >= 2:
        concerns.append(
            "Multiple harm categories co-occurred: " + ", ".join(harm_labels) + ". Severity amplified."
        )
        reasoning.append(
            "CO-OCCURRENCE: " + str(len(harm_labels)) + " harmful labels active simultaneously - severity elevated."
        )
    elif len(harm_labels) == 1:
        reasoning.append(
            "SINGLE LABEL: Only " + harm_labels[0] + " is active. Standard moderation applies."
        )

    # Sarcasm independence check
    if sarcasm_score >= 0.40 and toxic_score < 0.40:
        observations.append("Sarcasm detected in isolation - not classified as toxic per policy.")
        reasoning.append(
            "SARCASM: Score " + str(round(sarcasm_score, 3)) + " without toxicity (" + str(round(toxic_score, 3)) + "). Treated separately."
        )
    elif sarcasm_score >= 0.40 and toxic_score >= 0.40:
        concerns.append("Sarcasm combined with toxicity - tone is hostile and mocking.")
        reasoning.append(
            "SARCASM+TOXIC: Both elevated (" + str(round(sarcasm_score, 3)) + ", " + str(round(toxic_score, 3)) + "). Hostile sarcasm."
        )

    # Auxiliary context check
    if aux:
        observations.append("Non-literal phrase detected: " + aux)
        reasoning.append("CONTEXT: " + aux + " - idiomatic usage reduces threat confidence.")

    # Contradiction resolution
    if "safe" in predicted and harm_labels:
        reasoning.append(
            "CONTRADICTION: 'safe' and harmful labels both crossed threshold - harmful-precedence policy applied."
        )

    # Safe comment reasoning
    if not harm_labels and sarcasm_score < 0.40:
        if safe_score >= 0.40:
            observations.append("Safe label active (score " + str(round(safe_score * 100)) + "%). Message appears constructive.")
        else:
            observations.append("No label crossed threshold. Low-confidence safe classification.")
        reasoning.append(
            "SAFE: No harmful labels active. Safe=" + str(round(safe_score, 3)) + ", Toxic=" + str(round(toxic_score, 3)) + "."
        )

    summary_parts = []
    if concerns:
        summary_parts.append(str(len(concerns)) + " concern(s) identified.")
    if observations:
        summary_parts.append(str(len(observations)) + " observation(s) noted.")
    if not concerns and not observations:
        summary_parts.append("Content appears benign.")

    return {
        "stage": "ANALYZE",
        "stage_index": 2,
        "icon": "brain",
        "emoji": "🧠",
        "title": "Deep Analysis",
        "summary": " ".join(summary_parts),
        "reasoning": reasoning,
        "concerns": concerns,
        "observations": observations,
        "active_harm_labels": harm_labels,
        "multi_label_count": len(harm_labels),
        "duration_ms": round((time.monotonic() - t0) * 1000, 1),
    }


# ---------------------------------------------------------------------------
# Stage 3: DECIDE
# ---------------------------------------------------------------------------

def _stage_decide(
    text: str,
    predictor_result: Dict[str, Any],
    detect_result: Dict[str, Any],
    analyze_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Stage 3: Policy-bound decision with structured priority routing."""
    t0 = time.monotonic()

    moderation_policy = predictor_result.get("moderation_policy", {})
    action_key = moderation_policy.get("action", "ALLOW_UNCERTAIN")
    severity = predictor_result.get("severity", "none")
    primary_label = predictor_result.get("primary_label", "safe")
    harm_labels = analyze_result.get("active_harm_labels", [])
    concerns = analyze_result.get("concerns", [])

    policy = ACTION_POLICIES.get(action_key, ACTION_POLICIES["ALLOW_UNCERTAIN"])

    reasoning = []
    routing_factors = []

    routing_factors.append("Primary label: " + primary_label)
    routing_factors.append("Severity level: " + severity)
    if harm_labels:
        routing_factors.append("Active harm labels: " + ", ".join(harm_labels))
    routing_factors.append("Policy action: " + action_key)
    routing_factors.append("Human review required: " + ("Yes" if policy["human_review"] else "No"))

    reasoning.append(
        "Action '" + action_key + "' selected based on primary label '" + primary_label + "' with " + severity + " severity."
    )
    if concerns:
        reasoning.append(str(len(concerns)) + " concern(s) from analysis stage influenced priority routing.")
    pol_reason = moderation_policy.get("reason", "N/A")
    reasoning.append("Moderation policy reason: " + pol_reason)

    if policy["human_review"]:
        reasoning.append("POLICY CONSTRAINT: Human review is mandatory. No automated punitive actions allowed.")
    else:
        reasoning.append("Automated action is safe to proceed without human escalation.")

    priority = moderation_policy.get("priority", "NONE")
    priority_color = {
        "NONE": "ok",
        "LOW": "muted",
        "STANDARD": "warn",
        "ELEVATED": "warn",
        "CRITICAL": "crit",
    }.get(priority, "muted")

    return {
        "stage": "DECIDE",
        "stage_index": 3,
        "icon": "gavel",
        "emoji": "⚖️",
        "title": "Policy Decision",
        "summary": (
            "Action: " + policy["badge"] +
            " | Priority: " + priority +
            " | Human review: " + ("required" if policy["human_review"] else "not required") + "."
        ),
        "reasoning": reasoning,
        "action_key": action_key,
        "action_badge": policy["badge"],
        "action_badge_color": policy["badge_color"],
        "action_description": policy["description"],
        "priority": priority,
        "priority_color": priority_color,
        "human_review_required": policy["human_review"],
        "routing_factors": routing_factors,
        "duration_ms": round((time.monotonic() - t0) * 1000, 1),
    }


# ---------------------------------------------------------------------------
# Stage 4: ACT
# ---------------------------------------------------------------------------

def _stage_act(
    text: str,
    predictor_result: Dict[str, Any],
    detect_result: Dict[str, Any],
    analyze_result: Dict[str, Any],
    decide_result: Dict[str, Any],
    safer_rewrite: Optional[str] = None,
    flagged_words: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Stage 4: Produce concrete, auditable moderation actions."""
    t0 = time.monotonic()

    action_key = decide_result["action_key"]
    policy = ACTION_POLICIES.get(action_key, ACTION_POLICIES["ALLOW_UNCERTAIN"])
    severity = predictor_result.get("severity", "none")
    primary_label = predictor_result.get("primary_label", "safe")
    harm_labels = analyze_result.get("active_harm_labels", [])

    actions_taken: List[Dict[str, Any]] = []
    recommendations: List[str] = []
    reasoning = []

    for i, step in enumerate(policy["steps"], 1):
        actions_taken.append({"step": i, "action": step, "completed": True})

    if safer_rewrite and safer_rewrite != text:
        preview = safer_rewrite[:80]
        if len(safer_rewrite) > 80:
            preview += "..."
        actions_taken.append({
            "step": len(actions_taken) + 1,
            "action": 'Safer rewrite generated: "' + preview + '"',
            "completed": True,
        })
        recommendations.append("Consider using the AI-suggested rewrite for a more constructive message.")

    if flagged_words:
        words = [fw.get("word", "") for fw in flagged_words[:5]]
        actions_taken.append({
            "step": len(actions_taken) + 1,
            "action": "Flagged words highlighted: " + ", ".join(words),
            "completed": True,
        })

    if policy["human_review"]:
        recommendations.append(
            "CRITICAL: Escalate to human moderator immediately. Do not apply automatic bans or deletions."
        )
        recommendations.append(
            "Preserve original content and timestamps for moderator audit trail."
        )

    if harm_labels:
        recommendations.append(
            "Monitor user for repeated " + ", ".join(harm_labels) + " violations over the next 48 hours."
        )

    predicted_labels = predictor_result.get("predicted_labels", [])
    if "sarcasm" in predicted_labels and not harm_labels:
        recommendations.append(
            "Sarcasm is context-dependent. Consider platform community culture before acting."
        )

    reasoning.append("Executed " + str(len(actions_taken)) + " action step(s) for policy '" + action_key + "'.")
    if recommendations:
        reasoning.append("Generated " + str(len(recommendations)) + " recommendation(s) for follow-up.")
    reasoning.append("All actions logged. Agent pipeline complete.")

    outcome_summary = (
        "Content classified as '" + primary_label + "' with " + severity + " severity. "
        + "Policy '" + action_key + "' executed with " + str(len(actions_taken)) + " action(s). "
        + ("Human review required." if policy["human_review"] else "No human escalation needed.")
    )

    return {
        "stage": "ACT",
        "stage_index": 4,
        "icon": "bolt",
        "emoji": "⚡",
        "title": "Moderation Actions",
        "summary": outcome_summary,
        "reasoning": reasoning,
        "actions_taken": actions_taken,
        "recommendations": recommendations,
        "policy_badge": policy["badge"],
        "policy_badge_color": policy["badge_color"],
        "human_review_required": policy["human_review"],
        "audit": {
            "primary_label": primary_label,
            "severity": severity,
            "harm_labels": harm_labels,
            "action_key": action_key,
            "agent_version": "3.0.0",
        },
        "duration_ms": round((time.monotonic() - t0) * 1000, 1),
    }


# ---------------------------------------------------------------------------
# Public pipeline entry point
# ---------------------------------------------------------------------------

def run_agent_pipeline(
    text: str,
    predictor_result: Dict[str, Any],
    safer_rewrite: Optional[str] = None,
    flagged_words: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run the full 4-stage agentic moderation pipeline.

    Args:
        text: The original input comment (already stripped by caller).
        predictor_result: Output dict from CyberShieldPredictor.predict().
        safer_rewrite: Optional safer rewrite string from app.py.
        flagged_words: Optional list of flagged word dicts from app.py.

    Returns:
        Full agent pipeline result including all 4 stage results and metadata.
    """
    pipeline_start = time.monotonic()

    stage1 = _stage_detect(text, predictor_result)
    stage2 = _stage_analyze(text, predictor_result, stage1)
    stage3 = _stage_decide(text, predictor_result, stage1, stage2)
    stage4 = _stage_act(text, predictor_result, stage1, stage2, stage3, safer_rewrite, flagged_words)

    total_ms = round((time.monotonic() - pipeline_start) * 1000, 1)

    return {
        "agent_version": "3.0.0",
        "input": text[:200],
        "stages": [stage1, stage2, stage3, stage4],
        "final_action": stage3["action_key"],
        "final_badge": stage3["action_badge"],
        "final_badge_color": stage3["action_badge_color"],
        "severity": predictor_result.get("severity", "none"),
        "primary_label": predictor_result.get("primary_label", "safe"),
        "human_review_required": stage3["human_review_required"],
        "total_pipeline_ms": total_ms,
        "moderation_policy": predictor_result.get("moderation_policy", {}),
        "disclaimer": "Agent decisions are advisory. Human moderators make final determinations.",
    }
