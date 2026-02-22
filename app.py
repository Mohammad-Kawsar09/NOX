# app.py
# NOX — Night Operations Copilot (Gradio 6 compatible)
# - Chat history uses messages dict format (role/content)
# - No "AI" word displayed
# - Times New Roman everywhere
# - Free local lightweight model (google/flan-t5-small) for text generation
# - Works even if model isn't ready (deterministic fallback)

import json
import re
from typing import Any, Dict, List, Tuple

import gradio as gr
import matplotlib.pyplot as plt
from transformers import pipeline


# -------------------------
# Styling (Times New Roman everywhere)
# -------------------------
CUSTOM_CSS = """
* { font-family: "Times New Roman", Times, serif !important; }
.gradio-container { max-width: 1180px !important; }
#nox_header h1 { font-size: 2.25rem; letter-spacing: -0.03em; margin-bottom: 0.2rem; }
#nox_subtitle { font-size: 1.02rem; opacity: 0.85; margin-top: 0.15rem; }
.nox-card { border: 1px solid rgba(0,0,0,0.08); border-radius: 16px; padding: 14px; background: rgba(255,255,255,0.55); }
button, .gr-button { border-radius: 12px !important; }
button.primary, .gr-button-primary { border-radius: 14px !important; font-weight: 650 !important; padding: 12px 16px !important; }
"""

theme = gr.themes.Soft(radius_size="lg", text_size="md")
SAFETY_FOOTER = "Not a diagnosis; follow local escalation protocols."


# -------------------------
# De-identification (demo-safe)
# -------------------------
def scrub_phi(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b", "[REDACTED]", text)
    text = re.sub(r"\b(MRN|ID)\s*[:#]?\s*\w+\b", "[REDACTED]", text, flags=re.I)
    text = re.sub(r"\b\d{10,}\b", "[REDACTED]", text)
    return text


# -------------------------
# Simple deterministic risk engine
# -------------------------
NOTE_KEYWORDS = [
    ("confus", 0.20),
    ("fatigu", 0.12),
    ("short of breath", 0.22),
    ("breathing faster", 0.18),
    ("worsen", 0.14),
    ("distress", 0.12),
    ("letharg", 0.14),
]


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def slope_norm(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    first, last = values[0], values[-1]
    avg = sum(values) / len(values) if values else 0.0
    if avg == 0:
        return 0.0
    return (last - first) / avg


def evidence_gate(label: str, modality: Dict[str, bool], has_drivers: bool) -> Dict[str, Any]:
    modalities = sum(1 for _k, v in (modality or {}).items() if v)
    ok = (modalities >= 2) and has_drivers
    final_label = label
    if label == "HIGH" and not ok:
        final_label = "MED"
    return {
        "final_label": final_label,
        "modalities_triggered": modalities,
        "rule": "HIGH requires ≥2 evidence sources + structured evidence (Evidence Gate).",
    }


def heuristic_risk(patient: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    vitals = patient.get("vitals", []) or []
    labs = patient.get("labs", []) or []
    notes = patient.get("notes", []) or []

    lact_slope = slope_norm([x.get("lactate", 0.0) for x in labs]) if labs else 0.0
    wbc_slope = slope_norm([x.get("wbc", 0.0) for x in labs]) if labs else 0.0
    cre_slope = slope_norm([x.get("creat", 0.0) for x in labs]) if labs else 0.0

    note_hits = 0
    for n in notes[-3:]:
        t = (n.get("text") or "").lower()
        for k, _w in NOTE_KEYWORDS:
            if k in t:
                note_hits += 1
                break

    timeline: List[Dict[str, Any]] = []
    for i, p in enumerate(vitals):
        start = max(0, i - 3)
        w = vitals[start : i + 1]

        hr_s = slope_norm([x.get("hr", 0.0) for x in w])
        rr_s = slope_norm([x.get("rr", 0.0) for x in w])
        sp_s = slope_norm([x.get("spo2", 0.0) for x in w])
        sb_s = slope_norm([x.get("sbp", 0.0) for x in w])

        hr_r = clamp01(hr_s * 2.5)
        rr_r = clamp01(rr_s * 3.2)
        sp_r = clamp01((-sp_s) * 4.2)
        sb_r = clamp01((-sb_s) * 2.0)

        risk = 0.34 * rr_r + 0.24 * sp_r + 0.18 * hr_r + 0.10 * sb_r
        lab_r = clamp01(lact_slope * 2.2) * 0.10 + clamp01(wbc_slope * 1.6) * 0.06 + clamp01(cre_slope * 1.2) * 0.03
        note_r = clamp01(min(1.0, note_hits / 3)) * 0.07
        risk = clamp01(risk + lab_r + note_r)

        timeline.append({"t": p.get("t", ""), "risk": float(risk)})

    latest = timeline[-1]["risk"] if timeline else 0.0
    label = "HIGH" if latest >= 0.70 else ("MED" if latest >= 0.35 else "LOW")

    drivers = []
    last_window = vitals[max(0, len(vitals) - 4) :] if vitals else []
    if last_window:
        rr_s = slope_norm([x.get("rr", 0.0) for x in last_window])
        sp_s = slope_norm([x.get("spo2", 0.0) for x in last_window])
        hr_s = slope_norm([x.get("hr", 0.0) for x in last_window])
        if rr_s > 0.05:
            drivers.append({"key": "vitals:rr_trend", "weight": 0.30, "evidence": "RR rising (recent trend)."})
        if sp_s < -0.01:
            drivers.append({"key": "vitals:spo2_drop", "weight": 0.22, "evidence": "SpO₂ drifting down (recent trend)."})
        if hr_s > 0.05:
            drivers.append({"key": "vitals:hr_trend", "weight": 0.18, "evidence": "HR rising (recent trend)."})
    if lact_slope > 0.08:
        drivers.append({"key": "labs:lactate_drift", "weight": 0.12, "evidence": "Lactate rising across labs."})
    if wbc_slope > 0.08:
        drivers.append({"key": "labs:wbc_drift", "weight": 0.06, "evidence": "WBC rising across labs."})

    drivers.sort(key=lambda d: d["weight"], reverse=True)

    modality = {
        "vitals": bool(vitals),
        "labs": bool(labs) and (abs(lact_slope) > 0.05 or abs(wbc_slope) > 0.05 or abs(cre_slope) > 0.05),
        "notes": note_hits > 0,
    }

    return timeline, {
        "score": float(latest),
        "label": label,
        "confidence": 0.62,
        "drivers": drivers[:6],
        "modality": modality,
        "mode": "rules",
    }


# -------------------------
# Local language engine (no keys)
# -------------------------
_ENGINE = {"pipe": None, "last_err": None, "ready": False}


def init_engine():
    if _ENGINE["ready"] and _ENGINE["pipe"] is not None:
        return
    try:
        _ENGINE["pipe"] = pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
            device=-1,
        )
        _ENGINE["ready"] = True
        _ENGINE["last_err"] = None
    except Exception as e:
        _ENGINE["pipe"] = None
        _ENGINE["ready"] = False
        _ENGINE["last_err"] = str(e)


def engine_status() -> str:
    init_engine()
    if _ENGINE["ready"]:
        return "Engine: Ready ✅ (local model loaded)"
    if _ENGINE["last_err"]:
        return "Engine: Not available (model load issue). Check Space logs."
    return "Engine: Warming up… try again in 30–90 seconds."


def gen_text(prompt: str, max_new_tokens: int = 200) -> str:
    init_engine()
    if not _ENGINE["ready"] or _ENGINE["pipe"] is None:
        return ""
    try:
        out = _ENGINE["pipe"](prompt, max_new_tokens=max_new_tokens, do_sample=False)
        return (out[0].get("generated_text") or "").strip()
    except Exception as e:
        _ENGINE["last_err"] = str(e)
        _ENGINE["ready"] = False
        return ""


# -------------------------
# Plots
# -------------------------
def plot_risk_timeline(timeline: List[Dict[str, Any]], title: str):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if not timeline:
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    x = list(range(len(timeline)))
    y = [p["risk"] for p in timeline]
    labels = [timeline[i]["t"][11:16] if len(timeline[i]["t"]) >= 16 else str(i) for i in range(len(timeline))]

    ax.plot(x, y, linewidth=2)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xlabel("Time (HH:MM)")
    ax.set_ylabel("Timeline (0..1)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    return fig


def plot_ward_heatmap(board: List[Dict[str, Any]]):
    grid = [[0.0 for _ in range(7)] for _ in range(4)]
    labels = [["" for _ in range(7)] for _ in range(4)]

    for p in board:
        bed_str = str(p.get("bed", ""))
        m = re.search(r"(\d+)", bed_str)
        if not m:
            continue
        bed = int(m.group(1))
        if bed < 1 or bed > 28:
            continue
        r = (bed - 1) // 7
        c = (bed - 1) % 7
        risk = float(p.get("risk", 0.0))
        grid[r][c] = risk
        labels[r][c] = f"{bed}\n{p.get('final_label','')}"

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(grid, vmin=0.0, vmax=1.0)
    ax.set_title("Ward risk heatmap (beds)")
    ax.set_xticks(range(7))
    ax.set_yticks(range(4))
    ax.set_xticklabels([str(i) for i in range(1, 8)])
    ax.set_yticklabels([f"Row {i+1}" for i in range(4)])
    ax.set_xlabel("Bed column")
    ax.set_ylabel("Bed row")

    for r in range(4):
        for c in range(7):
            if labels[r][c]:
                ax.text(c, r, labels[r][c], ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Risk (0..1)")
    fig.tight_layout()
    return fig


# -------------------------
# File loader
# -------------------------
def load_from_file(file_obj) -> str:
    if file_obj is None:
        return ""
    try:
        with open(file_obj.name, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


# -------------------------
# Board build
# -------------------------
def build_board(ward_json: str):
    init_engine()

    try:
        ward = json.loads(ward_json)
        patients = ward.get("patients", [])
        if not isinstance(patients, list) or not patients:
            return [], "", None, "", "Board: Invalid ward JSON (missing patients)."
    except Exception:
        return [], "", None, "", "Board: Invalid ward JSON (parse error)."

    board: List[Dict[str, Any]] = []

    for p in patients:
        p["notes"] = [
            {"t": n.get("t"), "author": n.get("author"), "text": scrub_phi(n.get("text", ""))}
            for n in (p.get("notes", []) or [])
        ]

        timeline, heur = heuristic_risk(p)
        modality = heur.get("modality", {})
        drivers = heur.get("drivers", [])
        gate = evidence_gate(heur.get("label", "LOW"), modality, bool(drivers))

        risk = float(heur.get("score", 0.0))
        conf = float(heur.get("confidence", 0.62))

        why_parts = [d.get("key", "") for d in drivers[:2] if d.get("key")]
        why = ", ".join(why_parts) if why_parts else "stable signals"

        board.append({
            "id": p.get("id"),
            "bed": p.get("bed"),
            "dx": p.get("dx"),
            "risk": round(risk, 3),
            "risk_pct": f"{round(risk * 100)}%",
            "final_label": gate.get("final_label", heur.get("label", "LOW")),
            "confidence": round(conf, 2),
            "why": why,
            "modalities": modality,
            "mode": "engine+rules" if _ENGINE["ready"] else "rules",
            "timeline": timeline
        })

    board.sort(key=lambda x: float(x.get("risk", 0.0)), reverse=True)
    top_pid = str(board[0].get("id", "")) if board else ""

    rows = []
    for idx, x in enumerate(board, start=1):
        m = x.get("modalities", {}) or {}
        sources = []
        if m.get("vitals"): sources.append("Vitals")
        if m.get("labs"): sources.append("Labs")
        if m.get("notes"): sources.append("Notes")

        rows.append([
            x.get("bed", "—"),
            x.get("id", "—"),
            str(idx),
            f"{x.get('risk_pct', '0%')} ({x.get('final_label', 'LOW')})",
            str(x.get("confidence", 0.0)),
            x.get("why", "—"),
            ", ".join(sources) if sources else "—",
            x.get("mode", "—"),
        ])

    heatmap = plot_ward_heatmap(board)
    status = "Board: Built ✅  | " + ("Engine active ✅" if _ENGINE["ready"] else "Engine warming up (board still works).")
    return rows, json.dumps(board, indent=2), heatmap, top_pid, status


# -------------------------
# Focus mode
# -------------------------
def open_patient(ward_json: str, patient_id: str):
    init_engine()

    try:
        ward = json.loads(ward_json)
        patients = ward.get("patients", [])
    except Exception:
        return None, "Invalid ward JSON.", "{}", "{}"

    pid = (patient_id or "").strip()
    patient = next((p for p in patients if p.get("id") == pid), None)
    if not patient:
        return None, f"Patient {pid} not found.", "{}", "{}"

    timeline, heur = heuristic_risk(patient)
    fig = plot_risk_timeline(timeline, f"Timeline — {patient.get('bed')} ({patient.get('id')})")

    bullets = []
    vitals = patient.get("vitals", []) or []
    labs = patient.get("labs", []) or []
    notes = patient.get("notes", []) or []

    if len(vitals) >= 2:
        prev, last = vitals[-2], vitals[-1]
        if last.get("rr", 0) - prev.get("rr", 0) >= 3:
            bullets.append("RR increased in the most recent interval.")
        if prev.get("spo2", 0) - last.get("spo2", 0) >= 2:
            bullets.append("SpO₂ dropped in the most recent interval.")
        if last.get("hr", 0) - prev.get("hr", 0) >= 8:
            bullets.append("HR increased in the most recent interval.")

    if len(labs) >= 2:
        prev, last = labs[-2], labs[-1]
        if last.get("lactate", 0) - prev.get("lactate", 0) >= 0.4:
            bullets.append("Lactate increased across the last two labs.")
        if last.get("wbc", 0) - prev.get("wbc", 0) >= 1.5:
            bullets.append("WBC increased across the last two labs.")

    if notes:
        last_note = scrub_phi((notes[-1].get("text") or "")).lower()
        if "confus" in last_note or "fatigu" in last_note or "breathing faster" in last_note or "short of breath" in last_note:
            bullets.append("Recent note language suggests increased concern.")

    if not bullets:
        bullets = ["No major changes detected in the last interval based on available data."]

    prompt = (
        "Write a night-shift change summary. Do NOT diagnose. Do NOT recommend medications.\n"
        "Output:\n"
        "1) 2–3 sentence narrative\n"
        "2) 3 bullets: why it matters (non-diagnostic)\n"
        f"End with: '{SAFETY_FOOTER}'\n\n"
        "DATA:\n" + json.dumps({
            "patient_id": patient.get("id"),
            "bed": patient.get("bed"),
            "dx": patient.get("dx"),
            "recent_vitals": (patient.get("vitals", []) or [])[-3:],
            "recent_labs": (patient.get("labs", []) or [])[-2:],
            "recent_notes": [{"t": n.get("t"), "author": n.get("author"), "text": scrub_phi(n.get("text",""))}
                             for n in (patient.get("notes", []) or [])[-2:]],
            "detected_changes": bullets,
        }, indent=2)
    )

    enriched = gen_text(prompt, max_new_tokens=220)
    changed_text = enriched if enriched else ("Change summary:\n- " + "\n- ".join(bullets) + f"\n\n{SAFETY_FOOTER}")

    detail = {
        "id": patient.get("id"),
        "bed": patient.get("bed"),
        "dx": patient.get("dx"),
        "risk_score": heur.get("score"),
        "risk_label": heur.get("label"),
        "confidence": heur.get("confidence"),
        "top_drivers": heur.get("drivers"),
        "evidence_sources": heur.get("modality"),
        "constraints": SAFETY_FOOTER + " No medication advice."
    }

    return fig, changed_text, json.dumps(detail, indent=2), json.dumps(heur.get("drivers", []), indent=2)


# -------------------------
# Outputs
# -------------------------
def night_rounds_brief(board_json: str) -> str:
    init_engine()
    try:
        board = json.loads(board_json) if board_json else []
        if not isinstance(board, list) or not board:
            return "Build the ward board first."
    except Exception:
        return "Build the ward board first."

    prompt = (
        "Create a night rounds brief.\n"
        "Do NOT diagnose. Do NOT recommend medications.\n"
        "Include:\n"
        "1) Top 5 rounding order\n"
        "2) One-line 'why' per patient\n"
        "3) 3 non-diagnostic safety reminders\n"
        f"End with: '{SAFETY_FOOTER}'\n\n"
        "BOARD (top 10):\n" + json.dumps(board[:10], indent=2)
    )
    out = gen_text(prompt, max_new_tokens=260)
    if out:
        return out

    lines = ["NOX Night Rounds Brief\n"]
    for i, p in enumerate(board[:5], start=1):
        lines.append(f"{i}) {p['bed']} ({p['id']}): {p['final_label']} • {p['risk_pct']} — {p['why']}")
    lines.append(f"\n{SAFETY_FOOTER}")
    return "\n".join(lines)


def sbar_packet(ward_json: str, patient_id: str) -> str:
    init_engine()
    try:
        ward = json.loads(ward_json)
        patients = ward.get("patients", [])
    except Exception:
        return "Invalid ward JSON."

    pid = (patient_id or "").strip()
    patient = next((p for p in patients if p.get("id") == pid), None)
    if not patient:
        return f"Patient {pid} not found."

    _timeline, heur = heuristic_risk(patient)

    prompt = (
        "Write an SBAR packet for escalation-ready communication.\n"
        "Do NOT diagnose. Do NOT recommend medications.\n"
        "Use concise clinical style.\n"
        f"End with: '{SAFETY_FOOTER}'\n\n"
        "DATA:\n" + json.dumps({
            "patient_id": patient.get("id"),
            "bed": patient.get("bed"),
            "dx": patient.get("dx"),
            "recent_vitals": (patient.get("vitals", []) or [])[-3:],
            "recent_labs": (patient.get("labs", []) or [])[-2:],
            "recent_notes": [{"t": n.get("t"), "author": n.get("author"), "text": scrub_phi(n.get("text",""))}
                             for n in (patient.get("notes", []) or [])[-2:]],
            "risk_label": heur.get("label"),
            "drivers": (heur.get("drivers", []) or [])[:4],
        }, indent=2)
    )

    out = gen_text(prompt, max_new_tokens=260)
    if out:
        return out

    drivers = (heur.get("drivers", []) or [])[:4]
    base = [
        "NOX SBAR\n",
        f"S: Elevated deterioration risk ({heur.get('label','LOW')}).",
        f"B: Dx: {patient.get('dx')}",
        "A:",
        *[f"- {d.get('key')}: {d.get('evidence','')}" for d in drivers],
        "R:",
        "- Reassess, increase monitoring, escalate per local protocol if red flags.",
        SAFETY_FOOTER
    ]
    return "\n".join(base)


# -------------------------
# Chat (Gradio 6: Chatbot expects list of {role,content})
# -------------------------
def chatbot_respond(board_json: str, user_msg: str, chat_hist):
    init_engine()

    if chat_hist is None:
        chat_hist = []

    msg = (user_msg or "").strip()
    if not msg:
        return chat_hist, chat_hist, ""

    try:
        board = json.loads(board_json) if board_json else []
    except Exception:
        board = []

    # Append user message (dict format)
    chat_hist = chat_hist + [{"role": "user", "content": msg}]

    # If engine not ready, deterministic fallback
    if not _ENGINE["ready"]:
        if board:
            top = board[0]
            answer = (
                f"Top priority: {top.get('bed')} ({top.get('id')}) — "
                f"{top.get('final_label')} • {top.get('risk_pct')}.\n"
                f"Reason: {top.get('why')}.\n{SAFETY_FOOTER}"
            )
        else:
            answer = "Build the ward board first."
        chat_hist = chat_hist + [{"role": "assistant", "content": answer}]
        return chat_hist, chat_hist, ""

    prompt = (
        "You are NOX, a night operations copilot.\n"
        "Do NOT diagnose. Do NOT recommend medications.\n"
        "Be evidence-grounded using the ward board.\n"
        "If asked who/why, reference drivers and evidence sources.\n"
        f"End with: '{SAFETY_FOOTER}'\n\n"
        f"USER:\n{msg}\n\n"
        f"BOARD (top 10):\n{json.dumps(board[:10], indent=2)}\n"
    )

    answer = gen_text(prompt, max_new_tokens=240)
    if not answer:
        answer = f"Engine is warming up. Try again shortly.\n{SAFETY_FOOTER}"

    chat_hist = chat_hist + [{"role": "assistant", "content": answer}]
    return chat_hist, chat_hist, ""


# -------------------------
# UI (Gradio 6: put theme/css into launch(), not Blocks())
# -------------------------
app = gr.Blocks(title="NOX — Night Operations Copilot")

with app:
    gr.Markdown(
        """
<div id="nox_header"><h1>NOX</h1></div>
<div id="nox_subtitle">
  Night Operations Copilot — prioritization, change detection, and escalation-ready communication.
</div>
<br/>
"""
    )

    gr.Markdown(
        f"""
<div class="nox-card">
  <b>Scope:</b> Decision-support prototype for prioritization and communication. {SAFETY_FOOTER} No medication advice.
</div>
<br/>
"""
    )

    with gr.Row():
        btn_test = gr.Button("Test engine", variant="primary")
        engine_md = gr.Markdown("Engine: Unknown (click Test engine).")
    btn_test.click(fn=engine_status, inputs=[], outputs=[engine_md])

    with gr.Row():
        ward_file = gr.File(label="Upload ward JSON file", file_types=[".json"])
        btn_load_file = gr.Button("Load uploaded file", variant="secondary")

    ward_json = gr.Textbox(label="Ward JSON", lines=12, placeholder="Upload ward JSON or paste it here…")
    btn_load_file.click(fn=load_from_file, inputs=[ward_file], outputs=[ward_json])

    btn_board = gr.Button("Build ward board", variant="primary")
    board_status = gr.Markdown("Board: Not built yet.")

    board_table = gr.Dataframe(
        headers=["Bed", "Patient ID", "Priority", "Risk", "Confidence", "Why", "Evidence sources", "Mode"],
        datatype=["str", "str", "str", "str", "str", "str", "str", "str"],
        interactive=False,
        label="Ward board"
    )

    board_json_state = gr.State(value="")
    top_patient_state = gr.State(value="")
    heatmap_plot = gr.Plot(label="Ward heatmap")

    btn_board.click(
        fn=build_board,
        inputs=[ward_json],
        outputs=[board_table, board_json_state, heatmap_plot, top_patient_state, board_status],
    )

    # Ask NOX (NO type parameter in Gradio 6)
    gr.Markdown("## Ask NOX")
    chat = gr.Chatbot(label="NOX Chat", height=280)  # <-- no type kwarg
    chat_state = gr.State(value=[])

    with gr.Row():
        chat_in = gr.Textbox(
            label="Message",
            placeholder="Ask: Who is most concerning and why? What changed? What should I check next?",
            lines=2
        )
        chat_send = gr.Button("Send", variant="primary")

    chat_send.click(
        fn=chatbot_respond,
        inputs=[board_json_state, chat_in, chat_state],
        outputs=[chat, chat_state, chat_in],
    )

    # Focus mode
    gr.Markdown("## Focus mode")
    with gr.Row():
        patient_id = gr.Textbox(label="Patient ID", value="pt-001")
        btn_use_top = gr.Button("Use top priority", variant="secondary")
        btn_open = gr.Button("Open patient", variant="secondary")

    btn_use_top.click(fn=lambda top: top or "pt-001", inputs=[top_patient_state], outputs=[patient_id])

    risk_plot = gr.Plot(label="Timeline")
    changed_text = gr.Textbox(label="What changed", lines=10)
    detail_json = gr.Code(label="Patient detail", language="json")
    drivers_json = gr.Code(label="Drivers", language="json")

    btn_open.click(
        fn=open_patient,
        inputs=[ward_json, patient_id],
        outputs=[risk_plot, changed_text, detail_json, drivers_json]
    )

    # One-click outputs
    gr.Markdown("## One-click outputs")
    with gr.Row():
        btn_brief = gr.Button("Generate night rounds brief", variant="primary")
        brief_out = gr.Textbox(label="Night rounds brief", lines=12)
    btn_brief.click(fn=night_rounds_brief, inputs=[board_json_state], outputs=[brief_out])

    with gr.Row():
        btn_sbar = gr.Button("Generate SBAR packet", variant="primary")
        sbar_out = gr.Textbox(label="SBAR packet", lines=12)
    btn_sbar.click(fn=sbar_packet, inputs=[ward_json, patient_id], outputs=[sbar_out])


# Gradio 6: theme/css passed to launch
app.launch(theme=theme, css=CUSTOM_CSS)