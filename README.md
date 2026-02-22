# NOX — Night Operations Copilot

NOX is a ward-level prioritization and communication engine designed for hospital night operations.

## What It Does

- Ranks patients by deterioration risk
- Detects recent change patterns
- Surfaces interpretable drivers
- Visualizes ward risk with a heatmap
- Generates:
  - Night Rounds Brief
  - SBAR escalation packet
- Provides conversational interaction through Ask NOX

## Scope

Decision-support prototype for prioritization and communication.
Not a diagnosis system. Follow local escalation protocols.

## Demo

Live Demo:
[Hugging Face Space Link Here]

## Example Flow

1. Upload ward JSON file
2. Build ward board
3. Open top priority patient
4. Generate SBAR packet

## Built With

- Python
- Gradio
- NumPy
- Pandas
- Matplotlib
- Hugging Face Spaces

## Future Work

- FHIR integration
- Role-aware dashboards
- Calibration & uncertainty modeling
- Prospective validation

---

Developed during a hackathon.
