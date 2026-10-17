# Architecture Diagram Generator

Describe a software system in plain text, even with typos or missing parts, and get a
clean, interactive architecture diagram. Built for an **LSEG mini-hackathon** (2026).

## How it works
```
text description → Gemini (structured JSON output) → graph (NetworkX)
                 → layered layout → interactive diagram (Matplotlib in CustomTkinter)
```
1. **Understanding the text.** Gemini receives the description with a system prompt
   that fixes typos, finds every component (including implied ones), fills in gaps in
   the architecture and classifies each component. The answer is forced into a
   **Pydantic schema** (`ArchitectureDiagram`), so the output is always valid JSON.
2. **Resilient model calls.** If a model is rate-limited (`429`), the app waits and
   retries, then falls back through `gemini-2.0-flash` → `gemini-2.0-flash-lite` → `gemini-2.5-flash`.
3. **Layout.** A custom BFS-based layered layout places the input at the top and the
   output at the bottom, without needing Graphviz.
4. **Drawing.** Each component type gets its own shape and colour:

| Type | Shape | Default colour |
|---|---|---|
| input / output | Oval | Cyan / Brown |
| processor | Rectangle | Blue |
| classifier (e.g. orchestrator) | Diamond | Green |
| storage | Cylinder | Orange |
| tool (guardrails, loggers…) | Hexagon | Red |
| api | Parallelogram | Purple |

Colours can be changed straight from the text (e.g. *"make the databases purple"*).

## Features
- Desktop UI with dark theme (CustomTkinter)
- **Drag and drop** nodes to rearrange the diagram; arrows follow
- PNG export of the current layout (to `output/diagram.png`)
- Status messages while the model is working or waiting on rate limits

## Example input
> users send requests thru an api gateway, an orchestrator decides if its a search or a
> sumary request, results get cached in redis and a guardrail checks the final answer.
> make storage purple

## Run it
Requires Python 3.10+ and a free Gemini API key from [Google AI Studio](https://aistudio.google.com/).

```bash
pip install -r requirements.txt
cp .env.example .env        # then put your key in .env
python concurs.py
```
