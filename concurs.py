"""
Architecture Diagram Generator
UI: CustomTkinter  |  Interactive diagram with drag-and-drop nodes
"""

import os
import sys
import time
import threading
import subprocess
from collections import deque

import numpy as np
from google import genai
from google.genai import types
import networkx as nx
import matplotlib
matplotlib.use("Agg")          # default backend (subprocess PNG export still works)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image
import customtkinter as ctk
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

# ── Gemini client ──────────────────────────────────────────────────────────
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# ── Pydantic models ────────────────────────────────────────────────────────

class Component(BaseModel):
    id: str
    label: str
    type: str
    description: str

class Relationship(BaseModel):
    from_id: str
    to_id: str
    label: Optional[str] = None

class ColorOverride(BaseModel):
    component_type: str
    hex_color: str

class ArchitectureDiagram(BaseModel):
    title: str
    components: List[Component]
    relationships: List[Relationship]
    cleaned_description: str
    color_overrides: List[ColorOverride] = []

# ── Default colours ────────────────────────────────────────────────────────

DEFAULT_TYPE_COLORS = {
    "tool":       "#E53935",
    "classifier": "#43A047",
    "processor":  "#1E88E5",
    "storage":    "#FB8C00",
    "api":        "#8E24AA",
    "input":      "#00ACC1",
    "output":     "#6D4C41",
}
DEFAULT_COLOR = "#607D8B"

# Shape name for legend labels
SHAPE_NAMES = {
    "storage":    "Cylinder",
    "classifier": "Diamond",
    "tool":       "Hexagon",
    "input":      "Oval",
    "output":     "Oval",
    "api":        "Parallelogram",
    "processor":  "Rectangle",
}

# ── System prompt ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a software architect. You receive a text description of a system
architecture, possibly with typos, grammatical errors, vague terms, or missing parts.

Your tasks:
1. Fix all typos and grammatical errors silently.
2. Identify ALL components — both explicitly mentioned and logically implied.
3. Fill in any architectural gaps to produce a complete, coherent, logical system.
4. Classify each component by type:
   - "tool": external tools, guardrails, validators, loggers
   - "classifier": decision/classification components (Orchestrator also counts)
   - "processor": processing/transformation components
   - "storage": databases, caches, storage
   - "api": API endpoints or integrations
   - "input": system entry point
   - "output": final output
5. Extract color preferences and populate color_overrides.
   Map color names: red→#E53935, green→#43A047, blue→#1E88E5, orange→#FB8C00,
   purple→#8E24AA, yellow→#F9A825, pink→#E91E63, cyan→#00ACC1.

Rules:
- Unique snake_case id per component.
- Relationships form a logical flow from input toward output.
- No isolated nodes.
- If ambiguous, make sensible architectural decisions — do NOT leave gaps.
- color_overrides: only types the user explicitly coloured."""

# ── Gemini extract ─────────────────────────────────────────────────────────

_MODELS = [
    ("gemini-2.0-flash",      {}),
    ("gemini-2.0-flash-lite", {}),
    ("gemini-2.5-flash",      {"thinking_config": types.ThinkingConfig(thinking_budget=0)}),
]

def extract_architecture(user_text: str, _status_cb=None) -> ArchitectureDiagram:
    import time
    last_err = None
    for model, extra in _MODELS:
        if _status_cb:
            _status_cb(f"⏳  Procesez…  →  încerc {model}…")
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=f"Analyze this text and extract the system architecture:\n\n{user_text}",
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        response_schema=ArchitectureDiagram,
                        temperature=0.1,
                        **extra,
                    ),
                )
                return ArchitectureDiagram.model_validate_json(response.text)
            except Exception as e:
                last_err = e
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    wait = 15 * (attempt + 1)
                    if _status_cb:
                        _status_cb(f"⏳  429 pe {model} — aștept {wait}s…")
                    time.sleep(wait)
                else:
                    break
    raise last_err

def build_type_colors(arch: ArchitectureDiagram) -> dict:
    colors = dict(DEFAULT_TYPE_COLORS)
    for ov in arch.color_overrides:
        t = ov.component_type.lower().strip()
        if t in colors:
            colors[t] = ov.hex_color
    return colors

# ── Layered layout (no graphviz) ───────────────────────────────────────────

def layered_layout(G, h_gap=3.6, v_gap=2.6):
    if not G.nodes():
        return {}
    layers = {}
    roots = [n for n in G.nodes() if G.in_degree(n) == 0]
    if not roots:
        roots = [next(iter(G.nodes()))]
    q = deque()
    for r in roots:
        layers[r] = 0
        q.append(r)
    while q:
        node = q.popleft()
        for s in G.successors(node):
            nl = layers[node] + 1
            if s not in layers or layers[s] < nl:
                layers[s] = nl
                q.append(s)
    for n in G.nodes():
        if n not in layers:
            layers[n] = 0
    layer_groups: dict = {}
    for n, l in layers.items():
        layer_groups.setdefault(l, []).append(n)
    pos = {}
    for layer, nodes in layer_groups.items():
        total = (len(nodes) - 1) * h_gap
        for i, n in enumerate(nodes):
            pos[n] = (i * h_gap - total / 2, -layer * v_gap)
    return pos

# ── Drawing helpers ────────────────────────────────────────────────────────

def _tc(hex_color: str) -> str:
    """Return black or white text colour depending on background luminance."""
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return "#1A1A1A" if (0.299*r + 0.587*g + 0.114*b) / 255 > 0.62 else "white"

def _draw_arrow(ax, src, dst, bw, bh, label=None):
    sx, sy = src
    tx, ty = dst
    tol = bh * 0.1
    if sy - ty > tol:
        start, end, cs = (sx, sy - bh/2), (tx, ty + bh/2), "arc3,rad=0"
    elif ty - sy > tol:
        start, end, cs = (sx, sy + bh/2), (tx, ty - bh/2), "arc3,rad=0.4"
    else:
        if sx < tx:
            start, end = (sx + bw/2, sy), (tx - bw/2, ty)
        else:
            start, end = (sx - bw/2, sy), (tx + bw/2, ty)
        cs = "arc3,rad=0.25"
    ax.annotate("", xy=end, xytext=start,
        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.6,
                        mutation_scale=14, connectionstyle=cs), zorder=2)
    if label:
        mx, my = (start[0]+end[0])/2, (start[1]+end[1])/2
        ax.text(mx, my, label, ha="center", va="center", fontsize=7.5,
                color="#333", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white",
                          ec="#BBB", lw=0.8, alpha=0.92))

# ── Shape-specific node drawers ────────────────────────────────────────────

def _draw_rect(ax, x, y, label, color, bw, bh):
    """Processor / default: rounded rectangle."""
    ax.add_patch(mpatches.FancyBboxPatch(
        (x-bw/2, y-bh/2), bw, bh, boxstyle="round,pad=0.08",
        facecolor=color, edgecolor="#222", linewidth=1.6, zorder=3))
    ax.text(x, y, label, ha="center", va="center",
            fontsize=9, fontweight="bold", color=_tc(color), zorder=4)

def _draw_cylinder(ax, x, y, label, color, bw, bh):
    """Storage: cylinder."""
    ey = bh * 0.28
    # Rectangle body (no border on sides — drawn separately)
    ax.add_patch(mpatches.FancyBboxPatch(
        (x-bw/2, y-bh/2), bw, bh, boxstyle="square,pad=0",
        facecolor=color, edgecolor="none", zorder=3))
    ax.plot([x-bw/2, x-bw/2], [y-bh/2, y+bh/2], color="#222", lw=1.6, zorder=3)
    ax.plot([x+bw/2, x+bw/2], [y-bh/2, y+bh/2], color="#222", lw=1.6, zorder=3)
    # Bottom ellipse
    ax.add_patch(mpatches.Ellipse((x, y-bh/2), bw, ey*2,
        facecolor=color, edgecolor="#222", lw=1.6, zorder=3))
    # Top ellipse (drawn on top to cover rectangle edge)
    ax.add_patch(mpatches.Ellipse((x, y+bh/2), bw, ey*2,
        facecolor=color, edgecolor="#222", lw=1.6, zorder=4))
    ax.text(x, y, label, ha="center", va="center",
            fontsize=9, fontweight="bold", color=_tc(color), zorder=5)

def _draw_diamond(ax, x, y, label, color, bw, bh):
    """Classifier / decision: diamond."""
    pts = np.array([(x, y+bh*0.85), (x+bw*0.65, y),
                    (x, y-bh*0.85), (x-bw*0.65, y)])
    ax.add_patch(mpatches.Polygon(pts, closed=True,
        facecolor=color, edgecolor="#222", lw=1.6, zorder=3))
    ax.text(x, y, label, ha="center", va="center",
            fontsize=8, fontweight="bold", color=_tc(color), zorder=4)

def _draw_hexagon(ax, x, y, label, color, bw, bh):
    """Tool: hexagon."""
    angles = np.linspace(0, 2*np.pi, 7)[:-1] + np.pi/6
    pts = np.column_stack([x + bw*0.55*np.cos(angles),
                           y + bh*0.88*np.sin(angles)])
    ax.add_patch(mpatches.Polygon(pts, closed=True,
        facecolor=color, edgecolor="#222", lw=1.6, zorder=3))
    ax.text(x, y, label, ha="center", va="center",
            fontsize=8, fontweight="bold", color=_tc(color), zorder=4)

def _draw_oval(ax, x, y, label, color, bw, bh):
    """Input / Output: oval."""
    ax.add_patch(mpatches.Ellipse((x, y), bw*1.05, bh*1.3,
        facecolor=color, edgecolor="#222", lw=1.6, zorder=3))
    ax.text(x, y, label, ha="center", va="center",
            fontsize=9, fontweight="bold", color=_tc(color), zorder=4)

def _draw_parallelogram(ax, x, y, label, color, bw, bh):
    """API: parallelogram."""
    s = bw * 0.16
    pts = np.array([(x-bw/2+s, y+bh/2), (x+bw/2+s, y+bh/2),
                    (x+bw/2-s, y-bh/2), (x-bw/2-s, y-bh/2)])
    ax.add_patch(mpatches.Polygon(pts, closed=True,
        facecolor=color, edgecolor="#222", lw=1.6, zorder=3))
    ax.text(x, y, label, ha="center", va="center",
            fontsize=9, fontweight="bold", color=_tc(color), zorder=4)

_SHAPE_MAP = {
    "storage":    _draw_cylinder,
    "classifier": _draw_diamond,
    "tool":       _draw_hexagon,
    "input":      _draw_oval,
    "output":     _draw_oval,
    "api":        _draw_parallelogram,
    "processor":  _draw_rect,
}

def _draw_node(ax, x, y, label, color, node_type, bw, bh):
    _SHAPE_MAP.get(node_type, _draw_rect)(ax, x, y, label, color, bw, bh)

# ── Diagram generation (used by --draw subprocess & export) ────────────────

def generate_diagram(arch: ArchitectureDiagram, type_colors: dict,
                     output_path: str = "output/diagram.png"):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    G = nx.DiGraph()
    comp_map = {c.id: c for c in arch.components}
    valid_ids = set(comp_map)
    for comp in arch.components:
        G.add_node(comp.id)
    edge_labels = {}
    for rel in arch.relationships:
        if rel.from_id in valid_ids and rel.to_id in valid_ids:
            G.add_edge(rel.from_id, rel.to_id)
            if rel.label:
                edge_labels[(rel.from_id, rel.to_id)] = rel.label

    pos = layered_layout(G)
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    BW, BH = 2.8, 0.75
    fig_w = max(12, (max(all_x)-min(all_x)) + BW*3)
    fig_h = max(8,  abs(min(all_y)-max(all_y)) + BH*5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for u, v in G.edges():
        _draw_arrow(ax, pos[u], pos[v], BW, BH, edge_labels.get((u, v)))
    for node in G.nodes():
        comp = comp_map[node]
        _draw_node(ax, *pos[node], comp.label,
                   type_colors.get(comp.type, DEFAULT_COLOR),
                   comp.type, BW, BH)

    ax.set_title(arch.title, fontsize=14, fontweight="bold", color="#111", pad=16)
    desc = arch.cleaned_description
    if len(desc) > 140:
        desc = desc[:140] + "..."
    fig.text(0.5, 0.005, desc, ha="center", fontsize=8,
             color="#666", style="italic")

    present = {comp_map[n].type for n in G.nodes()}
    patches = [
        mpatches.Patch(facecolor=c, edgecolor="#333", lw=1,
                       label=f"{t.capitalize()} ({SHAPE_NAMES.get(t, '■')})")
        for t, c in type_colors.items() if t in present
    ]
    ax.legend(handles=patches, loc="upper right", facecolor="white",
              edgecolor="#CCC", fontsize=9, framealpha=0.96,
              title="Component Types", title_fontsize=9.5)

    ax.axis("off")
    ax.set_xlim(min(all_x)-BW*1.5, max(all_x)+BW*1.5)
    ax.set_ylim(min(all_y)-BH*3,   max(all_y)+BH*3)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(output_path, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════
# Interactive Diagram Widget
# ══════════════════════════════════════════════════════════════════════════

class InteractiveDiagram:
    """Matplotlib figure embedded via FigureCanvasTkAgg with drag-and-drop nodes."""
    BW = 2.8    # node width  (data units)
    BH = 0.78   # node height (data units)

    def __init__(self, parent):
        self.fig = Figure(facecolor="#0D0D1A")
        self.ax  = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.widget = self.canvas.get_tk_widget()
        self.widget.configure(bg="#0D0D1A")

        self.arch        = None
        self.type_colors = {}
        self.G           = None
        self.comp_map    = {}
        self.edge_labels = {}
        self.pos         = {}   # node_id → [x, y]  mutable list for dragging

        self._dragging    = None
        self._drag_last   = None
        self._last_redraw = 0.0

        self.canvas.mpl_connect("button_press_event",   self._on_press)
        self.canvas.mpl_connect("motion_notify_event",  self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

        self._draw_placeholder()

    # ── Placeholder ─────────────────────────────────────────────────────────

    def _draw_placeholder(self):
        self.ax.cla()
        self.ax.set_facecolor("#0D0D1A")
        self.fig.set_facecolor("#0D0D1A")
        self.ax.text(0.5, 0.5,
            "Descrie arhitectura în caseta de mai jos și apasă  Generate  →",
            ha="center", va="center", transform=self.ax.transAxes,
            fontsize=13, color="#44446A")
        self.ax.axis("off")
        self.canvas.draw_idle()

    # ── Load diagram ─────────────────────────────────────────────────────────

    def load(self, arch: ArchitectureDiagram, type_colors: dict):
        self.arch        = arch
        self.type_colors = type_colors
        self.comp_map    = {c.id: c for c in arch.components}
        valid_ids        = set(self.comp_map)

        self.G = nx.DiGraph()
        for comp in arch.components:
            self.G.add_node(comp.id)
        self.edge_labels = {}
        for rel in arch.relationships:
            if rel.from_id in valid_ids and rel.to_id in valid_ids:
                self.G.add_edge(rel.from_id, rel.to_id)
                if rel.label:
                    self.edge_labels[(rel.from_id, rel.to_id)] = rel.label

        raw = layered_layout(self.G)
        self.pos = {k: list(v) for k, v in raw.items()}   # mutable lists!
        self._redraw()

    # ── Redraw ───────────────────────────────────────────────────────────────

    def _redraw(self):
        self.ax.cla()
        self.ax.set_facecolor("white")
        self.fig.set_facecolor("white")

        if not self.pos:
            self._draw_placeholder()
            return

        all_x = [p[0] for p in self.pos.values()]
        all_y = [p[1] for p in self.pos.values()]

        for u, v in self.G.edges():
            _draw_arrow(self.ax, tuple(self.pos[u]), tuple(self.pos[v]),
                        self.BW, self.BH, self.edge_labels.get((u, v)))

        for nid in self.G.nodes():
            c = self.comp_map[nid]
            _draw_node(self.ax, self.pos[nid][0], self.pos[nid][1],
                       c.label, self.type_colors.get(c.type, DEFAULT_COLOR),
                       c.type, self.BW, self.BH)

        self.ax.set_title(self.arch.title, fontsize=14, fontweight="bold",
                          color="#111", pad=16)

        present = {self.comp_map[n].type for n in self.G.nodes()}
        patches = [
            mpatches.Patch(facecolor=col, edgecolor="#333", lw=1,
                           label=f"{t.capitalize()} ({SHAPE_NAMES.get(t, '■')})")
            for t, col in self.type_colors.items() if t in present
        ]
        self.ax.legend(handles=patches, loc="upper right", facecolor="white",
                       edgecolor="#CCC", fontsize=9, framealpha=0.96,
                       title="Component Types", title_fontsize=9.5)

        self.ax.axis("off")
        self.ax.set_xlim(min(all_x) - self.BW*1.5, max(all_x) + self.BW*1.5)
        self.ax.set_ylim(min(all_y) - self.BH*3,   max(all_y) + self.BH*3)
        self.canvas.draw_idle()

    # ── Drag-and-drop ────────────────────────────────────────────────────────

    def _hit_test(self, xd, yd):
        """Return the node_id under the cursor, or None."""
        if xd is None or yd is None:
            return None
        for nid, pos in self.pos.items():
            if abs(xd - pos[0]) < self.BW * 0.65 and abs(yd - pos[1]) < self.BH * 1.1:
                return nid
        return None

    def _on_press(self, event):
        if event.inaxes != self.ax or event.button != 1 or not self.pos:
            return
        hit = self._hit_test(event.xdata, event.ydata)
        if hit:
            self._dragging  = hit
            self._drag_last = (event.xdata, event.ydata)

    def _on_motion(self, event):
        if self._dragging is None or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        dx = event.xdata - self._drag_last[0]
        dy = event.ydata - self._drag_last[1]
        self.pos[self._dragging][0] += dx
        self.pos[self._dragging][1] += dy
        self._drag_last = (event.xdata, event.ydata)
        # Throttle to ~30 fps — avoids flooding matplotlib with redraws
        now = time.monotonic()
        if now - self._last_redraw >= 0.033:
            self._last_redraw = now
            self._redraw()

    def _on_release(self, event):
        if self._dragging is not None:
            self._redraw()   # final redraw at exact drop position
        self._dragging  = None
        self._drag_last = None

    # ── Export current state to PNG ──────────────────────────────────────────

    def export_png(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.fig.savefig(path, dpi=100, bbox_inches="tight", facecolor="white")

# ══════════════════════════════════════════════════════════════════════════
# CustomTkinter UI
# ══════════════════════════════════════════════════════════════════════════

_THIS   = os.path.abspath(__file__)
_VIEWER = os.path.join(os.path.dirname(_THIS), "diagram_viewer.py")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Architecture Diagram Generator")
        self.geometry("1280x820")
        self.minsize(900, 650)
        self._generating = False
        self._output_path = os.path.abspath("output/diagram.png")
        self._diagram: InteractiveDiagram | None = None
        self._build_ui()

    # ── Build UI ───────────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1)

        # ── Diagram area ───────────────────────────────────────────────────
        self.diagram_frame = ctk.CTkFrame(
            self, corner_radius=12, fg_color="#0D0D1A")
        self.diagram_frame.grid(
            row=0, column=0, sticky="nsew", padx=14, pady=(14, 6))

        # Embed the interactive matplotlib canvas
        self._diagram = InteractiveDiagram(self.diagram_frame)
        self._diagram.widget.pack(fill="both", expand=True, padx=4, pady=4)

        # Open-in-DPG button (top-right, enabled after first diagram)
        self.dpg_btn = ctk.CTkButton(
            self.diagram_frame,
            text="⤢  Open in DPG",
            width=140, height=30,
            corner_radius=6,
            font=ctk.CTkFont(size=11),
            command=self._open_dpg_viewer,
            state="disabled",
        )
        self.dpg_btn.place(relx=1.0, rely=0.0, anchor="ne", x=-8, y=8)

        # ── Input area ─────────────────────────────────────────────────────
        input_frame = ctk.CTkFrame(self, corner_radius=12)
        input_frame.grid(
            row=1, column=0, sticky="ew", padx=14, pady=(6, 14))
        input_frame.grid_columnconfigure(0, weight=1)

        self.textbox = ctk.CTkTextbox(
            input_frame, height=85,
            font=ctk.CTkFont(family="Helvetica Neue", size=13),
            border_width=1, corner_radius=8,
        )
        self.textbox.grid(
            row=0, column=0, sticky="ew", padx=(12, 6), pady=(12, 4))

        btn_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        btn_frame.grid(row=0, column=1, sticky="ns", padx=(0, 12), pady=(12, 4))

        self.gen_btn = ctk.CTkButton(
            btn_frame, text="Generate",
            width=120, height=42, corner_radius=8,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._on_generate,
        )
        self.gen_btn.pack(expand=True, fill="y")

        self.status_lbl = ctk.CTkLabel(
            input_frame, text="",
            font=ctk.CTkFont(size=11), text_color="gray",
            anchor="w",
        )
        self.status_lbl.grid(
            row=1, column=0, columnspan=2, sticky="w",
            padx=14, pady=(0, 8))

    # ── Generate ───────────────────────────────────────────────────────────

    def _step(self, msg: str, color: str = "orange"):
        self.after(0, lambda: self.status_lbl.configure(text=msg, text_color=color))

    def _on_generate(self):
        if self._generating:
            return
        prompt = self.textbox.get("1.0", "end").strip()
        if not prompt:
            return
        self._generating = True
        self.gen_btn.configure(state="disabled", text="Generating…")
        self._step("⏳  Procesez cu Gemini 2.5 Flash…  →  trimit cererea…")
        threading.Thread(
            target=self._bg_generate, args=(prompt,), daemon=True).start()

    def _bg_generate(self, prompt):
        try:
            self._step("⏳  Procesez…  →  trimit cererea…")
            arch = extract_architecture(prompt, _status_cb=self._step)

            self._step(f"⏳  Procesez…  →  construiesc graful ({len(arch.components)} noduri)…")
            colors = build_type_colors(arch)

            # Hand off to main thread — no subprocess needed for display
            self.after(0, lambda a=arch, c=colors: self._on_done(a, c))
        except Exception as exc:
            self.after(0, lambda e=str(exc): self._on_error(e))

    def _on_done(self, arch, colors):
        self._diagram.load(arch, colors)
        self.status_lbl.configure(
            text=f"✓  {arch.title}  —  trage nodurile pentru a le rearanja",
            text_color="#43A047")
        self.gen_btn.configure(state="normal", text="Generate")
        self.dpg_btn.configure(state="normal")
        self._generating = False

    def _on_error(self, err):
        self.status_lbl.configure(
            text=f"Eroare: {err[:90]}", text_color="#E53935")
        self.gen_btn.configure(state="normal", text="Generate")
        self._generating = False

    # ── DPG viewer — exports current (dragged) state, then opens viewer ────

    def _open_dpg_viewer(self):
        if not (self._diagram and self._diagram.arch):
            return
        try:
            # Export current interactive state (respects any drag changes)
            self._diagram.export_png(self._output_path)
            subprocess.Popen(
                [sys.executable, _VIEWER, self._output_path],
                close_fds=True,
            )
        except Exception as exc:
            self.status_lbl.configure(
                text=f"DPG viewer error: {exc}", text_color="#E53935")


def _run_diagram_subprocess(arch: ArchitectureDiagram, colors: dict, output_path: str):
    """Kept for backward compatibility / standalone use."""
    import json
    payload = json.dumps({"arch": arch.model_dump(), "colors": colors,
                          "output_path": output_path})
    result = subprocess.run(
        [sys.executable, _THIS, "--draw"],
        input=payload, text=True, capture_output=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Subprocess draw failed")


def main():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("green")
    app = App()
    app.mainloop()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--draw":
        # ── Subprocess mode: read JSON from stdin, draw PNG, exit ──────────
        import json
        data = json.loads(sys.stdin.read())
        _arch   = ArchitectureDiagram.model_validate(data["arch"])
        _colors = data["colors"]
        _out    = data["output_path"]
        generate_diagram(_arch, _colors, _out)
    else:
        main()
