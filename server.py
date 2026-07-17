#!/usr/bin/env python3
"""
MCP Server für Bild-Analyse mit YOLO Object Detection und OCR.

Ausgabe als LLM-optimierter Markdown-Text (kein JSON), weil LLMs natürliche
Sprache besser verarbeiten als verschachtelte Strukturen.
"""

import asyncio
import os
import sys
import tempfile
from typing import Optional, List, Dict, Any, Tuple
from functools import partial
from pathlib import Path

# ---------------------------------------------------------------------------
# Selbsttest-Vorspann
# ---------------------------------------------------------------------------
# Python-Version und Abhängigkeiten werden geprüft, BEVOR die schweren Importe
# laufen. Andernfalls stirbt genau der Aufruf, der die Diagnose liefern soll,
# schon beim Import von httpx/numpy/PIL — der Nutzer sähe einen nackten
# Traceback statt der Liste dessen, was ihm fehlt.

_REQUIRED_PACKAGES = (
    ("mcp", "mcp"),
    ("pydantic", "pydantic"),
    ("httpx", "httpx"),
    ("numpy", "numpy"),
    ("PIL", "Pillow"),
    ("ultralytics", "ultralytics"),
    ("easyocr", "easyocr"),
)


def _preflight() -> int:
    """Prüft Python-Version und Verfügbarkeit aller Pakete, ohne sie zu importieren."""
    import platform
    import importlib.util

    print(f"Python {sys.version.split()[0]} auf {platform.system()} ({platform.machine()})")
    if sys.version_info < (3, 10):
        print(f"\nFEHLER: Python 3.10 oder neuer wird benötigt (gefunden: {sys.version.split()[0]}).")
        print("Behebung: Neueres Python installieren und den Selbsttest damit erneut starten.")
        return 1

    print("\n[1/4] Abhängigkeiten prüfen ...")
    missing = []
    for module_name, package_name in _REQUIRED_PACKAGES:
        try:
            found = importlib.util.find_spec(module_name) is not None
        except (ImportError, ValueError):
            found = False
        print(f"      {'OK  ' if found else 'FEHLT'} {package_name}")
        if not found:
            missing.append(package_name)

    if missing:
        print(f"\nFEHLER: Es fehlen: {', '.join(missing)}")
        print("Behebung:  pip install -r requirements.txt")
        return 1
    return 0


if "--selftest" in sys.argv:
    _preflight_result = _preflight()
    if _preflight_result != 0:
        sys.exit(_preflight_result)

import httpx
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, ConfigDict
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("vision_mcp")

# Deutsche Anführungszeichen als Konstanten (f-string nested quote issues)
LQ = "\u201e"  # „
RQ = "\u201c"  # "

# ---------------------------------------------------------------------------
# Meta-Fußnote (einmal am Ende jedes Berichts)
# ---------------------------------------------------------------------------

META_FOOTER = (
    "\n---\n"
    "Koordinatensystem: (0,0) oben links, +X nach rechts, +Y nach unten. bbox.x/y = obere linke Ecke.\n"
    "Konfidenz: high ≥0.75 (verlässlich), medium ≥0.50 (plausibel, prüfen), low <0.50 (unsicher).\n"
    "Zentrierung: zentriert ≤5% Abweichung, fast zentriert ≤10%, off-center >10% (vom Bildzentrum)."
)

_yolo_model = None
_ocr_reader = None

def _model_cache_dir() -> Path:
    """Plattformübliches Cache-Verzeichnis für die YOLO-Gewichte."""
    override = os.environ.get("VISION_MCP_MODEL_DIR")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.cache")
    return Path(base) / "vision-mcp"


YOLO_MODEL_NAME = os.environ.get("VISION_MCP_YOLO_MODEL", "yolov8n.pt")
OCR_LANGUAGES = os.environ.get("VISION_MCP_OCR_LANGS", "de,en").split(",")


def _resolve_yolo_weights() -> str:
    """Absoluter Pfad zu den YOLO-Gewichten im Cache-Verzeichnis.

    Ohne absoluten Pfad lädt ultralytics die Gewichte ins jeweilige Arbeitsverzeichnis —
    also bei jedem Aufruf aus einem neuen Projektordner erneut. Ein fester Cache-Pfad
    hält die Datei an genau einer Stelle, unabhängig davon, wo der Server gestartet wird.
    """
    if os.path.isabs(YOLO_MODEL_NAME):
        return YOLO_MODEL_NAME
    cache_dir = _model_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir / YOLO_MODEL_NAME)


def _get_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO(_resolve_yolo_weights())
    return _yolo_model


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(OCR_LANGUAGES, gpu=False)
    return _ocr_reader


def _preimport_heavy_modules() -> None:
    """Importiert easyocr und ultralytics im Main-Thread, bevor der Server läuft.

    Die Tools reichen die Modellaufrufe über `_run_sync` an Worker-Threads weiter.
    Findet dort der *erste* `import easyocr` statt, bleibt unter Windows der
    DLL-Load von `scipy.linalg.blas` dauerhaft stehen (easyocr → scipy.ndimage →
    scipy.special → scipy.linalg): Der Tool-Call antwortet nie, der Client läuft in
    seinen Idle-Timeout. Im Main-Thread ist derselbe Import unauffällig.

    Der Selbsttest lief nie hinein, weil er `_detect_text_sync` direkt aufruft —
    also im Main-Thread. Nur die Importe werden vorgezogen; die Gewichte bleiben
    lazy und kosten weiterhin erst beim ersten Aufruf Zeit.
    """
    import easyocr  # noqa: F401
    from ultralytics import YOLO  # noqa: F401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _load_image(source: str) -> Image.Image:
    if source.startswith(("http://", "https://")):
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(source, timeout=30.0)
            resp.raise_for_status()
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp.write(resp.content)
            tmp.close()
            img = Image.open(tmp.name).convert("RGB")
            os.unlink(tmp.name)
            return img
    path = Path(source).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Bild nicht gefunden: {path}")
    return Image.open(str(path)).convert("RGB")


def _run_sync(func, *args, **kwargs):
    loop = asyncio.get_running_loop()
    return loop.run_in_executor(None, partial(func, *args, **kwargs))


def _bbox_center(box: Dict[str, int]) -> Tuple[float, float]:
    return (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)


def _conf_level(value: float) -> str:
    if value >= 0.75:
        return "high"
    elif value >= 0.5:
        return "medium"
    return "low"


def _int_bbox(x1: float, y1: float, x2: float, y2: float) -> Dict[str, int]:
    return {"x": round(x1), "y": round(y1), "width": round(x2 - x1), "height": round(y2 - y1)}


def _centering_text(box: Dict[str, int], img_w: int, img_h: int) -> Tuple[str, str]:
    """Liefert (horizontal, vertikal) als zwei lesbare Strings."""
    cx, cy = _bbox_center(box)
    h_off = cx - img_w / 2
    v_off = cy - img_h / 2
    h_dev = round((h_off / img_w) * 100, 1)
    v_dev = round((v_off / img_h) * 100, 1)

    def describe(offset: float, dev: float, pos: str, neg: str) -> str:
        abs_dev = abs(dev)
        direction = pos if offset > 0 else neg
        px = round(abs(offset))
        if abs_dev <= 5:
            return f"zentriert ({abs_dev}% Abweichung)"
        elif abs_dev <= 10:
            return f"fast zentriert, {px}px nach {direction} ({abs_dev}%)"
        return f"off-center, {px}px nach {direction} ({abs_dev}%)"

    return (
        describe(h_off, h_dev, "rechts", "links"),
        describe(v_off, v_dev, "unten", "oben"),
    )


def _distance_sentence(a: Dict[str, int], b: Dict[str, int], a_name: str, b_name: str) -> str:
    """Liefert einen vollständigen Satz über die räumliche Beziehung."""
    ca = _bbox_center(a)
    cb = _bbox_center(b)
    ax2, ay2 = a["x"] + a["width"], a["y"] + a["height"]
    bx2, by2 = b["x"] + b["width"], b["y"] + b["height"]

    overlap_x = max(0, min(ax2, bx2) - max(a["x"], b["x"]))
    overlap_y = max(0, min(ay2, by2) - max(a["y"], b["y"]))
    overlap_area = overlap_x * overlap_y
    area_a = a["width"] * a["height"]
    area_b = b["width"] * b["height"]
    smaller_area = min(area_a, area_b) if min(area_a, area_b) > 0 else 1

    h_gap = max(0, max(b["x"] - ax2, a["x"] - bx2))
    v_gap = max(0, max(b["y"] - ay2, a["y"] - by2))

    h_off = cb[0] - ca[0]
    v_off = cb[1] - ca[1]
    h_of_b = "rechts von" if h_off > 0 else ("links von" if h_off < 0 else "auf gleicher X-Achse wie")
    v_of_b = "unterhalb von" if v_off > 0 else ("oberhalb von" if v_off < 0 else "auf gleicher Y-Achse wie")

    if overlap_area == 0:
        touches = (ax2 == b["x"] or bx2 == a["x"] or ay2 == b["y"] or by2 == a["y"])
        if touches:
            return f"{b_name} berührt {a_name} an einer Kante."
        parts = []
        if v_gap > 0:
            parts.append(f"{v_gap}px {v_of_b}")
        if h_gap > 0:
            parts.append(f"{h_gap}px {h_of_b}")
        return f"{b_name} liegt {' und '.join(parts)} {a_name}."
    if overlap_area >= smaller_area * 0.99:
        if area_a >= area_b:
            return f"{a_name} enthält {b_name} vollständig."
        return f"{b_name} enthält {a_name} vollständig."
    overlap_pct = round((overlap_area / smaller_area) * 100, 1)
    return f"{b_name} überlappt {a_name} ({overlap_pct}% des kleineren Elements sind überdeckt)."


def _pair_sort_key(a: Dict[str, int], b: Dict[str, int]) -> float:
    ax2, ay2 = a["x"] + a["width"], a["y"] + a["height"]
    bx2, by2 = b["x"] + b["width"], b["y"] + b["height"]
    h_gap = max(0, max(b["x"] - ax2, a["x"] - bx2))
    v_gap = max(0, max(b["y"] - ay2, a["y"] - by2))
    return (h_gap ** 2 + v_gap ** 2) ** 0.5


def _detect_objects_sync(img: Image.Image, confidence: float, classes: Optional[List[str]]) -> List[Dict[str, Any]]:
    model = _get_yolo_model()
    results = model(np.array(img), conf=confidence, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if classes and label.lower() not in [c.lower() for c in classes]:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = round(float(box.conf[0]), 2)
            detections.append({
                "kind": "object",
                "name": label,
                "confidence": conf,
                "level": _conf_level(conf),
                "bbox": _int_bbox(x1, y1, x2, y2),
            })
    return detections


def _detect_text_sync(img: Image.Image) -> List[Dict[str, Any]]:
    reader = _get_ocr_reader()
    results = reader.readtext(np.array(img))
    detections = []
    for (coords, text, conf) in results:
        xs = [p[0] for p in coords]
        ys = [p[1] for p in coords]
        conf = round(float(conf), 2)
        detections.append({
            "kind": "text",
            "name": text,
            "confidence": conf,
            "level": _conf_level(conf),
            "bbox": _int_bbox(min(xs), min(ys), max(xs), max(ys)),
        })
    return detections


def _display_name(det: Dict) -> str:
    """Name für den Bericht: Objekt in fett, Text in Anführungszeichen."""
    if det["kind"] == "object":
        return f"**{det['name']}**"
    return f"Text {LQ}{det['name']}{RQ}"


def _ref_name(det: Dict) -> str:
    """Kurzer Referenz-Name für Sätze."""
    if det["kind"] == "object":
        return det["name"]
    return f"{LQ}{det['name']}{RQ}"


def _format_element(det: Dict, num: int, img_w: int, img_h: int, match_note: str = "") -> str:
    """Rendert ein Element als Markdown-Block."""
    bbox = det["bbox"]
    h_center, v_center = _centering_text(bbox, img_w, img_h)
    area_pct = round((bbox["width"] * bbox["height"]) / (img_w * img_h) * 100, 1)
    return (
        f"{num}. {_display_name(det)} — Konfidenz: {det['level']} ({det['confidence']}){match_note}\n"
        f"   Position: x={bbox['x']}, y={bbox['y']} | Größe: {bbox['width']}×{bbox['height']} px | {area_pct}% des Bildes\n"
        f"   Horizontal: {h_center}\n"
        f"   Vertikal: {v_center}"
    )


# ---------------------------------------------------------------------------
# Element-Suche mit Match-Typ
# ---------------------------------------------------------------------------

def _find_element(name: str, objects: List[Dict], texts: List[Dict]) -> Optional[Dict[str, Any]]:
    name_lc = name.lower()
    # Exakter Klassen-Match
    for obj in objects:
        if obj["name"].lower() == name_lc:
            return {**obj, "match_type": "exact_class", "match_note": ""}
    # Exakter Text-Match
    for txt in texts:
        if txt["name"].lower() == name_lc:
            return {**txt, "match_type": "exact_text", "match_note": ""}
    # Partieller Text-Match
    for txt in texts:
        if name_lc in txt["name"].lower():
            note = f" [partielle Text-Übereinstimmung: Suchbegriff {LQ}{name}{RQ} in {LQ}{txt['name']}{RQ} gefunden]"
            return {**txt, "match_type": "partial_text", "match_note": note}
    # Partieller Klassen-Match
    for obj in objects:
        if name_lc in obj["name"].lower():
            note = f" [partielle Klassen-Übereinstimmung: {LQ}{name}{RQ} in {LQ}{obj['name']}{RQ} gefunden]"
            return {**obj, "match_type": "partial_class", "match_note": note}
    return None


def _not_found_message(term: str, objects: List[Dict], texts: List[Dict]) -> str:
    obj_list = ", ".join(o["name"] for o in objects) if objects else "(keine)"
    txt_list = ", ".join(f'{LQ}{t["name"]}{RQ}' for t in texts) if texts else "(keine)"
    return (
        f"FEHLER: Element {LQ}{term}{RQ} wurde im Bild nicht gefunden.\n\n"
        f"Erkannte Objekte: {obj_list}\n"
        f"Erkannte Texte: {txt_list}\n\n"
        f"Hinweis: Die Suche versucht der Reihe nach: exakte Objektklasse → exakter Text "
        f"→ Teilstring in Text → Teilstring in Klasse."
    )


# ---------------------------------------------------------------------------
# Input Models
# ---------------------------------------------------------------------------

class DetectObjectsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    image: str = Field(..., description="Pfad zum Bild oder URL")
    confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    classes: Optional[List[str]] = Field(default=None, description="Filter für Objektklassen")


class DetectTextInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    image: str = Field(..., description="Pfad zum Bild oder URL")


class MeasureDistanceInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    image: str = Field(..., description="Pfad zum Bild oder URL")
    element_a: str = Field(..., description="Erstes Element (Klasse oder Text-Inhalt)")
    element_b: str = Field(..., description="Zweites Element")
    confidence: float = Field(default=0.25, ge=0.0, le=1.0)


class CheckAlignmentInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    image: str = Field(..., description="Pfad zum Bild oder URL")
    element: str = Field(..., description="Element zum Prüfen")
    confidence: float = Field(default=0.25, ge=0.0, le=1.0)


class AnalyzeLayoutInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    image: str = Field(..., description="Pfad zum Bild oder URL")
    confidence: float = Field(default=0.25, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool(
    name="vision_detect_objects",
    annotations={"title": "Objekte im Bild erkennen", "readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def vision_detect_objects(params: DetectObjectsInput) -> str:
    """Erkennt Objekte in einem Bild (YOLOv8) und liefert einen Markdown-Bericht.

    Nutze dieses Tool wenn: Du wissen willst WELCHE Objekte in einem Bild sind,
    wo sie liegen, wie groß sie sind, ob sie zentriert sind. Ideal um Personen,
    Gegenstände oder Tiere in Fotos/Creatives zu lokalisieren.

    Parameter:
    - image: Lokaler Pfad (~/foo.png auch erlaubt) oder http(s)-URL
    - confidence: Minimale Konfidenz 0.0-1.0 (Standard 0.25; höher = nur sichere Erkennungen)
    - classes: Optional Filter (z.B. ['person', 'car']) — ohne Angabe alle Klassen

    Ausgabe (Markdown, kein JSON): Pro Objekt Klasse, Konfidenz-Level, Pixelposition,
    Größe, Flächenanteil, horizontale + vertikale Zentrierung als lesbare Sätze.
    Am Ende eine Fußnote mit Koordinatensystem- und Konfidenz-Legende.

    Limits: YOLOv8n erkennt nur die ~80 COCO-Klassen (person, car, dog, cat, chair,
    bottle, laptop, cell phone, book u.v.m.). KEINE UI-Elemente (Buttons, Logos,
    Icons), KEINE Design-Shapes. Für Text-Erkennung bitte vision_detect_text nutzen.
    """
    img = await _load_image(params.image)
    w, h = img.size
    objects = await _run_sync(_detect_objects_sync, img, params.confidence, params.classes)

    lines = [f"# Objekt-Erkennung", f"Bild: {w}×{h} px ({params.image})", ""]

    if not objects:
        lines.append("Keine Objekte erkannt (bei dieser Konfidenz-Schwelle).")
        return "\n".join(lines) + META_FOOTER

    lines.append(f"{len(objects)} Objekt(e) erkannt:")
    lines.append("")
    for i, obj in enumerate(objects, 1):
        lines.append(_format_element(obj, i, w, h))
        lines.append("")

    return "\n".join(lines) + META_FOOTER


@mcp.tool(
    name="vision_detect_text",
    annotations={"title": "Text im Bild erkennen (OCR)", "readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def vision_detect_text(params: DetectTextInput) -> str:
    """Erkennt Text in einem Bild via OCR (EasyOCR) und liefert einen Markdown-Bericht.

    Nutze dieses Tool wenn: Du wissen willst WELCHER Text in einem Bild steht
    (Überschriften, Labels, Buttons, eingebettete Worte) und wo genau. Ideal
    für Ad-Creatives, Screenshots, Landing-Pages, Print-Layouts.

    Parameter:
    - image: Lokaler Pfad oder http(s)-URL

    Ausgabe (Markdown, kein JSON): Pro Textblock erkannter Inhalt, Konfidenz-Level,
    Pixelposition, Größe, horizontale + vertikale Zentrierung als lesbare Sätze.

    Sprachen: Deutsch + Englisch (anpassbar via VISION_MCP_OCR_LANGS-Env).

    Limits: Kann bei stark stilisierten Schriften, Handschriften oder sehr
    kleinen/kontrastarmen Texten versagen. Gruppiert zusammengehörige Worte
    manchmal nicht so, wie ein Mensch erwarten würde.
    """
    img = await _load_image(params.image)
    w, h = img.size
    texts = await _run_sync(_detect_text_sync, img)

    lines = [f"# Text-Erkennung (OCR)", f"Bild: {w}×{h} px ({params.image})", ""]

    if not texts:
        lines.append("Kein Text erkannt.")
        return "\n".join(lines) + META_FOOTER

    lines.append(f"{len(texts)} Textblock/Textblöcke erkannt:")
    lines.append("")
    for i, txt in enumerate(texts, 1):
        lines.append(_format_element(txt, i, w, h))
        lines.append("")

    return "\n".join(lines) + META_FOOTER


@mcp.tool(
    name="vision_measure_distance",
    annotations={"title": "Abstand zwischen zwei Elementen messen", "readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def vision_measure_distance(params: MeasureDistanceInput) -> str:
    """Misst den Pixelabstand zwischen zwei benannten Elementen in einem Bild.

    Nutze dieses Tool wenn: Du konkret wissen willst, wie weit zwei Elemente
    voneinander entfernt sind (z.B. 'wie viel Abstand ist zwischen der
    Überschrift und der Person?' oder 'berühren sich Button und Headline?').

    Parameter:
    - image: Lokaler Pfad oder http(s)-URL
    - element_a / element_b: Bezeichner (Objektklasse wie 'person', 'car' ODER
      Text-Inhalt wie 'Willkommen' oder ein Teilstring davon)
    - confidence: Minimale Konfidenz für die Erkennung (Standard 0.25)

    Element-Suche (in dieser Reihenfolge): exakte Objektklasse → exakter Text
    → Teilstring in Text → Teilstring in Klasse. Bei partiellem Match wird im
    Bericht ein Hinweis ausgegeben.

    Ausgabe (Markdown, kein JSON): Zwei Element-Blöcke (A und B) mit Position
    und Konfidenz, plus ein vollständiger natürlichsprachlicher Satz über die
    räumliche Beziehung (z.B. 'Text „Überschrift" liegt 276px oberhalb von
    person.') — inkl. Hinweis bei Überlappung, Berührung oder Enthaltensein.

    Wenn ein Element nicht gefunden wird, listet der Bericht alle erkannten
    Objekte und Texte als Vorschläge auf.
    """
    img = await _load_image(params.image)
    w, h = img.size
    objects = await _run_sync(_detect_objects_sync, img, params.confidence, None)
    texts = await _run_sync(_detect_text_sync, img)

    a = _find_element(params.element_a, objects, texts)
    if not a:
        return _not_found_message(params.element_a, objects, texts)
    b = _find_element(params.element_b, objects, texts)
    if not b:
        return _not_found_message(params.element_b, objects, texts)

    sentence = _distance_sentence(a["bbox"], b["bbox"], _ref_name(a), _ref_name(b))

    lines = [
        "# Abstandsmessung",
        f"Bild: {w}×{h} px",
        "",
        "## Element A",
        _format_element(a, 1, w, h, a.get("match_note", "")),
        "",
        "## Element B",
        _format_element(b, 1, w, h, b.get("match_note", "")),
        "",
        "## Räumliche Beziehung",
        sentence,
    ]

    # Zusätzliche Detail-Info wenn hilfreich
    ax2, ay2 = a["bbox"]["x"] + a["bbox"]["width"], a["bbox"]["y"] + a["bbox"]["height"]
    bx2, by2 = b["bbox"]["x"] + b["bbox"]["width"], b["bbox"]["y"] + b["bbox"]["height"]
    h_gap = max(0, max(b["bbox"]["x"] - ax2, a["bbox"]["x"] - bx2))
    v_gap = max(0, max(b["bbox"]["y"] - ay2, a["bbox"]["y"] - by2))
    if h_gap == 0 and v_gap > 0:
        lines.append(f"(Horizontal keine Lücke — beide Elemente überlappen sich auf der X-Achse.)")
    elif v_gap == 0 and h_gap > 0:
        lines.append(f"(Vertikal keine Lücke — beide Elemente überlappen sich auf der Y-Achse.)")

    return "\n".join(lines) + META_FOOTER


@mcp.tool(
    name="vision_check_alignment",
    annotations={"title": "Ausrichtung/Zentrierung eines Elements prüfen", "readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def vision_check_alignment(params: CheckAlignmentInput) -> str:
    """Prüft ob ein einzelnes Element im Bild zentriert/ausgerichtet ist.

    Nutze dieses Tool wenn: Du gezielt wissen willst, ob ein bestimmtes Element
    (z.B. 'Ist die Überschrift horizontal zentriert?' oder 'Wo liegt die Person
    im Bild?') zentriert ist — auf welcher Achse und wie stark die Abweichung.

    Parameter:
    - image: Lokaler Pfad oder http(s)-URL
    - element: Bezeichner (Objektklasse oder Text-Inhalt/Teilstring)
    - confidence: Minimale Konfidenz für die Erkennung (Standard 0.25)

    Ausgabe (Markdown, kein JSON): Element-Block mit Position, Größe,
    Flächenanteil und zwei lesbaren Zentrierungs-Sätzen (horizontal + vertikal).

    Zentrierungs-Kategorien:
    - 'zentriert' (≤5% Abweichung vom Bildzentrum)
    - 'fast zentriert' (≤10%, mit Angabe der genauen Pixel-Abweichung und Richtung)
    - 'off-center' (>10%, mit Angabe der Pixel-Abweichung und Richtung)

    Wenn das Element nicht gefunden wird, listet der Bericht erkannte Objekte
    und Texte als Vorschläge auf.
    """
    img = await _load_image(params.image)
    w, h = img.size
    objects = await _run_sync(_detect_objects_sync, img, params.confidence, None)
    texts = await _run_sync(_detect_text_sync, img)

    elem = _find_element(params.element, objects, texts)
    if not elem:
        return _not_found_message(params.element, objects, texts)

    lines = [
        "# Ausrichtungsprüfung",
        f"Bild: {w}×{h} px",
        "",
        _format_element(elem, 1, w, h, elem.get("match_note", "")),
    ]

    return "\n".join(lines) + META_FOOTER


@mcp.tool(
    name="vision_analyze_layout",
    annotations={"title": "Komplette Layout-Analyse eines Bildes", "readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def vision_analyze_layout(params: AnalyzeLayoutInput) -> str:
    """Komplette Layout-Analyse eines Bildes in EINEM Aufruf (Objekte + Texte + Beziehungen).

    Nutze dieses Tool wenn: Du einen Gesamtüberblick über das Layout brauchst
    (z.B. bei Ad-Creatives, Landing-Pages, Screenshots), statt jedes Element
    einzeln zu prüfen. Spart Aufrufe gegenüber detect_objects + detect_text +
    mehrfachem measure_distance.

    Parameter:
    - image: Lokaler Pfad oder http(s)-URL
    - confidence: Minimale Konfidenz für Objekterkennung (Standard 0.25)

    Ausgabe (Markdown, kein JSON) — drei Abschnitte:
    1. Objekte: Liste aller erkannten Objekte mit Zentrierung und Größe
    2. Texte: Liste aller OCR-erkannten Texte mit Zentrierung und Größe
    3. Räumliche Beziehungen: Bis zu 20 Sätze über die nächsten Element-Paare
       (sortiert nach Nähe), z.B. 'Text „Headline" liegt 276px oberhalb von person.'

    Ideal für Fragen wie: 'Beschreibe das Layout des Creatives', 'Passt Text und
    Grafik zusammen?', 'Welche Elemente überlappen?'.

    Limits wie bei den Einzel-Tools: YOLO findet nur COCO-Klassen, OCR kann
    stilisierten Text übersehen.
    """
    img = await _load_image(params.image)
    w, h = img.size
    objects = await _run_sync(_detect_objects_sync, img, params.confidence, None)
    texts = await _run_sync(_detect_text_sync, img)

    lines = [
        f"# Layout-Analyse",
        f"Bild: {w}×{h} px ({params.image})",
        f"Erkannt: {len(objects)} Objekt(e), {len(texts)} Textblock/Textblöcke (insgesamt {len(objects) + len(texts)} Elemente).",
        "",
    ]

    if objects:
        lines.append("## Objekte")
        lines.append("")
        for i, obj in enumerate(objects, 1):
            lines.append(_format_element(obj, i, w, h))
            lines.append("")

    if texts:
        lines.append("## Texte")
        lines.append("")
        for i, txt in enumerate(texts, 1):
            lines.append(_format_element(txt, i, w, h))
            lines.append("")

    # Paarweise Abstände als natürliche Sätze
    all_elements = objects + texts
    if len(all_elements) >= 2:
        pairs = []
        for i, a in enumerate(all_elements):
            for j, b in enumerate(all_elements):
                if j <= i:
                    continue
                sentence = _distance_sentence(a["bbox"], b["bbox"], _ref_name(a), _ref_name(b))
                pairs.append((_pair_sort_key(a["bbox"], b["bbox"]), sentence))
        pairs.sort(key=lambda p: p[0])

        lines.append("## Räumliche Beziehungen (sortiert nach Nähe)")
        lines.append("")
        for _, sentence in pairs[:20]:
            lines.append(f"- {sentence}")
        lines.append("")

    return "\n".join(lines) + META_FOOTER


def _selftest() -> int:
    """Lädt die Modelle vor und analysiert ein erzeugtes Testbild.

    Gedacht als erster Schritt nach dem Auschecken: ein Befehl, der auf jedem
    Betriebssystem sagt, ob der Server einsatzbereit ist — statt den Fehler erst
    beim ersten Tool-Aufruf im Agenten zu sehen. Läuft bewusst ohne Netz-Zugriff,
    sobald die Modelle einmal im Cache liegen.

    Python-Version und Abhängigkeiten hat bereits _preflight() geprüft, bevor die
    Importe dieses Moduls liefen.
    """
    print(f"\n[2/4] YOLO-Gewichte laden ({YOLO_MODEL_NAME}) ...")
    print(f"      Ziel: {_resolve_yolo_weights()}")
    print("      (Beim ersten Mal wird heruntergeladen — das braucht Internet.)")
    _get_yolo_model()
    print("      OK")

    print(f"\n[3/4] OCR-Modelle laden (Sprachen: {', '.join(OCR_LANGUAGES)}) ...")
    print("      (Beim ersten Mal ~100 MB Download.)")
    _get_ocr_reader()
    print("      OK")

    print("\n[4/4] Testbild analysieren ...")
    from PIL import ImageDraw
    img = Image.new("RGB", (600, 300), "white")
    ImageDraw.Draw(img).text((250, 140), "VISION MCP", fill="black")
    texts = _detect_text_sync(img)
    _detect_objects_sync(img, 0.25, None)
    if not texts:
        print("      WARNUNG: OCR hat im Testbild keinen Text gefunden.")
        print("      Die Installation ist grundsätzlich in Ordnung, aber die Texterkennung")
        print("      lieferte kein Ergebnis — bitte mit einem echten Bild gegenprüfen.")
        return 1
    print(f"      OK — erkannter Text: {', '.join(t['name'] for t in texts)}")

    print("\nAlles bereit. Der Server kann als MCP-Server eingebunden werden.")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    _preimport_heavy_modules()
    mcp.run()
