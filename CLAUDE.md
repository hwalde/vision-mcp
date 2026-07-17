# Vision MCP — Setup & Konfiguration

MCP-Server (stdio, Python/FastMCP) für Bild-Analyse: **Objekterkennung** (YOLOv8) und
**Texterkennung/OCR** (EasyOCR), plus abgeleitete Layout-Aussagen (Abstände, Zentrierung).
Läuft auf **Windows, macOS und Linux**, komplett **lokal und offline** — es geht kein Bild an
eine Cloud-API.

Die Ausgabe ist bewusst **LLM-optimierter Markdown-Text, kein JSON**: natürliche Sätze wie
„Text „Headline" liegt 276px oberhalb von person." verarbeitet ein LLM zuverlässiger als
verschachtelte Koordinaten-Strukturen.

---

## Was der Server braucht

| Baustein | Zweck |
|---|---|
| Python ≥ 3.10 | Laufzeit |
| Pakete aus `requirements.txt` | u. a. `ultralytics` (YOLO), `easyocr`, `torch` |
| YOLOv8-Gewichte (`yolov8n.pt`, ~6 MB) | werden beim ersten Start **automatisch geladen** |
| EasyOCR-Sprachmodelle (~100 MB) | werden beim ersten OCR-Aufruf **automatisch geladen** |

**Kein API-Key, kein Cloud-Konto, keine Auth.** Nur der erste Start braucht Internet, um die
Modelle zu holen.

---

## Installation

```bash
pip install -r requirements.txt
```

Achtung: `ultralytics` und `easyocr` ziehen **PyTorch** mit — das sind mehrere hundert MB und die
Installation dauert entsprechend.

---

## Konfiguration über Umgebungsvariablen

Alles optional — ohne jede Variable läuft der Server mit sinnvollen Defaults.

| Variable | Bedeutung | Default |
|---|---|---|
| `VISION_MCP_YOLO_MODEL` | YOLO-Gewichte: Dateiname (wird in den Cache geladen) oder absoluter Pfad. Größere Modelle = genauer, langsamer: `yolov8n.pt` < `yolov8s.pt` < `yolov8m.pt` < `yolov8l.pt` < `yolov8x.pt`. | `yolov8n.pt` |
| `VISION_MCP_OCR_LANGS` | OCR-Sprachen, kommagetrennt (EasyOCR-Codes). | `de,en` |
| `VISION_MCP_MODEL_DIR` | Cache-Verzeichnis der YOLO-Gewichte. | siehe unten |

### Modell-Cache

Ohne `VISION_MCP_MODEL_DIR` liegen die Gewichte plattformüblich:

- **Windows:** `%LOCALAPPDATA%\vision-mcp\`
- **macOS/Linux:** `~/.cache/vision-mcp/`

Der Server löst den Gewichts-Namen bewusst zu einem **absoluten** Pfad auf. Ohne das lädt
ultralytics die Datei ins jeweilige **Arbeitsverzeichnis** — der Grund, warum früher in mehreren
Projektordnern je eine eigene `yolov8n.pt` lag.

EasyOCR verwaltet seine Sprachmodelle selbst unter `~/.EasyOCR/`.

---

## Eintrag in die MCP-Konfiguration

In `~/.claude.json` (bzw. per `claude mcp add`):

```json
{
  "mcpServers": {
    "vision_mcp": {
      "type": "stdio",
      "command": "python3",
      "args": ["/absoluter/pfad/zu/vision_mcp/server.py"],
      "env": {}
    }
  }
}
```

Unter **Windows** `"command": "python"` und einen Windows-Pfad verwenden.

---

## Tools

| Tool | Zweck |
|---|---|
| `vision_detect_objects` | Welche Objekte sind im Bild, wo, wie groß, zentriert? (`image`, `confidence`, `classes`) |
| `vision_detect_text` | OCR: welcher Text steht wo? (`image`) |
| `vision_measure_distance` | Pixelabstand/Beziehung zwischen zwei benannten Elementen (`image`, `element_a`, `element_b`, `confidence`) |
| `vision_check_alignment` | Ist ein bestimmtes Element zentriert? (`image`, `element`, `confidence`) |
| `vision_analyze_layout` | Gesamtüberblick in einem Aufruf: Objekte + Texte + bis zu 20 Beziehungssätze |

`image` akzeptiert einen lokalen Pfad (`~` wird aufgelöst) **oder** eine `http(s)`-URL.

Die **Element-Suche** in `measure_distance`/`check_alignment` probiert der Reihe nach: exakte
Objektklasse → exakter Text → Teilstring in Text → Teilstring in Klasse. Bei einem partiellen
Treffer weist der Bericht ausdrücklich darauf hin. Wird nichts gefunden, listet die Fehlermeldung
alle erkannten Objekte und Texte als Vorschläge auf.

---

## Grenzen (wichtig fürs Erwartungsmanagement)

- **YOLOv8 erkennt nur die ~80 COCO-Klassen** (person, car, dog, laptop, bottle …). Es erkennt
  **keine UI-Elemente** — keine Buttons, Logos, Icons, keine Design-Shapes. Wer ein Creative auf
  „ist der Button mittig?" prüfen will, kommt über den **Text** des Buttons (OCR) ans Ziel, nicht
  über Objekterkennung.
- **OCR versagt** bei stark stilisierten Schriften, Handschrift und sehr kleinem oder
  kontrastarmem Text. Die Wortgruppierung entspricht nicht immer der menschlichen Erwartung.
- **Läuft auf der CPU** (`gpu=False` bei EasyOCR). EasyOCR warnt beim Start entsprechend — das ist
  erwartet, kein Fehler.
- **Der erste Aufruf ist langsam**: Modelle werden lazy geladen und ggf. erst heruntergeladen.

## Konventionen der Ausgabe

- Koordinatensystem: (0,0) oben links, +X rechts, +Y unten. `bbox.x/y` = obere linke Ecke.
- Konfidenz: `high` ≥ 0.75, `medium` ≥ 0.50, `low` < 0.50.
- Zentrierung: „zentriert" ≤ 5 % Abweichung, „fast zentriert" ≤ 10 %, „off-center" > 10 %.

Jeder Bericht trägt diese Legende als Fußnote (`META_FOOTER`), damit ein LLM die Zahlen ohne
Zusatzkontext richtig deutet.
