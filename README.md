# Vision MCP Server

**Ein MCP-Server, der einem LLM Augen für Bilder gibt — vollständig lokal, ohne API-Key, ohne Cloud.**

Objekterkennung (YOLOv8) und Texterkennung (EasyOCR), aufbereitet zu Aussagen, mit denen ein
Sprachmodell tatsächlich arbeiten kann: Wo liegt was, wie groß ist es, ist es zentriert, wie weit
ist es von etwas anderem entfernt?

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

---

## Warum Markdown statt JSON?

Die meisten Vision-Tools geben Koordinaten zurück und überlassen dem Modell das Rechnen. Dieser
Server dreht das um und liefert fertige Sätze:

```
Text „Jetzt anmelden" liegt 276px unterhalb von person.
person — Konfidenz: high (0.91)
   Position: x=310, y=120 | Größe: 280×410 px | 24.3% des Bildes
   Horizontal: zentriert (1.2% Abweichung)
   Vertikal: off-center, 88px nach oben (14.7%)
```

Ein LLM verarbeitet „liegt 276px unterhalb von" zuverlässiger als vier verschachtelte Bounding-Box-
Zahlen, aus denen es die Beziehung erst selbst ableiten müsste. Jeder Bericht trägt eine Fußnote
mit Koordinatensystem- und Konfidenz-Legende, damit die Zahlen ohne Zusatzkontext richtig gedeutet
werden.

## Tools

| Tool | Beantwortet die Frage |
|---|---|
| `vision_detect_objects` | Welche Objekte sind im Bild — wo, wie groß, zentriert? |
| `vision_detect_text` | Welcher Text steht wo im Bild? (OCR) |
| `vision_measure_distance` | Wie weit sind zwei Elemente voneinander entfernt? Überlappen sie? |
| `vision_check_alignment` | Ist dieses eine Element zentriert — und wie stark daneben? |
| `vision_analyze_layout` | Gesamtüberblick in einem Aufruf: Objekte + Texte + Beziehungen |

Als Bildquelle wird ein lokaler Pfad (`~` wird aufgelöst) oder eine `http(s)`-URL akzeptiert.
Elemente werden über ihren Namen angesprochen — eine YOLO-Klasse wie `person` oder ein Textinhalt
wie `Jetzt anmelden`, auch als Teilstring.

## Installation

```bash
pip install -r requirements.txt
python3 server.py --selftest     # Windows: python server.py --selftest
```

Der Selbsttest prüft die Abhängigkeiten, lädt die Modelle vor und analysiert ein Testbild. Sagt er
„Alles bereit", ist der Server einsatzfähig.

Dann in `~/.claude.json` eintragen:

```json
{
  "mcpServers": {
    "vision_mcp": {
      "type": "stdio",
      "command": "python3",
      "args": ["/absoluter/pfad/zu/vision_mcp/server.py"]
    }
  }
}
```

Beim ersten Start werden die Modelle automatisch geladen (YOLO ~6 MB, EasyOCR ~100 MB) — danach
läuft alles offline. **Kein API-Key, kein Cloud-Konto:** Es verlässt kein Bild den Rechner, was den
Server auch für vertrauliches Material brauchbar macht.

## Grenzen — bitte vorher lesen

- **YOLOv8 erkennt nur die ~80 COCO-Klassen** (person, car, dog, laptop, bottle …), also
  **keine UI-Elemente**: keine Buttons, Logos, Icons, keine Design-Shapes. Ein Creative auf „ist der
  Button mittig?" prüft man über den **Text** des Buttons (OCR), nicht über Objekterkennung.
- **OCR versagt** bei stark stilisierten Schriften, Handschrift und sehr kleinem oder
  kontrastarmem Text; die Wortgruppierung entspricht nicht immer der menschlichen Erwartung.
- **Läuft auf der CPU.** Der erste Aufruf ist langsam, weil die Modelle lazy geladen werden.

## Konfiguration

Alles optional — ohne jede Variable läuft der Server mit sinnvollen Defaults.

| Variable | Bedeutung | Default |
|---|---|---|
| `VISION_MCP_YOLO_MODEL` | YOLO-Gewichte (`yolov8n/s/m/l/x.pt` oder absoluter Pfad) | `yolov8n.pt` |
| `VISION_MCP_OCR_LANGS` | OCR-Sprachen, kommagetrennt | `de,en` |
| `VISION_MCP_MODEL_DIR` | Cache-Verzeichnis der Gewichte | `~/.cache/vision-mcp` bzw. `%LOCALAPPDATA%\vision-mcp` |

## Dokumentation

Vollständige Setup-, Konfigurations- und Betriebsdoku: **[AGENTS.md](./AGENTS.md)**.

## Lizenz

[MIT](./LICENSE) © Herbert Walde
