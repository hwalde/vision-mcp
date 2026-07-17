# Vision MCP Server

MCP-Server (stdio, Python) für lokale Bild-Analyse: **Objekterkennung** (YOLOv8) und
**OCR** (EasyOCR), plus Layout-Aussagen zu Abständen und Zentrierung. Läuft auf Windows, macOS
und Linux — **vollständig lokal, ohne API-Key und ohne Cloud**.

Die Berichte sind LLM-optimierter Markdown-Text statt JSON: „Text „Headline" liegt 276px oberhalb
von person." statt verschachtelter Koordinaten.

## Features

- **Objekte erkennen** — Klasse, Konfidenz, Position, Größe, Flächenanteil, Zentrierung
- **Text erkennen (OCR)** — Deutsch + Englisch, Sprachen konfigurierbar
- **Abstände messen** zwischen zwei benannten Elementen, inkl. Überlappung/Berührung
- **Ausrichtung prüfen** eines einzelnen Elements
- **Komplette Layout-Analyse** in einem Aufruf (Objekte + Texte + Beziehungen)

Bildquelle: lokaler Pfad oder `http(s)`-URL.

## Schnellstart

```bash
pip install -r requirements.txt
```

MCP-Eintrag (`~/.claude.json`):

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

Beim ersten Start werden die Modell-Gewichte automatisch geladen (YOLO ~6 MB, EasyOCR ~100 MB).

## Tools

| Tool | Zweck |
|---|---|
| `vision_detect_objects` | Objekte im Bild erkennen |
| `vision_detect_text` | Text im Bild erkennen (OCR) |
| `vision_measure_distance` | Abstand zwischen zwei Elementen messen |
| `vision_check_alignment` | Zentrierung eines Elements prüfen |
| `vision_analyze_layout` | Komplette Layout-Analyse |

## Grenzen

YOLOv8 erkennt nur die ~80 COCO-Klassen — **keine UI-Elemente** wie Buttons, Logos oder Icons.
OCR kann bei stilisierten oder sehr kleinen Schriften versagen. Läuft auf der CPU.

## Dokumentation

**Vollständige Setup-, Konfigurations- und Betriebsdoku: [CLAUDE.md](./CLAUDE.md).**

## Lizenz

Proprietär
