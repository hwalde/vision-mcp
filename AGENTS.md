# Vision MCP — Setup & Konfiguration

> Diese Datei ist die **einzige Quelle** der Projekt-Doku. `CLAUDE.md` importiert sie per
> `@AGENTS.md`, damit Claude Code sie automatisch mitlädt — nicht duplizieren, hier pflegen.

MCP-Server (Python/FastMCP, stdio **oder** streamable-http — siehe „Transporte") für Bild-Analyse: **Objekterkennung** (YOLOv8) und
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
| Pakete aus `requirements.txt` | u. a. `ultralytics` (YOLO) und `easyocr` — beide ziehen `torch` transitiv mit |
| YOLOv8-Gewichte (`yolov8n.pt`, ~6 MB) | werden beim ersten Start **automatisch geladen** |
| EasyOCR-Sprachmodelle (~100 MB) | werden beim ersten OCR-Aufruf **automatisch geladen** |

**Kein API-Key, kein Cloud-Konto, keine Auth.** Nur der erste Start braucht Internet, um die
Modelle zu holen.

---

## Installation

Nach dem Auschecken genügen zwei Befehle — identisch auf allen Betriebssystemen:

**macOS / Linux**

```bash
pip3 install -r requirements.txt
python3 server.py --selftest
```

**Windows (PowerShell)**

```powershell
pip install -r requirements.txt
python server.py --selftest
```

`--selftest` ist der schnellste Weg zur Gewissheit: Er prüft die Python-Version, meldet jedes
fehlende Paket namentlich, lädt YOLO- und OCR-Modelle vor (der einzige Schritt, der Internet
braucht) und analysiert ein selbst erzeugtes Testbild. Endet er mit „Alles bereit", ist der Server
einsatzfähig — man muss den Fehler nicht erst beim ersten Tool-Aufruf im Agenten entdecken.
Exit-Code 0 = alles in Ordnung, 1 = Problem.

**Was dabei zu erwarten ist:** `ultralytics` und `easyocr` ziehen **PyTorch** mit — mehrere hundert
MB, die Installation dauert entsprechend. Der erste `--selftest` lädt zusätzlich ~6 MB
YOLO-Gewichte und ~100 MB OCR-Modelle. Danach läuft alles offline.

**Python ≥ 3.10** ist Pflicht (das `mcp`-Paket setzt es voraus); der Selbsttest bricht bei
älteren Versionen mit klarer Meldung ab.

### Plattform-Hinweise

- **Windows:** In den MCP-Eintrag `"command": "python"` schreiben (nicht `python3`, das existiert
  dort in der Regel nicht). PyPI liefert für Windows ein CPU-PyTorch — kein CUDA-Setup nötig.
- **Linux:** `pip install torch` zieht per Default die CUDA-Variante (mehrere GB). Wer das nicht
  will, installiert vorab das schlanke CPU-Wheel:
  `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- **macOS:** Läuft auf Intel und Apple Silicon. EasyOCR wird bewusst mit `gpu=False` betrieben;
  die Warnung „Using CPU" beim Start ist erwartet.

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

## Transporte: stdio oder streamable-http

Der Server kann auf zwei Wegen betrieben werden. Die Wahl trifft `--transport` (schlägt alles)
oder die Umgebungsvariable `VISION_MCP_TRANSPORT`; ohne Angabe bleibt es bei `stdio`.

| Transport | Betrieb | Wann |
|---|---|---|
| `stdio` (Default) | Der Agent startet **je Sitzung einen eigenen** Serverprozess. | Einzelplatz, Windows, Debugging |
| `streamable-http` | **Ein geteilter** Prozess bedient alle Sitzungen unter `http://127.0.0.1:8902/mcp`. | Viele parallele Claude-Code-Sitzungen auf einem Rechner |

```bash
# geteilter Dienst
python3 server.py --transport streamable-http --host 127.0.0.1 --port 8902
```

| Variable / Argument | Bedeutung | Default |
|---|---|---|
| `--transport` / `VISION_MCP_TRANSPORT` | `stdio` oder `streamable-http` | `stdio` |
| `--host` / `VISION_MCP_HOST` | Bind-Adresse. **Nur `127.0.0.1`** — der Server hat keine Auth, `0.0.0.0` würde ihn im LAN öffnen. | `127.0.0.1` |
| `--port` / `VISION_MCP_PORT` | Port im HTTP-Betrieb | `8902` |
| `VISION_MCP_PRELOAD` | `1` = YOLO- und OCR-Modelle beim Start laden (erster Aufruf schnell, Ruhezustand ~1 GB), `0` = verzögert beim ersten Aufruf (Ruhezustand ~84 MB) | `1` |

### Pfad-Regel im geteilten Betrieb

Bei `stdio` erbt der Server das **Arbeitsverzeichnis der Sitzung**, relative Bildpfade
funktionieren also. Der geteilte HTTP-Prozess hat ein **einziges festes** Arbeitsverzeichnis und
kennt das der aufrufenden Sitzung nicht. Deshalb **lehnt er relative Pfade ausdrücklich ab** —
mit einer Meldung, die auf absolute Pfade hinweist. Ohne diese Sperre würde ein relativer Pfad
still gegen das falsche Verzeichnis aufgelöst. Absolute Pfade, `~`-Pfade und `http(s)`-URLs
funktionieren in beiden Betriebsarten unverändert.

### Thread-Sicherheit

Im geteilten Betrieb können mehrere Sitzungen gleichzeitig Aufrufe schicken. Das verzögerte Laden
**und** die Inferenz von YOLO und EasyOCR sind darum je durch eine Sperre serialisiert
(`_yolo_lock`, `_ocr_lock`) — beide Bibliotheken sind nicht als thread-sicher dokumentiert.
Fachlichen Zustand teilen die Sitzungen nicht: jeder Tool-Aufruf lädt sein Bild selbst und gibt
seinen Bericht zurück, die Modelle sind reine Lese-Caches.

### Als Dienst unter macOS

`~/Library/LaunchAgents/de.herbe.mcp.vision.plist` (`RunAtLoad`, `KeepAlive`, Logs unter
`~/Library/Logs/mcp-vision.{out,err}.log`).

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/de.herbe.mcp.vision.plist   # starten
launchctl kickstart -k gui/$(id -u)/de.herbe.mcp.vision                            # neu starten
launchctl bootout   gui/$(id -u)/de.herbe.mcp.vision                               # stoppen
```

Der MCP-Eintrag lautet dann:

```json
{
  "mcpServers": {
    "vision_mcp": {
      "type": "http",
      "url": "http://127.0.0.1:8902/mcp"
    }
  }
}
```

### Speicherverhalten

EasyOCR/PyTorch allokieren bei der Erkennung großzügig und geben nur träge zurück. Gemessen auf
einem M4 Pro mit `VISION_MCP_PRELOAD=0`: **84 MB** im Ruhezustand, **0,7–1,8 GB** nach Analysen
(1440×931 bis 3456×2234 px). In einem Testlauf **mit** Preload stieg der Prozess nach mehreren
Retina-Vollbild-Analysen kurzzeitig auf ~9 GB. Wer den Dienst über Wochen laufen lässt, sollte ihn
gelegentlich per `launchctl kickstart -k` erneuern.

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
