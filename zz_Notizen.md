# PrivateGPT Anpassungen und Möglichkeiten

---

## 1. DB: Storage Context

### Alle drei Store-Arten

```python
self.storage_context = StorageContext.from_defaults(
    vector_store=vector_store_component.vector_store_text,
    docstore=node_store_component.doc_store,
    index_store=node_store_component.index_store,
)
```

In `node_store` werden Index und DocStore definiert

Wir verwenden hier noch 'simple' → JSON, local

#### Doc Store

- Speichert die Chunks aller Dokumente und nicht nur Vectoren
- Speichert auch welcher Chunk zu welchem gehört

#### Index Store

- Speichert die Struktur und Organisation deiner Indizes
- NICHT die Inhalte

### Kein Store (DB) die die originalen Dateien speichert

#### File Store

- Bei Funktionen `ingest` und `delete` in dem `ingest_component.py` Anbindung zu originaler Datenbank hinzufügen
- Bei allen anderen egal, weil für das RAG nur der Doc Store, Vector Store und Index Store wichtig sind
- File Store nur zur Verwaltung

## 2. Alternativen für Laszlo

- tatsächlich gibt es keine Alternative die in eine komplett andere rRchtung geht
- was man dazu nehmen kann sind Tools, das sind Funktionen die der Agent selber ausführen kann.
- Wo sind hier die Stärken? Bei Datenbanken mit exakten Daten, Tabellen und auch bei berechenbaren Werten
- und wenn er aktionen ausführen soll, versende eine email
- glaube das brauchen wir nicht weil wir ja wahrscheinlihc nur text, code, zeichnungen haben

## 3. Code ingestion:

### RAG Ingestion Pipeline – Code-Dateien

```
┌─────────────────────────────────┐
│          Code-Datei             │
│     Input: .py, .st, .scl       │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│           Chunking              │
│   Aufteilen in Funktionsblöcke  │
│        (FunctionSplitter)       │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│         Kleine LLM              │
│    Beschreibung pro Chunk       │
│    -> Code zu Sprache           │
└───────────────┬─────────────────┘
                │ Code Sprache
                ▼
┌─────────────────────────────────┐       ┌ ─ ─ ─ ─ ─ ─ ─ ─ ┐
│         Vektorsuche             │◄──────   Vector DB
│  Kontext aus bestehenden Docs   │       └ ─ ─ ─ ─ ─ ─ ─ ─ ┘
└───────────────┬─────────────────┘               ▲
                │                                 │
                ▼                                 │
┌─────────────────────────────────┐               │
│          Große LLM              │               │
│   Angereicherter finaler Text   │               │
│   ganze Datei als kontext noch  │               │
└───────────────┬─────────────────┘               │
                │                                 │
                ▼                                 │
┌─────────────────────────────────┐               │
│     Embedding + Speichern       │───────────────┘
│       In Vector DB ablegen      │  Speichert
└─────────────────────────────────┘
```

## 4. CAD ingestion

### Genrelles Tool

- technischen zeichungen -> autodesk export funktion?? \\
  -> geht sowohl als pdf, 3d pdf und csv. tabelle der paramter

Quelle: PDF: https://www.autodesk.com/learn/ondemand/tutorial/export-drawing-to-pdf-file?us_oa=dotcom-us&us_si=221c5be1-fdc8-4402-a9c1-c62262ce37a0&us_st=export%20to%20pdf

3d-PDF: https://forums.autodesk.com/t5/inventor-forum/can-t-export-to-3d-pdf/td-p/11017641?us_oa=dotcom-us&us_si=7dbd5cc1-62f6-453c-b6f9-0ccfd0900b6d&us_st=export%20to%203d%20pdf

To csv: https://www.autodesk.com/support/technical/article/caas/sfdcarticles/sfdcarticles/How-to-export-all-parameters-in-Revit-families-in-a-Content-Catalog-collection-to-Excel-CSV-or-Power-BI.html?us_oa=forums-us&us_si=5dae9988-b396-4739-8e7f-3bdcd6b3b779&us_st=export%20parameter%20to%20csv

### Was ist AI Autodesk?

- Ein **AI-Agent**, der direkt in die Autodesk-UI integriert ist
- Spezialisiert auf die Übersetzung von **Sprache → 3D-Modell** (nicht umgekehrt)
- Ob eine Nutzung in die andere Richtung (Modell → Sprache) möglich ist, wurde nicht dokumentiert

### API-Verfügbarkeit

- Es gibt **keine klassische API**
- Eine Integration müsste eigenständig bei AsTech entwickelt werden

## 5. TODOs:

- wie viel kosten großen llm im agent modus: GPT-5 Mini~$0.25~$2.00
- wie funktioniert eine api, geht das easy mit llamaindex.
- Hybrid system implemntieren:

- Gedanken machen was man mit metadaten alles machen kann
  - wenn user in prompt synnomu benutzt(ventik 4), ergänzen mit allen anderen begrifflickeitem
  - wenn im ingestion docuemtn doppeldeutgkeiten gefunden werden auch ersetzetn

- Wie sehen Fragen wirklich aus? muss da was in den system propmt zum beispiel.

## 6. Befehle

- source /Users/tilldetermann/Arbeit/Code/private-gpt/.venv/bin/activate
- python -m private_gpt
