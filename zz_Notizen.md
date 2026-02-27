# PrivateGPT Anpassungen und Möglichkeiten

## 1. CAD & Electronic Schematics Support

### CAD und elektronische Schaltpläne erkennen

#### Erste Frage: Welches Format?
- Welches CAD-Format wird verwendet?
- Bietet das Tool mit dem die CADS erstellt werden eine API oder Export-Option an?
- Beispiel: KiCad

#### Eigenen Reader hinzufügen
Eigenen Reader in `ingest_helper.py` hinzufügen, der das Format von den anderen LlamaIndex Readern hat.

#### Problem
Keine AI direkt für CAD, vor allem nicht lokal. Die großen Modelle können das.


---

## 2. Code Parser

### Aktueller Stand
```python
node_parser = SentenceWindowNodeParser.from_defaults()  # für alles (das ist der chunker)
```

### Verbesserung
Das dynamisch, also abhängig vom Datei-Typ machen.

#### Lesen
```python
SimpleDirectoryReader(
    input_files=["src/auth.py"]
)
```

#### Chunken
```python
CodeSplitter(
    language="python",
    chunk_lines=50
)
```

---

## 3. DB: Storage Context

### Alle drei Store-Arten
```python
self.storage_context = StorageContext.from_defaults(
    vector_store=vector_store_component.vector_store,
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

#### Eigene Stores machen
- Einfach mal MongoDocumentStore angucken
- https://developers.llamaindex.ai/python/framework/module_guides/storing/


### Kein Store (DB) die die originalen Dateien speichert

#### File Store
- Bei Funktionen `ingest` und `delete` in dem `ingest_component.py` Anbindung zu originaler Datenbank hinzufügen
- Bei allen anderen egal, weil für das RAG nur der Doc Store, Vector Store und Index Store wichtig sind
- File Store nur zur Verwaltung 

## 4. Alternativen für Laszlo
- tatsächlich gibt es keine Alternative die in eine komplett andere rRchtung geht
- was man dazu nehmen kann sind Tools, das sind Funktionen die der Agent selber ausführen kann.
- Wo sind hier die Stärken? Bei Datenbanken mit exakten Daten, Tabellen und auch bei berechenbaren Werten
- und wenn er aktionen ausführen soll, versende eine email
- glaube das brauchen wir nicht weil wir ja wahrscheinlihc nur text, code, zeichnungen haben

## 5. Beispiel Prompt:

Please explain the standard steps of a Genetic Algorithm (GA) in detail.

Focus particularly on the Mutation step by explaining:
- What mutation does and why it's important
- How mutation rate affects algorithm performance

----

Analyze the strengths and weaknesses of Genetic Algorithms (GAs) regarding the number of model evaluation metric. 

Please structure your answer as follows:

- Explain the metric

**Strengths:**
- How GAs optimize the number of evaluations compared to exhaustive search

**Weaknesses:**
- Total number of fitness evaluations typically required for convergence

**Context:** I'm interested in understanding whether GAs are suitable for energy-constrained optimization problems where each model evaluation is computationally expensive.

## 6. Notes:
-  technischen zeichungen -> inventor, autocad export funktion??
    -> geht sowohl als pdf, 3d pdf und csv. tabelle der paramter

mal testen mit cad modell open source step oder pdf oder andere formate

Quelle: PDF: https://www.autodesk.com/learn/ondemand/tutorial/export-drawing-to-pdf-file?us_oa=dotcom-us&us_si=221c5be1-fdc8-4402-a9c1-c62262ce37a0&us_st=export%20to%20pdf

3d-PDF: https://forums.autodesk.com/t5/inventor-forum/can-t-export-to-3d-pdf/td-p/11017641?us_oa=dotcom-us&us_si=7dbd5cc1-62f6-453c-b6f9-0ccfd0900b6d&us_st=export%20to%203d%20pdf

To csv: https://www.autodesk.com/support/technical/article/caas/sfdcarticles/sfdcarticles/How-to-export-all-parameters-in-Revit-families-in-a-Content-Catalog-collection-to-Excel-CSV-or-Power-BI.html?us_oa=forums-us&us_si=5dae9988-b396-4739-8e7f-3bdcd6b3b779&us_st=export%20parameter%20to%20csv

CADdy: ???

- kleine githib opensource pürojekt emit cad raus finden und gucken wie die aufbereitet werden können
- COde incht als raw sondern als text beschirben, pre chunken von code in funktion???
