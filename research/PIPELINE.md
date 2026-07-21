# Pipeline di ricerca (modello: The AI Scientist, Lu et al. — adattato da Dario + Claude, 17/7/2026)
# [RICOSTRUITO 20/7: il file originale è sparito da disco prima di ogni commit]

Il valore del processo non è l'intelligenza, è la disciplina: ogni fase produce un artefatto
scritto, ogni esperimento ha un budget, le decisioni le prende il processo (non l'entusiasmo,
non l'ansia, non l'assistente che asseconda).

## Artefatti

- **`BACKLOG.md`** — archivio delle idee. Una riga per idea: punteggi Interesse/Fattibilità/
  Novità (1–10), costo stimato in run, stato. Un'idea scritta qui è un'idea al sicuro:
  non serve eseguirla per non perderla.
- **`ideas/<data>_<nome>.md`** — una card per l'idea ATTIVA (una sola alla volta), con 4 campi
  compilati PRIMA di lanciare:
  1. **Ipotesi** — una frase falsificabile.
  2. **Criterio di successo pre-registrato** — il numero che decide, scritto prima dei risultati.
  3. **Budget** — numero massimo di training (default 6–16).
  4. **Journal** — ogni risultato + una riga di lettura, in ordine cronologico.

## Le 5 regole

1. **Una sola idea attiva.** Le altre stanno nel backlog.
2. **Nuova idea a metà corsa → backlog, non GPU.** Per scavalcare l'idea attiva serve la parola
   esplicita "swap", e prima si chiude la card attiva con un verdetto.
3. **Criterio prima, verdetto dopo.** A budget esaurito la card si chiude: *promossa* (paper/
   mail), *negativa* (archiviata COL numero — un risultato negativo è un risultato), o
   *ri-budgettata* (decisione esplicita, mai inerzia).
4. **Review avversariale** (Claude) su ogni claim prima che esca verso coautori/paper — la
   regola del gruppo "affermare solo dopo aver dimostrato (~99.99%)" resa strutturale.
   Dario può (deve) chiedere: "fammi da reviewer, non da assistente".
5. **Il journal è l'unica fonte del write-up.** Nel tex entrano solo numeri presenti in un
   file di `results/paper/`.

## Ciclo

BACKLOG → (selezione) → CARD con ipotesi/criterio/budget → esecuzione crash-safe con journal
→ verdetto → BACKLOG aggiornato (+ eventuale promozione a paper) → prossima idea.