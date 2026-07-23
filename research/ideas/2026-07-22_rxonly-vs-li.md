# Idea attiva: chiudere il gap RX-only vs la NN di Li (~2.6x)

Aperta: 2026-07-22. Origine: Marco chiede di confrontare la nostra RX-only NN con quella di
Asfand/Li; Stella manda la figura: le loro curve blu (NN, RX-side) battono l'adaptive equalizer,
le nostre ci coincidono.

## Numeri del confronto (N_ADC=6)
| OSNR | Li (blu) | noi Vpeak0.6 | noi Vpeak1.0 | Ad.Eq. |
|------|----------|--------------|--------------|--------|
| 13   | 2.3e-2   | 3.5e-2       | 3.2e-2       | ~3.5e-2 |
| 15   | 5e-3     | 1.31e-2      | 1.09e-2      | ~1.2e-2 |
| 17   | 1.2e-3   | 3.68e-3      | 2.54e-3      | ~4e-3   |
| 19   | 2e-4     | 7.45e-4      | 3.63e-4      | ~7e-4   |

## Ipotesi TESTATE ed ESCLUSE
- **Finestra RX** 5 / 11 / 21 simboli: identiche (1.31-1.38e-2 @15). Non è il contesto.
- **Decoder differenziale esplicito** (bpam_diff_target="coded"): 8.6e-2 @15, **6x PEGGIO**.
  La rete che decodifica internamente è meglio; il decoder esterno raddoppia gli errori di stato.
- **4 campioni/simbolo al RX** (invece di 2): 1.36e-2 @15, identico. Più campioni non aiutano.
- **Punto di lavoro MZM (Vpeak)**: MIGLIORA ma non basta. 0.6→1.0 dà ~1.4x (7.45e-4→3.63e-4 @19).
  Trade-off misurato @17: fase 4.9e-3→2.0e-3, ampiezza 2.4e-3→3.1e-3.

- **DUE RETI INDIPENDENTI al RX** (ipotesi di Dario, coerente con "2 NN receivers" di Luca):
  una rete per a_k e una per dphi_k, tronchi separati, 7912 param vs 3990. Implementato
  (knob rx_dual_network, receiver.py). Risultato: 1.36e-2 @15 vs 1.31e-2 del tronco condiviso —
  **IDENTICO**. Per-bit @17: ampiezza 2.4e-3 (uguale), fase 5.8e-3 (leggermente peggio).
  → il tronco condiviso NON era un collo di bottiglia: le due reti hanno tutta la capacità che
    vogliono e non migliorano. La limitazione non è nella struttura né nella capacità del RX.

## SVOLTA (22/7 sera): la diagnosi di Marco spiega il fallimento delle 2 reti

Dario solleva l'obiezione decisiva: *"se il loro segnale ricevuto e' migliore del nostro, allora
perche' i risultati BPAM+adaptive eq. sono pressoche' identici?"* — ed e' corretta. Le due Ad.Eq.
coincidono, quindi il segnale ricevuto e' LO STESSO. Confermato per via indipendente: i nostri
campioni dentro l'equalizzatore LLS di Marco danno 1.46e-2 @15 contro 1.5e-2 del riferimento
(check_py_chain.m). **La catena e' validata: il gap e' TUTTO nel ricevitore NN.**

Marco (mail 22/7): le 2 reti di Asfand lavorano su context window di 11 campioni **sfalsate di 1
campione**, ciascuna centrata sulla grandezza da stimare — ampiezza in mezzo all'impulso, cambio di
segno all'intersezione dei due impulsi.

→ Questo spiega perche' il mio test "2 reti indipendenti" era risultato IDENTICO al tronco singolo:
  le due reti vedevano la **stessa finestra**. Capacita' doppia ma stesso ingresso = niente su cui
  specializzarsi. Il contenuto informativo, non la capacita', era il vincolo.

Nota geometrica emersa implementando: 11 campioni e' **dispari** → la finestra ha un centro esatto,
quindi puo' essere davvero centrata sul campione di picco (offset 0) o sul crossing T/2 (offset 1).
La nostra finestra a 22 campioni (11 simboli x 2 sps) e' pari e non ha centro.

## Varianti in test (drive_stagger.py, N_ADC=6, Vpeak 0.6)
- **wide12**: 1 tronco, finestra 12 campioni (6 simboli), neuroni RADDOPPIATI [64,128,32].
  La prova semplice suggerita da Stella e Marco (una finestra che contiene entrambe le sfalsate).
- **stag11**: 2 reti, finestre 11 campioni, offset 1 — l'architettura di Asfand come descritta.
- **stag22**: 2 reti, le nostre finestre 22 campioni, offset 1 — isola lo sfalsamento dalla
  lunghezza della finestra (controllo: se stag11 vince e stag22 no, conta la finestra corta).

### ESITO (22/7 notte): lo sfalsamento NON chiude il gap — architettura ESCLUSA
| OSNR | tronco singolo | wide12 | stag11 | stag22 | Li |
|------|----------------|--------|--------|--------|-----|
| 15   | 1.31e-2        | 1.25e-2| 1.43e-2| 1.36e-2| 5e-3 |
| 17   | 3.68e-3        | 3.56e-3| 4.54e-3| 4.22e-3| 1.2e-3 |
| 19   | 7.45e-4        | 7.11e-4| 1.09e-3| 1.02e-3| 2e-4 |

Le due reti sfalsate vanno ~1.1x PEGGIO del tronco singolo (l'algebra delle finestre e' verificata,
non e' un bug: dimezzare i dati per tronco costa piu' di quanto renda la specializzazione).
wide12 (finestra 12 + neuroni raddoppiati) guadagna appena il 5%.
→ **Dopo window-length, numero di reti, sfalsamento, capacita', sps e decoder esplicito,
   l'ARCHITETTURA del ricevitore e' esclusa come causa del gap.**

Implementazione: `rx_phase_window_offset` in config, `windows_from(offset=)` in receiver.py.
Verificata l'algebra su una rampa (le due finestre differiscono di esattamente 1 campione a 2 sps)
e il gradiente su entrambi i tronchi — il test precedente girava ma era silenziosamente equivalente
al tronco singolo, quindi ora controllo l'ALGEBRA, non solo che il codice non crashi.

## Stato per-bit (Vpeak 1.0, @OSNR17)
ampiezza 3.1e-3, fase 2.0e-3 — **entrambi i bit al limite, nessuno "rotto"**. Prima del fix del
drive la fase era 1.9e-2 e l'ampiezza 3e-5 (segno irrecuperabile); ora sono bilanciati. Il collo
di bottiglia non è più un singolo bit ma la qualità complessiva del segnale ricevuto.

## Lettura
Restiamo ~2x sopra Li dopo aver escluso 4 leve strutturali. Il fatto che finestra, campioni,
decoder e (in parte) punto di lavoro non chiudano il gap suggerisce una differenza di SETUP a
monte, non di architettura del ricevitore. Serve la risposta di Li ai parametri (Stella gliel'ha
chiesta): finestra e unità, sps al suo input, drive del MZM, filtri, definizione OSNR.
Documento pronto col NOSTRO setup nel formato da chiedergli:
C:\Users\celli\Documents\MATLAB\setup_rxonly_dario.md

## NUOVA IPOTESI (22/7 notte): non e' l'architettura, e' l'ALLENAMENTO

Il gap CRESCE monotonicamente con l'OSNR:
  13 -> 1.46x | 15 -> 2.50x | 17 -> 2.97x | 19 -> 3.56x

Questa e' la firma dell'allenamento su RANGE (noi: Eb/N0 7-18) confrontato con uno PER PUNTO
OPERATIVO: a basso OSNR il range copre la condizione di test, ad alto OSNR la rete e'
sotto-ottimizzata perche' sta mediando su condizioni molto piu' rumorose. **Tutte** le nostre curve
finora usano range training.

NB: "fisso vs range" era stato provato il 21/7 e scartato — ma PRIMA del fix del drive, quando il
bit di fase era rotto (1.9e-2) e il collo di bottiglia era altrove: il confronto era confuso.
Va rifatto sulla catena corretta, dove i due bit sono bilanciati e al limite.

Criterio pre-registrato: se l'ipotesi e' giusta, il miglioramento deve CRESCERE con l'OSNR
(massimo a 19). Se il guadagno e' piatto o assente, l'ipotesi cade e il sospetto si sposta sui
parametri di setup (ER 30 dB come dice Stella, punto di lavoro MZM, banda del filtro RX).

### (a) TRAINING A PUNTO FISSO: ESCLUSA (22/7 notte)
@OSNR17: fisso **4.157e-3** vs range 3.68e-3 → il punto fisso e' PEGGIORE (0.89x), non migliore.
Per-bit: ampiezza 2.63e-3, fase 5.60e-3 (range: 2.4e-3 / 4.9e-3) — peggiora su entrambi.
Il criterio pre-registrato e' fallito al primo punto, quindi 15 e 19 NON sono stati eseguiti e la
GPU e' andata sulla diagnostica piu' informativa (sotto). Conferma il verdetto del 21/7, ma ora su
catena corretta e quindi non piu' confuso dal bug del segno. Dati: rxonly_fixedtrain.txt.
Lettura: allenare su un range e' una REGOLARIZZAZIONE che aiuta, non un handicap.

### (b) LR COSTANTE + la domanda che avrei dovuto farmi prima
Osservazione nei log: il training e' in PLATEAU gia' a 12.5k step su 100k (4.76e-3 → 4.82e-3 a 25k,
4.46e-3 a 37.5k, 4.03e-3 a 50k) = **fermo, non lento**. Nessuno scheduler nel loop principale
(surrogate.py ne aveva uno, train() no): Adam a 1e-3 costante per 100k step.

**Il fatto anomalo che avrei dovuto notare prima**: la nostra NN non-lineare (migliaia di parametri)
fa ESATTAMENTE quanto un equalizzatore LINEARE a 11+11 tap. Su un canale a legge quadratica con CD
un ricevitore non-lineare dovrebbe vincere — e infatti Li batte quello stesso equalizzatore di 3x.
→ DIAGNOSTICA DECISIVA: stessa rete con ffe_nonlinear=False (puramente lineare). Se pareggia la
  non-lineare, la non-linearita' NON viene sfruttata = problema di OTTIMIZZAZIONE, non di capacita'.
Test in corso: drive_lr.py (linear @17, cosine @17/19, fixed+cosine @17/19). File: rxonly_lrsched.txt.
Knob aggiunto: config.lr_schedule="cosine" (default None → tutti i risultati precedenti invariati).

## SVOLTA 2 (23/7): la validazione nell'altra direzione + la traslazione di 1.3 dB

Dario: *"Hai provato a implementare qua su python lo stesso equalizer?"* → il port Python era fallito
3 volte, quindi avevo fatto il test OPPOSTO (nostri campioni → suo equalizzatore MATLAB). Ma quel
test **prova meno di quanto sembri**: l'equalizzatore LLS riottimizza tap E soglie, quindi e' robusto
a differenze di catena che una rete a ingresso fisso sentirebbe. Mancava la direzione inversa.

FATTA: hook `p.export_file` in simulatore_PAMb_DD_adaptive.m (default spento) che salva i campioni a
2 sps post-ADC = esattamente cio' che vede il suo equalizzatore. Poi la NOSTRA rete su QUEI campioni
(banco veloce: finestre precalcolate, niente simulazione di canale nel loop).
Allineamento preso dal suo codice: z(0::2)↔a_tx (ampiezza), z(1::2)↔b_tx (segno) — cioe' la
descrizione delle finestre sfalsate di Marco E' letteralmente la struttura del suo simulatore.

| | OSNR 15 | OSNR 17 |
|---|---------|---------|
| suo eq. lineare (suoi campioni) | 1.538e-2 | 4.125e-3 |
| **nostra NN sui SUOI campioni**  | **1.254e-2** | **2.93e-3** |
| nostra NN sulla nostra catena    | 1.31e-2  | 3.68e-3 |
| Li                               | 5e-3     | 1.2e-3 |

1. **CATENE EQUIVALENTI, validate nelle DUE direzioni**: 1.25e-2 sui suoi vs 1.31e-2 sui nostri.
   Nessuna differenza nascosta che solo la NN vedeva. Il nostro TX BPAM e' a posto.
2. **La non-linearita' SERVE ma poco**: sui campioni identici battiamo il suo eq. lineare del 18%
   (@15) e del 28% (@17). Diagnostica indipendente sulla nostra catena: FFE puramente lineare
   6.19e-3 vs 3.68e-3 della non-lineare @17 = **1.68x**. Quindi la non-linearita' E' sfruttata:
   NON era un fallimento di ottimizzazione. L'ipotesi di Dario ("il lineare e' imbattibile") va
   RAFFINATA: non e' imbattibile, ma vale 1.4-1.7x, non 3x.
3. **LR SCHEDULE: ESCLUSO.** Sui campioni del prof cosine 1.266e-2 vs costante 1.254e-2 (identici);
   sulla nostra catena cosine 4.20e-3 vs 3.68e-3 (PEGGIO). L'ipotesi principale della notte e' morta.

### LA TRASLAZIONE — probabile spiegazione del gap
A quale OSNR la NOSTRA curva raggiunge le BER di Li?
  2.3e-2: lui 13 → noi 13.93 (+0.93 dB) | 5e-3: lui 15 → noi 16.62 (+1.62)
  1.2e-3: lui 17 → noi 18.59 (+1.59)    | 2e-4: lui 19 → noi 20.00 (+1.00)
  → shift medio **+1.29 dB** (dev.std 0.32)
Non e' un ricevitore 3x migliore: e' la NOSTRA curva TRASLATA di ~1.3 dB. Un rivelatore migliore
cambia pendenza o da' un guadagno variabile; una traslazione orizzontale quasi costante su due
decadi e' la firma di una differenza di SETUP o di CONVENZIONE.
CAVEAT: i numeri di Li sono digitalizzati da me dalla figura → lo spread 0.93-1.62 puo' essere in
buona parte errore di lettura. L'ordine di grandezza pero' tiene.

1.3 dB e' esattamente il budget dei due parametri ancora aperti: **ER 30 dB invece di 25** (segnalato
da Stella) e **Vpeak 1.0 invece di 0.6** (misurato ~1.4x ≈ 0.5 dB). In test sul simulatore DEL PROF
(cosi' il confronto e' massimamente credibile): export_prof_samples(osnr, er_db, vpeak), griglia
ER{25,30} x Vpeak{0.6,1.0}. File: nn_on_prof_samples.txt.

## CAUSA TROVATA (23/7): IL GAUSSIANO TX DISTRUGGE IL BIT DI FASE

La nostra catena ha DUE gaussiani in cascata (10 GHz al TX + 10 GHz al RX = banda equivalente
~7.1 GHz); la baseline di Stella ha SOLO quello al RX (10 GHz), Li nessuno. Su 20 GBaud e' una
penalita' di banda che produce esattamente la traslazione orizzontale rigida misurata.

Campioni dal simulatore DEL PROF con TX NRZ (senza gaussiano), ER30, Vpeak 0.6, nostra NN:
| OSNR 17 (ER30)          | ampiezza | fase     | totale   | vs Li  | vs suo eq. lineare |
|-------------------------|----------|----------|----------|--------|--------------------|
| CON gaussiano TX        | 1.65e-3  | 2.62e-3  | 2.135e-3 | 1.78x  | 1.35x meglio       |
| **SENZA gaussiano TX**  | 1.89e-3  | 3.55e-4  | 1.121e-3 | 0.93x  | **2.89x meglio**   |
OSNR 15: 6.87e-3 vs Li 5e-3 (1.37x, era 2.5x).

MECCANISMO, e spiega perche' era invisibile dal lato ricevitore: il gaussiano TX **non toglie quasi
nulla all'ampiezza** (1.65 -> 1.89e-3) ma **distrugge la fase** (2.62e-3 -> 3.55e-4, 7.4x). Il segno
vive nel termine incrociato a T/2, cioe' nell'intersezione fra impulsi adiacenti: e' la componente
piu' veloce del segnale, la prima che un filtro a meta' della symbol rate cancella. Nessun intervento
sul RICEVITORE poteva recuperarla — l'informazione era gia' stata buttata via a monte.
E torna il rapporto: senza il filtro la NN batte il lineare di 2.89x = il ~3x che Li riporta.

LEZIONE DI PROCESSO (da non ripetere): questo sospetto era **gia' annotato nell'handoff il 20/7**
("la baseline di Stella ha SOLO il gaussiano 10 GHz al RX; noi ne abbiamo DUE... parte del gap 1 dB
vs 3 puo' venire da li'"), con perfino una probe pronta (drive_fig8_notxgauss.py). Poi due giorni su
architettura del RX, training e learning rate senza tornarci. Cio' che ha rimesso in carreggiata e'
stata la domanda di Dario sull'equalizzatore, che ha spostato il lavoro da OTTIMIZZARE LA RETE a
VALIDARE LA CATENA. Regola: quando una discrepanza resiste a 3+ ipotesi sul componente sospettato,
rileggere le note di setup PRIMA di provare la quarta.

IMPLICAZIONE PER IL PAPER (piu' grande del confronto con Li): la nostra baseline BPAM+AE E la nostra
curva RX-only sono state generate ENTRAMBE con un filtro che la baseline del gruppo non ha -> il
confronto con le loro curve non era alla pari. Da segnalare a Stella/Marco PRIMA che le figure vadano
nel paper. DECISIONE DI DARIO (non mia): il gaussiano TX era stato messo per fedelta' alla catena di
Marco (tx_gaussian_bw, commit 70e6df6); toglierlo allinea a Stella/Li ma ALLONTANA da Marco. Serve
sapere da loro qual e' la configurazione di riferimento per il paper.

## Prossimi passi possibili (non ancora fatti)
- Training più lungo / multi-seed sulla config Vpeak 1.0 (potrebbe valere un altro ~1.2x)
- Chiedere a Li i parametri e riprodurre il suo setup esatto
- NB: Vpeak 1.0 NON è più "fedele al prof" (che usa 0.6) -> se lo teniamo va dichiarato, oppure
  va chiesto a Li/Stella quale punto di lavoro usano loro.
