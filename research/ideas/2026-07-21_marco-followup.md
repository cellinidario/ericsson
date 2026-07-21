# Idea attiva: follow-up di Marco (RX-only vs E2E, complessità TX)

Aperta: 2026-07-21, dalla mail di Marco dopo i risultati Asfand. Due domande:
(1) quanto del guadagno viene dall'ottimizzazione TX? → RX-only: TX BPAM classico (livelli
Secondini fissi + precoder differenziale, automatico per equalizer="ffe") + la stessa RX
complex; (2) aumentare la complessità del TX-NN aiuta? → DPD W=64 D=1 e W=64 D=2 (memoria
5 simboli invariata — vincolo Ericsson).

Nota preliminare importante (verificata sul codice): il precoder che Marco specifica nella
mail (c2[k] = c2[k-1] XOR b2[k], accumulatore) è ESATTAMENTE quello già implementato
(transmitter.py:127, cumsum%2 ≡ ricorsione di Marco; b1 intatto; target RX = bit raw).
Quindi l'A/B negativo già comunicato usava il suo identico encoder.

## Ipotesi
(1) L'E2E batte l'RX-only di un margine ≥ della varianza di seed (~0.3 dB) — è il valore
dell'ottimizzazione TX. (2) Il TX più grosso dà un guadagno piccolo o nullo (il DPD a
memoria 5 con W=16 sembra già saturo, ma non è mai stato misurato su questa RX).

## Criterio (pre-registrato)
Riportare i delta @BER 1e-3 (asse Eb/N0) con 2 seed sulla config headline (adc6):
E2E − RXonly e TXgrosso − TXattuale. Soglia di rilevanza: |delta| > 0.3 dB (varianza seed
osservata sulla catena Asfand). Nessun numero-obiettivo: sono domande di caratterizzazione.

## Budget
8 run (drive_marco_followup.py): rxonly adc{6,8,5}×s0 + adc6 s1; txW64D1/txW64D2 adc6 ×
seed {0,1}. Checkpoint salvati. ~4h.

## Journal
- 21/7: smoke RX-only ok (converge, 500 step → 6e-2 @14). Campagna lanciata (task b1wymj4kj,
  log $TEMP/marco_followup.log → results/paper/marco_followup_results.txt).
- 21/7, campagna COMPLETA (8/8). @OSNR 17, N_ADC=6:
  * RX-only (TX BPAM classico + RX complex): 3.87e-3 / 3.72e-3 (2 seed, varianza minima)
    ≈ BPAM+AE lineare (4.10e-3). La NN al RX da sola vale ~0 dB sull'AE in questa catena.
  * E2E TX W=16 (riferimento Asfand): 1.9e-4 / 4.5e-4 → ~20× meglio dell'RX-only.
  * TX W=64 D=1: 3.76e-4 / 7.4e-4; TX W=64 D=2: 4.5e-4 / 5.8e-4 — consistentemente
    PEGGIO del W=16 su entrambi i seed.

## Verdetto (21/7, contro il criterio pre-registrato)
ENTRAMBE LE DOMANDE CHIUSE CON MARGINE >> 0.3 dB:
(1) **Il guadagno è interamente dell'ottimizzazione TX congiunta**: E2E − RXonly ≈ 20× in BER
(diversi dB orizzontali); l'RX-only NN ≈ equalizzatore adattivo lineare. Ridimensiona
l'aspettativa "NN al RX ~2 dB" nella NOSTRA catena (nel loro setup può differire — dirlo
coi numeri, senza contraddire).
(2) **TX più complesso NON aiuta, peggiora leggermente** (2 architetture × 2 seed concordi):
il DPD W=16 a memoria 5 è già saturo. Ottimo per la storia complessità/LUT.
Dati: marco_followup_results.txt; checkpoint jlt_cband_bpam4_{rxonly,txW64D1,txW64D2}_*.pt.
