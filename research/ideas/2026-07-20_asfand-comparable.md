# Idea attiva: E2E "Asfand-comparable" (RX complex 32-64-16, window 5, precoder differenziale)

Aperta: 2026-07-20. Origine: meeting Marco/Stella + mail di Luca (stesso giorno).
Contesto: il gruppo autorizza le reti "complex" (i risultati di Stella usano quella di Asfand);
Luca specifica la struttura: 3 hidden FC (32, 64, 16) + sigmoide, context window 5, con
encoding differenziale. Marco propone il precoder differenziale IN INGRESSO al DPD (uscita
in bit normali) per aggirare la memoria infinita della decodifica differenziale. La vecchia
regola "no precoder in E2E" è superata su proposta di Marco stesso.

## Ipotesi
Con la RX complex di Asfand (32-64-16, win 5) e il precoder differenziale al DPD, la nostra
E2E (TX-DSP nostro, catena JLT) migliora la curva attuale W=16 D=1 e diventa confrontabile
1:1 coi risultati di Asfand/Stella. Il precoder è la leva che sblocca il bit di fase (il bit
raw vive nella transizione di segno → finestra finita basta).

## Criterio di successo (pre-registrato)
Battere la nostra curva attuale N_ADC=6 (riferimento: 1.46e-3 @ OSNR 17, file
fig10_dualaxis.txt riga "osnr 6") su ≥2 seed a parità di N_ADC=6, e quantificare il delta
del precoder (A/B). Soglia di rilevanza per il precoder: |delta| > varianza di seed osservata.

## Budget
11 training (QUEUE in $TEMP/drive_asfand.py): {pre ON/OFF}×{8,6,5}×s0 (6), poi seed 1 su
4 config chiave e seed 2 sulla headline (pre1-adc6). Checkpoint SALVATI (per gli eye).

## Journal
- 20/7 smoke test: architettura verificata (3606 param RX: 10→32→64→16+teste; legacy intatta
  a 470); precoder propagato al TX. Segnale early: a 500 step BER 0.14 con precoder vs 0.21
  del W=16 senza — da confermare a convergenza.
- 20/7 sera: campagna lanciata (task b6f4ta7cs, log $TEMP/asfand.log →
  results/paper/asfand_results.txt, entrambe le griglie native).
- 20/7 sera, primi 3 run (seed 0): CRITERIO GIÀ BATTUTO. N_ADC=6 @OSNR17: no-precoder
  1.9e-4 (gain 2.9 dB @1e-3 vs baseline in-house!), precoder 3.6e-4 (2.5 dB); adc8 pre1
  3.9e-4. La Asfand (3.6k param, win 5) batte il nostro W=64 (win 11) di ~2.5×: la finestra
  corta NON penalizza, il gruppo aveva ragione. SORPRESA A/B: il precoder di Marco PEGGIORA
  (~0.4 dB) al seed 0 — plausibile: senza precoder l'E2E usa già il trucco T/2 (bit su slot
  indipendenti), il precoder vincola senza sbloccare. DA CONFERMARE sui seed 1-2 prima di
  dirlo a Marco.
- 20/7 sera: eye al fotorivelatore generati dai checkpoint (eye_jlt_*.png, 2 pannelli:
  uscita PD e post-Gauss 10 GHz, noiseless, CPU): forma d'onda multilivello con ~4-5 bande
  di ampiezza e transizioni fitte — eye "chiuso" come previsto dall'handoff (la decisione
  vive nella finestra RX, non a soglia); il post-Gauss pulisce ma non apre. Da accompagnare
  a Marco con questa lettura.

- 20/7 notte, 8/11 run: **CRITERIO SODDISFATTO SU 2 SEED** — no-precoder N_ADC=6 @OSNR17:
  s0 1.9e-4, s1 4.5e-4, entrambi 3–7.5× sotto la soglia (1.46e-3). Varianza di seed ~2.3×
  (molto meglio del ~10× dei W piccoli a finestra 11). **A/B precoder CONFERMATO su 2 seed:
  il precoder PEGGIORA di ~1.8–1.9× in BER** (s0: 3.6e-4 vs 1.9e-4; s1: 8.0e-4 vs 4.5e-4)
  — coerente: senza precoder l'E2E usa il trucco T/2, il precoder aggiunge vincolo senza
  sbloccare nulla. no-precoder famiglia completa seed 0 @OSNR17: adc8 1.5e-4, adc6 1.9e-4,
  adc5 5.6e-4.

- 20/7 notte, campagna COMPLETA (11/11). Figure finali: e2e_asfand_{osnr,ebn0}.{pdf,png}
  (stile identico alla coppia inviata a Luca, baseline tratteggiate); gain in
  asfand_gains.txt.

## Verdetto (20/7 notte, contro il criterio pre-registrato)
**PROMOSSA (variante SENZA precoder).** Criterio battuto su 2 seed (N_ADC=6 @OSNR17:
1.9e-4 / 4.5e-4 vs soglia 1.46e-3). Gain @1e-3 vs baseline in-house a parità di N_ADC:
**+2.8 dB (adc8), +2.9/+2.4 dB (adc6, 2 seed), +3.5 dB (adc5)** — il "almeno 3 dB con E2E"
di Luca è sostanzialmente raggiunto con una RX da 3.6k parametri (metà del nostro W=64).
**A/B precoder: NEGATIVO, consistente su 5 coppie** (ON peggiora 1.3–2.5×; ON è anche più
instabile: seed2 a 2.15e-3). Spiegazione: senza precoder l'E2E aggira la memoria infinita
col trucco T/2 (bit su slot indipendenti); il precoder aggiunge un vincolo senza sbloccare
nulla. DA COMUNICARE A MARCO con tatto e coi numeri: la sua diagnosi (memoria infinita) era
giusta, ma la rete aveva già trovato da sola una soluzione diversa dal rimedio proposto.
Candidata a sostituire le curve nel paper (decisione con Dario/Luca): confrontabile 1:1 con
Asfand per struttura RX, window 5, stessa famiglia N_ADC.
Caveat onesti: seed singolo per adc8/adc5 (adc6 ha 2 seed); figure a seed 0 dichiarato.
