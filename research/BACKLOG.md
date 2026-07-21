# BACKLOG idee (I=interesse, F=fattibilità, N=novità, 1–10; costo in # training)
# [RICOSTRUITO 20/7: il file originale è sparito da disco prima di ogni commit]

## Attiva
- (nessuna — la prossima si sceglie con Dario; candidate: chiarimento window Asfand,
  multi-seed/mediana per le curve definitive del paper, dati Stella per il surrogato)

## Da chiarire col gruppo
- **"Context window is 5" di Asfand: simboli o campioni?** Noi abbiamo assunto 5 SIMBOLI
  (10 campioni a 2 sps). Se fossero 5 campioni il confronto va riqualificato. Chiedere a
  Luca/Asfand; in alternativa misurare la variante a 5 campioni (~1h GPU).

## Regole aggiornate (20/7, dal meeting)
- Il precoder differenziale in E2E è ORA AMMESSO (proposta di Marco: input DPD codificato,
  uscita bit normali) — la vecchia regola "no priori/no precoder" si restringe a: niente
  level-spacing imposto, niente pulse shape imposta.
- D=2 / reti complex AUTORIZZATE dal gruppo (Stella usa la complex di Asfand): decade la
  riserva "D=2 ultima risorsa".

## In coda
- **Surrogato data-driven sui dati sperimentali di Stella** (dal meeting 20/7): sbloccare i
  Conv1d/parametri di channel.py e fittare su sequenze I/O misurate (MSE, validazione stile
  legacy = confronto tap FIR). I9 F5 N8 — BLOCCATA finché Stella non manda i dati
  (chiedere: sample rate, allineamento TX/RX, lunghezze, catena del setup).
  NARRATIVA INTERNA (Dario 20/7, NON per il paper — per il paper a Ramin va solo
  modeling+training): i due approcci sono COMPLEMENTARI — sistema simulato → funzioni
  differenziabili esatte (surrogato a errore zero, allenarne uno non serve); sistema
  sperimentale → surrogato allenato sui dati misurati, poi congelato per l'E2E. Stessa
  struttura, cambia solo fissato-da-specifiche vs appreso-dai-dati. Punto di partenza
  MATLAB: MLforCPO/autoencoderJointBPAM.m (ultimo script di Dario, "non funziona benissimo")
  → porting/rifacimento in PyTorch autorizzato.
  CODE-READY 20/7 (direttiva di Dario prima di uscire): functions/surrogate.py —
  build_channel(config) con channel_source "physics"|"surrogate" (innestato in train.py:
  swap senza toccare l'E2E), fit_surrogate (Adam+MSE, ricetta del .m con LR ×0.9),
  white_noise_drive (eccitazione format-agnostic), compare_taps (validazione fisica),
  load_measured_io (stub .npz/.mat con TODO sul formato di Stella). SELF-TEST PASSATO su
  CPU: canale "sperimentale" perturbato (PD 18 GHz, driver 35 GHz) recuperato dai soli I/O
  white-noise — val MSE 7e-3→5e-6 (1400×), reload dalla factory congelato e fedele su dati
  held-out (errore 1600× sotto la potenza di segnale). CAVEAT IDENTIFICABILITÀ per i dati
  veri: la risposta in CASCATA si identifica benissimo, i singoli blocchi no (i tap si
  ridistribuiscono tra FIR in serie) — per tap interpretabili blocco per blocco servirà
  regolarizzare o congelare i blocchi noti.
- **Train su range di OSNR** (come Li, [9,17]) invece che punto fisso @14: possibile
  stabilizzatore di bacino. I7 F9 N5, ~4 run.
- **Best-of-3-seed / mediana come metodologia dichiarata** per le curve del paper: risolve la
  varianza di bacino C-band. I8 F10 N3, decisione editoriale.
- **weight_bits (STE sui pesi)** per la risposta complessità ASIC ("a 8 bit la BER non
  cambia"). I6 F8 N4, ~3 run.
- **Stabilizzatori di bacino**: LR schedule/warmup. I6 F8 N4, ~4 run.
- **Chirp α=+2** — existence proof fatta (parity b2b in C-band), NON azionabile: solo frase
  nel paper. I4 F10 N7, 0 run.
- **Fotocorrente a 4 sps al RX** — già provata, non aiuta. NEGATIVA, per la discussione.

## Chiuse
- **Follow-up Marco** (21/7) — CHIUSA, entrambe le domande: (1) il guadagno è TUTTO
  dell'ottimizzazione TX congiunta (RX-only NN ≈ AE lineare, E2E 20× meglio); (2) TX più
  complesso peggiora (W=16 mem-5 saturo). Card: `ideas/2026-07-21_marco-followup.md`.
- **E2E Asfand-comparable** (20/7) — PROMOSSA: gain +2.8/+2.9/+3.5 dB @1e-3 (N_ADC 8/6/5)
  vs baseline in-house; A/B precoder NEGATIVO consistente (verificato = encoder esatto di
  Marco). Card: `ideas/2026-07-20_asfand-comparable.md`. Figure/dati: e2e_asfand_*.
- **3 dB @ D=1** (17/7) — NEGATIVA contro criterio pre-registrato: tetto D=1 = 1.9–2.2 dB
  mediani @1e-3 (N_ADC=6). Card: `ideas/2026-07-17_d1-3db-gap.md`. 36 training.
- **Rimuovere TX-Gaussian** (17/7) — NEGATIVA: vale 0.15–0.5 dB. fig8_notxgauss.txt.
- **W=128 D=1 come sostituto di D=2** (17/7) — NEGATIVA: satura al livello W=64.
- **W=16 con finestra RX larga** (17/7) — NEGATIVA: peggiora (diluizione).
- **Baseline BPAM+AE in-house** (20/7) — PROMOSSA: simulatore del prof + quantizzatori
  DAC/ADC aggiunti, stesso scenario; gain reali W=16: +1.25/+1.26/+1.89 dB @1e-3 (N_ADC
  8/6/5). bpam_ae_reference.txt, fig10_adc_dario, fig10_ebn0_dario.