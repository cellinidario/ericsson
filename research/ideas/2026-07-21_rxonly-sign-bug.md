# Indagine: RX-only sopra l'adaptive equalizer (bug del segno) — 21/7

## Origine
Domanda 1 di Marco (RX-only vs E2E). La figura rxonly_osnr_dario mostra il nostro RX-only
SOPRA l'adaptive equalizer (impossibile per una NN se il segnale è buono) → Dario segnala il
bug. Indagine lunga, molte ipotesi mie SBAGLIATE (elencate sotto per non ripeterle).

## Fatto centrale (verificato)
Il segno differenziale in DD È RECUPERABILE sulla nostra catena: il simulatore del prof
(equalizzatore lineare LLS 23 tap, decisione sign(z2-soglia)), sulla NOSTRA catena esatta
(cband 10.238, MZM ER25, gauss10 TX, WSS, gauss10 RX, N_DAC=6), fa @OSNR17:
  BER_tot 4.1e-3, BER_amp 2.7e-3, **BER_phase 5.4e-3**  (diag_sign_prof.m).
Il NOSTRO RX-NN fa invece BER_phase ~1.9e-1 (per-bit split, diag_perbit.py). Ampiezza perfetta
(3e-5) in entrambi. → IL PROBLEMA È NEL NOSTRO RICEVITORE NN, non nel segnale né nel TX.

## Punto chiave emerso alla fine
L'equalizzatore lineare del prof È GIÀ la baseline "BPAM + adaptive eq." di complex_osnr_dario.
Quindi NON va replicato (darebbe la curva già presente). La domanda di Marco vuole il NOSTRO
RX-NN senza TX-NN; il nodo è: perché la NOSTRA NN (più potente di un lineare 23-tap) NON
estrae il segno? Da capire, NON ancora risolto.

## Ipotesi MIE SMENTITE (non ritentare)
- Finestra RX corta (5): NO, win 5/11/15 identiche (~9.6e-3). diag_rxonly_window.py
- Training a OSNR fisso vs range: NO, il range aiuta pochissimo. (la complex-headline va bene
  allenata a OSNR fisso → non è quello)
- Init collassato del TX complex: NO, spread iniziale ok (1.1). (era un altro problema: floor
  txcomplex = under-training, confermato a 200k)
- Target differenziale sbagliato (bpam_diff_target coded): implementato e testato, PEGGIORA
  (BER_phase 0.20). Il decoder differenziale amplifica l'errore residuo. Codice corretto
  (unit-test decode(precode)=identità) ma NON è la causa dominante.
- Saturazione MZM / drive swing: NO, invariante da 0.3 a 1.0 Vpi. diag_drive.py
- Separazione segno "intrinsecamente ~0.26": stima MIA sbagliata (guardavo residuo 3-termini,
  non l'equalizzatore 23-tap); il prof separa bene sullo stesso segnale.

## Prossimo passo (deciso con Dario: "replicare i 2 rami del prof")
PRECISAZIONE emersa: replicare l'equalizzatore LINEARE del prof = la baseline già esistente.
La cosa utile è invece capire perché la NOSTRA NN fallisce dove il lineare riesce. Sospetti
non ancora testati: (a) la sigmoid/BCE perde la struttura bipolare del cross-term che il
sign()-su-valore-con-segno del prof sfrutta; (b) il nostro RX-NN per bpam prende i campioni a
2 sps ma la decisione di fase potrebbe non essere ancorata al campione T/2 giusto.

## SVOLTA (21/7 sera): sampling rate sotto-campiona il segno
- diag_phases.py: separazione segno a T/2 = 0.26 a sps_sim=4, **0.69 a sps_sim=8** (quasi 3x).
  Il cross-term del segno a T/2 è ALIASATO dai nostri 4 campioni/simbolo. Ampiezza identica
  (0.238) a 4 e 8 → l'ampiezza non ne soffre, il segno sì. Il prof usa Nxs=8: ecco perché
  lui decodifica il segno e noi no. (I miei LLS Python artigianali NON sono test affidabili —
  scartati; il dato solido è la mappa di separazione a sps 4 vs 8.)
- RISULTATO sps=8 (diag_rxonly_sps8.py, RX-NN vero, 100k): bit di fase 1.9e-1 -> **1.9e-2**
  (10x meglio!), ampiezza 4e-5. MA non basta: il prof fa BER_phase 5.4e-3 sullo stesso segnale
  → resta un fattore ~4x tra il nostro RX-NN e l'equalizzatore lineare del prof. Il sampling
  rate era UNA causa (grossa, 10x) ma NON l'unica. BER_tot @OSNR17 = 9.96e-3, dominata dalla
  fase.
  → CONFERMA la strategia di Dario: replicare l'equalizzatore del prof a sps=8 isola il 4x
    residuo (ricevitore neurale vs lineare, o altro nella catena).

## STRATEGIA (decisa da Dario 21/7 sera)
1. Replicare l'equalizzatore LINEARE del prof in Python (pamb_equalizer_LLS + soglie), verificare
   che riproduca la baseline BPAM+AE → punto di partenza GARANTITO funzionante.
2. Da lì: TENERE il TX classico (ora verificato corretto) e riattaccare il RX-NN, isolando il
   problema al solo ricevitore. Metodo giusto: partire da qualcosa che funziona, non indovinare.
Ricevitore del prof letto e pronto: 2 filtri LLS 23-tap (C pari→|A|² ampiezza, D dispari→
0.5·√(A_k A_{k+1})·s segno), soglie ottime (fminbnd), decisione sign(z2_eq−soglia). b_tx ha N-1
elementi (fase tra simboli adiacenti).

## SCOPERTA target del segno (21/7 notte, diag_crossterm_target.py)
Il residuo a T/2 (sps=8) correla **+0.75 col PRODOTTO dei segni adiacenti s_k*s_{k+1}** (= il
cross-term fisico 2Re(x_k x*_{k+1})), e solo -0.02 col segno assoluto s_k. Il phase-diff bit
raw È il prodotto (segni uguali->bit0, diversi->bit1): si legge direttamente, NIENTE decoder
differenziale serve. → il target corretto del ramo segno è il prodotto, non s_k.
Un classificatore LINEARE sul residuo T/2 con questo target dà phase BER 1.2e-1 @OSNR17
(linear_sign_clean.py) — usa il segnale (non più 0.5) ma lontano dal prof (5.4e-3): il cross-term
è MODULATO dalle ampiezze (√(A_k A_{k+1})), un lineare puro non lo sfrutta appieno. Serve il
decisore giusto (la NN, o il √ esplicito). QUESTO è il pezzo che mancava alla RX-NN: non sapeva
che il segno vive nel prodotto modulato dall'ampiezza.

## SVOLTA DEFINITIVA (22/7 notte): il gap è nel SEGNALE, non nel decisore
Test cross-linguaggio (export_samples.py -> test_py_samples.m): il VERO equalizzatore del prof
(pamb_equalizer_LLS + sue decisioni), applicato ai NOSTRI campioni Python (sps8, rx_gauss,
::4), fa BER_phase **1.82e-2** — identico ai nostri decisori Python (2e-2), 4x PEGGIO del suo
5.4e-3 sulla sua catena. Ampiezza perfetta (8.7e-5).
→ IL DECISORE NON È IL PROBLEMA (è letteralmente il suo codice). La differenza è nel SEGNALE:
  la nostra catena Python produce un cross-term di segno più debole della catena MATLAB.
Escluso finora nel decisore: tap, fase, target (assoluto vs prodotto vs scalato-ampiezza),
indicizzazione, singola vs doppia finestra. Escluso nel segnale: sampling rate (sps4->8 = 10x,
già fatto), WSS (toglierlo PEGGIORA). Ampiezza sempre perfetta.
CANDIDATI residui nella catena (TX->fotocorrente) da confrontare Python vs MATLAB:
  - pulse shaping TX: noi rect20+gauss10; il prof 'G' 10 GHz singolo (o NRZ). L'impulso diverso
    cambia la SOVRAPPOSIZIONE a T/2 -> forza del cross-term.
  - livelli di campo BPAM: noi {+-0.5,+-1} via arccos su tutta la caratt. MZM; prof Vpeak=0.6
    (quasi-lineare). Drive-swing test dava invariante MA era sul residuo 3-tap (metrica inaffidabile).
  - filtro post-detection / matched: dettagli di implementazione.
PROSSIMO: confronto diretto del campo ottico / fotocorrente Python vs MATLAB sullo stesso
pattern noto, per vedere DOVE il cross-term si indebolisce.

## CAUSA TROVATA (22/7 notte): i LIVELLI DI DRIVE BPAM
Test col decisore VERO (prof_exact.py, LEVELMODE): i nostri livelli via arccos su TUTTA la
caratteristica MZM ({+-0.5,+-1} -> V ai bordi +-Vpi) COMPRIMONO il cross-term del segno.
Livelli LINEARI alla maniera del prof (drive prop. al campo, Vpeak frazione di Vpi):
  ours (arccos):        BER_phase 2.07e-2 @17
  prof Vpeak=0.6:       1.03e-2
  prof Vpeak=0.9:       **6.1e-3**  ~= prof (5.4e-3)  ← GAP CHIUSO
Fisica: arccos manda +-1.0 a +-Vpi (bordi caratt. MZM) dove la fase ottica satura e il cross-term
tra impulsi adiacenti (= il segno in DD) si comprime. Il drive lineare mantiene la relazione di
fase. → per il BPAM CLASSICO (ffe/threshold) usare livelli di drive lineari, non arccos.
VALORE: FEDELI AL PROF = Vpeak 0.6 (driver_baseline_jlt.m riga 35). Il 0.9/1.0 è ottimo per NOI
ma è nostra scelta → scartato, usiamo 0.6 come il prof (già chiude gran parte del gap: phase 1.0e-2
vs 2.0e-2). Il fix VERO è la mappa drive->campo LINEARE, non il valore di picco.
NB: questo NON tocca l'E2E (che scopre i suoi livelli). Solo il ramo BPAM-fisso.
PROSSIMO: portare il fix nel codice (transmitter fixed_levels per bpam classico = drive lineare a
Vpeak) e verificare che la RX-NN vera ne benefici -> obiettivo 1.

## VERDETTO OBIETTIVI (22/7, campagna ob1_results.txt)
OB.1 RX-only: **CENTRATO.** RX-NN + drive lineare 0.6 + sps=8 @OSNR17-20 (adc6):
  3.9e-3 / 1.9e-3 / 8.8e-4 / 3.7e-4  vs baseline BPAM+AE 4.1e-3 / 1.8e-3 / 7.1e-4 / 2.5e-4.
  ALLA PARI (prima 9.6e-3 @17, 2.5x peggio). Bug del segno RISOLTO.
  Fix = bpam_classic_drive_swing=0.6 (drive MZM lineare, non arccos-ai-rail) + sps_sim=8.
  TX BPAM ora equivalente al prof (campi ottici identici a meno di coniugazione, |field|^2 uguale).
OB.2 TX complex: NON ancora centrato. txcx 200k @17-19 (adc6): 6.9e-4 / 2.1e-4 / 6.1e-5 vs
  TX-simple 1.85e-4 / 3.9e-5 / 6.9e-6 -> 3.7-8.8x peggio, gap CRESCE a bassa BER. Floor sparito
  (era under-training a 100k) ma resta gap sistematico. Serve indagine come per OB.1
  (non solo "converge peggio"). Candidati: init dei layer intermedi, o il TX grande su bit RAW
  (no precoder) fatica a scoprire la segnalazione che il TX-16 trova.

## OB.2 MECCANISMO TROVATO (22/7, diag_constellation.py)
Il TX complex 32-64-16 scopre una costellazione COMPRESSA: livelli fotocorrente
[0.199,0.216,0.233,0.262] escursione 0.063, vs TX-16 [0.110,0.143,0.245,0.285] escursione
0.175 (~3x più aperta). Livelli ammassati -> più confondibili col rumore -> BER ~3x peggio.
CAUSA: la BER di TRAINING del complex si ferma a 4.3e-3 (vs 4.3e-4 del TX-16) — 10x peggio
ANCHE in training. Non è generalizzazione: la rete grande NON trova una buona soluzione. Con
3538 param e gradiente attraverso tutto il canale non-lineare (MZM cos + square-law),
l'ottimizzazione del TX è più difficile e si ferma in un minimo coi livelli compressi. Il TX-16
ha un paesaggio più semplice -> costellazione aperta. Escluso: under-training (200k),
overfitting (range), sampling (TX-16 sps8=sps4).
→ RISULTATO SCIENTIFICO (non bug): al TX più capacità = ottimizzazione più difficile attraverso
  il canale = costellazione peggiore. Il TX è più duro da allenare del RX (gradiente attraversa
  tutta la catena). Il TX-16 è OTTIMALE. Risponde a Marco domanda 2: ingrandire il TX non serve.
  Coerente con storia complessità/LUT.
POSSIBILE SBLOCCO (se si volesse comunque): warm-start del TX complex dai pesi del TX-16 (parte
dalla costellazione aperta). Non necessario per la conclusione.

## Stato codice (NON committato)
- functions/transmitter.py: coded_bpam_target, differential_decode (module-level, unit-tested)
- functions/train.py: bpam_diff_target path (coded target + differential decode in eval),
  _measure_ber_diff
- functions/config.py: bpam_diff_target knob (default "raw" = legacy, E2E invariato)
Da decidere se tenere (il target coded non risolve, ma il codice è pulito e potrebbe servire).
