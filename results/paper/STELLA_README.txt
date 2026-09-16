================================================================================
 Risultati Dario Cellini - JLT invited (AI for short-reach optical interconnects)
 Aggiornato: 22 luglio 2026
================================================================================

Stella, questi sono i risultati per le figure. Tutte le curve sono sull'asse OSNR
con la convenzione di Luca (mail 16/7):

    OSNR = Eb/N0 + 10log10(2) + 10log10(Rs / (2*Bref)) = Eb/N0 + 2.05 dB
    (Rs = 20 GBaud, Bref = 12.5 GHz, segnale 1 pol, ASE 2 pol)

Le versioni in Eb/N0 non sono convertite ma RI-SIMULATE nativamente su griglia
Eb/N0 intera (file *_ebn0_*), cosi' i due assi sono entrambi verificabili.

--------------------------------------------------------------------------------
SCENARIO COMUNE A TUTTE LE CURVE
--------------------------------------------------------------------------------
  20 GBaud BPAM-4 (bipolar PAM, precoding differenziale), direct detection
  C-band, 10.238 km SMF (beta2 = -21.7 ps^2/km)
  MZM ER = 25 dB, drive lineare a Vpeak = 0.6 (identico al simulatore di Marco)
  TX: 20 GHz rect + Gaussiano 10 GHz | WSS ottico B=1.6, BWotf=18 GHz
  RX: Gaussiano 10 GHz, 2 campioni/simbolo
  N_DAC = 6 bit; N_ADC = 8 / 6 / 5 bit

--------------------------------------------------------------------------------
FILE PRINCIPALI (in ordine di utilita' per le figure)
--------------------------------------------------------------------------------

1) complex_osnr_dario.txt   +   complex_ebn0_dario.txt
   *** E2E autoencoder (TX-DSP + RX-DSP allenati insieme) ***
   RX-DSP: FC 32-64-16 + sigmoide, context window 5 simboli (setup di Asfand,
   come specificato da Luca); TX-DSP: FC 16.
   Colonne: OSNR | E2E N_ADC 8,6,5 | BPAM+Ad.Eq. N_ADC 8,6,5
   -> e' la curva "headline": ~4 dB di guadagno sulla BPAM+adaptive eq. a 1e-3.

2) rxonly_osnr_dario.txt
   *** RX-only (TX BPAM standard, solo il ricevitore e' una NN) ***
   Stesse colonne. Confronto diretto e onesto con la Ad.Eq. a parita' di TX.
   NOTA: queste sono le curve su cui sto ancora lavorando per allinearmi ai
   risultati di Asfand (vedi sotto).

3) bpam_ae_reference_osnrgrid.txt
   *** Baseline BPAM + adaptive LLS equalizer ***
   Generata IN-HOUSE con il simulatore di Marco (PAM_bipolare_DD_nuovi__sorgenti),
   non digitalizzata da figure. Equalizzatore Lc = Ld = 11.
   Contiene anche le colonne SNR(Es/N0) ed Eb/N0 per il controllo delle conversioni.

4) fig10_adc_dario.txt + fig10_ebn0_dario.txt
   E2E con la nostra RX "semplice" (W=16, D=1) + famiglia N_ADC, se serve la
   versione a bassa complessita'.

--------------------------------------------------------------------------------
VALIDAZIONE DELLA CATENA (per rispondere al dubbio sollevato nel confronto)
--------------------------------------------------------------------------------
Ho passato i campioni della NOSTRA catena Python dentro l'equalizzatore LLS di
Marco (script check_py_chain.m): BER 1.46e-2 a OSNR 15, contro 1.5e-2 della
catena MATLAB di riferimento. Le due catene coincidono, quindi il segnale
ricevuto e' lo stesso e i confronti NN-vs-NN sono puliti.

--------------------------------------------------------------------------------
PUNTO APERTO: RX-only vs i risultati di Asfand
--------------------------------------------------------------------------------
La nostra RX-only e' ~2.6x sopra la curva di Asfand (a OSNR 15: 1.31e-2 contro
5e-3), mentre le due BPAM+Ad.Eq. coincidono - il che conferma che la differenza
e' nel ricevitore, non nella catena.

Ipotesi gia' testate ed ESCLUSE (nessuna chiude il gap):
  - lunghezza della context window (5 / 11 / 21 simboli: identiche)
  - decoder differenziale esplicito a valle (6x PEGGIO)
  - 4 campioni/simbolo al RX invece di 2 (identico)
  - due reti indipendenti sulla STESSA finestra (identico al tronco singolo)

TESTATA anche la configurazione descritta da Marco (mail 22/7): due reti su
context window di 11 campioni SFALSATE di 1 campione, ciascuna centrata sulla
propria grandezza. File: rxonly_stagger.txt. ESITO NEGATIVO:

  OSNR | tronco singolo | wide12  | stag11  | stag22  | Li
    15 |    1.31e-2     | 1.25e-2 | 1.43e-2 | 1.36e-2 | 5e-3
    17 |    3.68e-3     | 3.56e-3 | 4.54e-3 | 4.22e-3 | 1.2e-3
    19 |    7.45e-4     | 7.11e-4 | 1.09e-3 | 1.02e-3 | 2e-4

  wide12 = 1 rete, finestra 12 campioni, neuroni raddoppiati (la prova semplice
           che suggerivate tu e Marco) -> guadagna solo il 5%
  stag11 = 2 reti, finestre 11 campioni sfalsate di 1 (Asfand come descritto)
  stag22 = 2 reti, le nostre finestre da 22 campioni, sfalsate di 1 (controllo)

Le due reti sfalsate vanno ~1.1x PEGGIO del tronco singolo. L'implementazione e'
verificata (ho controllato l'algebra delle finestre su una rampa, non solo che il
codice giri): la spiegazione plausibile e' che dimezzare i dati per tronco costi
piu' di quanto renda la specializzazione.

QUINDI: dopo lunghezza della finestra, numero di reti, sfalsamento, capacita',
sps e decoder differenziale, l'architettura del ricevitore e' esclusa.

Sto seguendo una pista diversa: il gap CRESCE monotonicamente con l'OSNR
(1.46x a 13 -> 3.56x a 19). Questa e' la firma di un problema di efficienza di
ALLENAMENTO ad alto SNR, non di capacita' della rete. Due candidati concreti:
alleniamo su un RANGE di Eb/N0 (7-18) invece che sul punto operativo, e usiamo un
learning rate COSTANTE per 100k step senza scheduler (nei log il training e' in
plateau gia' a 12.5k step: fermo, non lento).

Una domanda per Li, se riuscite a chiedergliela: **come allena la sua rete?**
In particolare (1) allena una rete per ogni punto di OSNR o una sola su un range,
e (2) usa un learning rate decrescente. Se la risposta e' "una rete per punto,
con LR decay" il gap e' spiegato senza cercare altro.

--------------------------------------------------------------------------------
FORMATO
--------------------------------------------------------------------------------
Tutti i file sono testo con header commentato da '#', colonne separate da spazi,
direttamente leggibili con np.loadtxt() o load() di MATLAB.
