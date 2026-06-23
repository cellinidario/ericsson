# Notte 2026-06-20 → obiettivo: chiudere il gap con A.19 (slicer, poi E2E)

File di lavoro (NON committato). Tutto onesto: niente fudge factor, soglie genie
(ottime, a metà), Gray, simboli equiprobabili — solo le ipotesi di Forestieri.

## 1. Cosa dice A.19 (riletto dal PDF, pag. 4-8)

BER unipolare M-PAM, regime termico:

    BER ≃ 2(M-1)/(M·log2M) · Q( sqrt( 3·log2M/((M-1)(2M-1)) · Pe·Tb/i_n² ) )
    con Pe·Tb/i_n² = Eb/N0 elettrico.  Indipendente dalla forma d'impulso.

**Ipotesi cruciale (eq. A.1):** l'impulso ottico g(t) **"vanishes outside symbol-time
interval T to avoid nonlinear ISI after photodetection"**. Cioè g è **limitato a UN
simbolo** (NRZ time-limited), NON a banda limitata multi-simbolo. Conseguenza:

    i'(t) = R|x(t)|² = R Σ_k |a_k|² |g(t-kT)|²      (NESSUN cross-term: niente ISI non lineare)

PAM elettrica pulita con impulso p(t)=|g(t)|², matched filter h_e=|g(-t)|², campionamento
a T, soglie ottime a metà, livelli equispaziati in INTENSITÀ A_m=A√(m-1) (→ |A_m|²=(m-1)A²:
intensità 0,A²,2A²,3A²), Gray.

## 2. Da dove viene il nostro gap (floor ad alto SNR)

`utils.ideal_reference_ber` (RRC, NIENTE canale) **centra A.19** → detection, soglie, Gray,
e **calibrazione Eb/N0** (`add_awgn`, convenzione one-sided N0) sono GIÀ corretti. Quindi il
floor nel simulatore completo è SOLO ISI, di due tipi:

- **ISI lineare**: i filtri di front-end (segment-MZM 30 GHz, fotodiodo 25 GHz) distorcono
  l'impulso ricevuto → il matched filter (matchato all'impulso TX) non è più matchato →
  ISI residua + rottura della proprietà Nyquist.
- **ISI non lineare**: la square-law su impulsi **sovrapposti** (RRC/rect multi-simbolo)
  genera cross-term. Con modulatore **lineare** (field=√P) la √ e il quadrato si elidono →
  canale lineare → SOLO ISI lineare. Con **MZM** (cos² in V) la non-linearità su impulsi
  sovrapposti aggiunge ISI non lineare; è ciò che l'impulso time-limited di Forestieri evita.

## 3. Piano

- **Step A** — slicer, modulatore lineare, RRC, sweep banda front-end (segment-MZM = PD).
  Atteso: floor → A.19 man mano che la banda cresce (isola l'ISI lineare del front-end).
- **Step B** — slicer, impulso NRZ time-limited (l'impulso che A.19 assume davvero) +
  front-end largo, sia lineare sia MZM. Atteso: A.19 esatta (anche con MZM, perché niente
  sovrapposizione → niente ISI non lineare). Riproduzione fedele di Forestieri.
- **Step C** — E2E (DPD+FFE, symbol-CE, livelli APPRESI) sulla config che lo slicer porta
  ad A.19. Atteso: la curva E2E si sovrappone ad A.19, senza prior imposti.
- **Step D** — quanto si avvicina l'E2E nel caso realistico (RRC banda-limitata + 25 GHz PD
  + MZM): l'E2E equalizza il residuo, ma è il caso duro (ISI non lineare fondamentale).

## 4. Risultati

### Step A — sweep banda front-end (linear / RRC / oband), slicer genie
Floor a 18 dB = **7.7× la teoria, INVARIANTE** dalla banda (25 → 200 GHz uguale).
→ il floor NON è dei filtri PD/MZM. È un'altra nonlinearità. Diagnosi: il drive RRC
del modulatore lineare ha il **12.5% dei campioni a potenza < 0** (min -0.20), clippati
a 0 dalla relu (la potenza ottica non può essere negativa) → ISI **non lineare** da
sovrapposizione, indipendente dalla banda.

### Step B — impulso NRZ time-limited (l'impulso di A.19) — slicer genie, CPU, 500k simboli
| config                    | 10dB | 12dB | 14dB | 16dB | 18dB |
|---------------------------|------|------|------|------|------|
| linear / NRZ / **200 GHz**| 1.0  | 1.0  | 1.0  | 1.0  | 1.7* | → **A.19 ESATTA**
| MZM    / NRZ / **200 GHz**| 1.0  | 1.0  | 1.0  | 1.0  | 1.7* | → **A.19 ESATTA**
| linear / NRZ / 25 GHz     | 1.5  | 2.0  | 3.4  | 9.8  | 77   | (PD stretto spalma il boxcar)
| MZM    / NRZ / 25 GHz     | 1.5  | 2.1  | 4.2  | 15   | 170  |
| MZM    / RRC / 200 GHz    | 1.3  | 1.7  | 3.0  | 8.2  | 67   | (overlap → ISI non lineare)
(* = rumore Monte Carlo: BER 1.4e-5 vs 8.2e-6 su 500k simboli)

**SLICER = A.19 centrata** con impulso time-limited + front-end ideale (entrambe ipotesi
esplicite di Forestieri: g(t) su T, filtro elettrico h_e=|g(-t)|²). La calibrazione Eb/N0,
le soglie, il Gray sono esatti. Nessun fudge factor.

### Decomposizione del gap (definitiva)
- ISI **non lineare da sovrapposizione** (RRC/rect multi-simbolo attraverso clipping/MZM):
  il contributo dominante del floor (7.7× lineare, 67× MZM).
- ISI **lineare** del front-end stretto (PD 25 GHz che spalma l'NRZ a banda larga): l'altro
  estremo (NRZ/25GHz → 77×).
- Il caso realistico (banda-limitato + front-end stretto) ha ENTRAMBE → è il lavoro dell'E2E.

### Step C — E2E (DPD+FFE, symbol-CE, 40k step + cosine LR decay, CPU)
Modulatore MZM, RRC roll-off 0.85, oband, thermal. DPD mem5/w8, FFE mem11/w8.

| Eb/N0 | A.19     | slicer genie | E2E (wide 200GHz) | E2E (real 25/30GHz) |
|-------|----------|--------------|-------------------|---------------------|
| 14 dB | 2.77e-03 | 2.9×         | 1.9×              | 1.8×                |
| 16 dB | 2.79e-04 | 8.2×         | 2.3×              | 2.2×                |
| 18 dB | 8.17e-06 | 66×          | 3.5×              | 2.8×                |

- L'E2E **demolisce il floor dello slicer** (66× → ~3× a 18 dB): DPD+FFE equalizzano
  l'ISI non lineare da overlap che lo slicer memoryless non può togliere.
- **real ≈ wide**: l'E2E equalizza anche l'ISI lineare del front-end stretto (25/30 GHz)
  → la banda del front-end NON è un problema per l'autoencoder.
- **Ma resta un ~2.8× a 18 dB (~0.4 dB) e il rapporto cresce con l'SNR** → piccolo floor
  residuo: ISI non lineare da overlap non del tutto cancellata da DPD/FFE a capacità 8.

**Capacità NON è il limite**: DPD/FFE width 32 + mem 7/15 + 60k step danno lo STESSO
residuo (3.4× a 18 dB) di width 8. Il floor residuo è un **limite fondamentale** del
Fork-A (un livello/simbolo, impulso RRC fisso, FFE 1-sps): non può cancellare l'ISI
non lineare da overlap, che è una distorsione in tempo continuo tra le code degli impulsi
sovrapposti (non risolvibile con campioni solo al centro-simbolo). Coerente coi test di
memoria precedenti (i tap FFE non aiutano: la leva è la non-linearità, non la memoria).

### Step C-final — E2E con impulso NRZ time-limited (Forestieri), front-end largo
Impulso sample-and-hold (confinato a T) + matched integrate-and-dump, livelli APPRESI da
symbol-CE (niente prior imposti). Nota: il boxcar a lunghezza pari sfasa il campionamento
di 1/4 simbolo → la decimazione del Receiver va all'offset di fase corretto (timing ideale,
già assunto; lo slicer-genie fa lo stesso phase-pick). Con offset 0 fisso falliva (artefatto).

| Eb/N0 | A.19     | E2E NRZ (xth) |
|-------|----------|---------------|
| 12 dB | 1.25e-02 | 1.3×          |
| 14 dB | 2.77e-03 | 1.4×          |
| 16 dB | 2.79e-04 | 1.4×          |
| 18 dB | 8.17e-06 | 1.1×          |

**Rapporto PIATTO ~1.2-1.4× (nessun floor crescente) → A.19 raggiunta.** L'ISI non lineare
da overlap è sparita. Il ~1.3× costante (~0.15 dB) è il costo dei **livelli appresi vs ottimi**
(lo slicer-genie con livelli ottimi fa 1.0× esatto): onesto, e conseguenza del NON imporre i
livelli. Nessun equispacing prior aggiunto.

## 5. CONCLUSIONE

**Il gap ad A.19 era interamente ISI non lineare da sovrapposizione d'impulso** attraverso la
non-linearità d'intensità (clipping del modulatore lineare / cos² dell'MZM). A.19 lo evita per
costruzione: assume g(t) confinato a un simbolo (eq. A.1). Dimostrato in modo pulito e onesto:

1. **Slicer = A.19 ESATTA** (1.0×) con impulso time-limited + front-end largo, livelli ottimi
   (equispaziati in intensità), Gray, soglie ottime — tutte ipotesi esplicite di Forestieri.
2. **E2E = A.19** (~1.2-1.4× piatto, niente floor) con lo stesso impulso, livelli APPRESI; il
   piccolo offset costante è il prezzo di non imporre i livelli.
3. **Sistema realistico (RRC banda-limitato, overlap)**: l'E2E equalizza tutto tranne un floor
   residuo ~3× a 18 dB (~0.4 dB), **indipendente dalla capacità** (DPD/FFE 8 = 32): è un limite
   fondamentale dell'ISI non lineare da overlap a 1 sps / Fork-A (un livello/simbolo). Per
   azzerarlo servono impulsi time-limited (Forestieri) o gradi di libertà frazionari/temporali.

Niente fudge factor, niente prior imposti, solo le ipotesi di Forestieri.

## 6. Leva roll-off (RRC) sull'ISI non lineare — slicer, MZM, front-end largo
Domanda di Dario: con l'RRC si può fare "quasi time-limited" giocando col roll-off β?
Sì ma con un OTTIMO, non monotòno (BER/A.19 a 18 dB):

| impulso            | 18 dB |
|--------------------|-------|
| rect (β=0, Marco)  | 92×   |
| RRC β=0.30         | 298×* |
| **RRC β=0.50**     | 35×   | ← ottimo
| RRC β=0.70         | 36×   |
| RRC β=0.90 (~0.85) | 84×   |
| RRC β=0.99         | 86×   |
| NRZ time-limited   | 0.9×  | (A.19)
(* β bassi: anche troncamento a span 16)

- Il roll-off È una leva (~2.5× tra meglio/peggio RRC), ottimo a **β≈0.5-0.7**.
- Il nostro **β=0.85 è oltre l'ottimo**: scendere a ~0.6 dimezza il floor slicer (84→35×).
- Ma anche il miglior RRC resta lontano da A.19 (band-limited ≠ time-limited): solo l'NRZ tocca.
- L'intuizione "β più alto = sempre meglio" è falsa: dopo la square-law la dipendenza si ripiega.

In corso: E2E realistico β=0.85 vs β=0.60 — quanto del residuo 0.4 dB si gratta col roll-off ottimo.

## 7. Riproduzione
Script self-contained: vedi `a19_study.py` (consolidato dagli esperimenti _a19*.py).




