# Argine

Runtime locale per eseguire LLM quantizzati in formato **GGUF** su PC Windows con
**16 GB di RAM**, senza cloud, senza API a pagamento, senza Docker e senza che
l'utente debba compilare niente.

Argine non reimplementa l'inferenza: usa `llama.cpp` come motore e ci mette
sopra il livello che manca su una macchina con poca RAM, cioe' un **Memory
Planner** che decide *prima* di avviare se la configurazione ci sta in RAM, e in
caso contrario la riduce da solo invece di far esaurire la memoria a meta'
generazione.

```
Argine
    |
    +-- Hardware Detector      CPU, RAM, GPU, Vulkan, disco
    +-- Model Inspector        parser GGUF, detection MoE, contabilita' expert
    +-- Memory Planner         budget RAM + scala di sicurezza
    +-- Runtime Manager        llama.cpp precompilato o build patchata
    +-- llama.cpp (CPU/Vulkan)
    |     +-- Expert Store     buffer type C++: expert da disco, tetto rigido
    +-- Quantizzatore          expert a 2 bit + matrice di importanza
    +-- Misure                 perplessita', benchmark, diskbench
    +-- Local REST API         /v1/chat/completions compatibile OpenAI
```

**Lo stato in una riga:** la macchina di riferimento — Windows, 16 GB, nessuna GPU
dedicata — fa girare **Qwen3-Next-80B-A3B a 6,87 tok/s**, con 34,31 GiB di expert
serviti da un budget di 4 GiB e un picco di RAM di 6,11. Su Qwen3-30B-A3B, piu'
piccolo e a qualita' piena, fa **9,62 tok/s** (12,97 accettando il 13% di
perplessita' in piu'); due giorni prima erano 4,23.

Tutti i numeri di questo documento sono misurati, e la sezione
[Risultati misurati](#risultati-misurati) dice su quale macchina e con quale
binario — perche' si e' scoperto che conta piu' di quanto sembri. Il numero da 80
miliardi e' arrivato dopo due previsioni sbagliate, entrambe in difetto, entrambe
scritte prima della misura e lasciate scritte: vedi
[La misura](#la-misura-687-toks-cioe-previsione-sbagliata-del-37).

---

## Indice

<!-- indice: generato da strumenti/controlla_readme.py, non modificare a mano -->

- [Migrare da PicoStar16](#migrare-da-picostar16)
  - [Un messaggio che mentiva, trovato dalla rinomina stessa](#un-messaggio-che-mentiva-trovato-dalla-rinomina-stessa)
- [Installazione](#installazione)
  - [Uso offline](#uso-offline)
- [Comandi](#comandi)
  - [argine probe](#argine-probe)
  - [argine inspect modello.gguf](#argine-inspect-modellogguf)
  - [argine run modello.gguf](#argine-run-modellogguf)
  - [argine benchmark modello.gguf](#argine-benchmark-modellogguf)
    - [--sweep-expert-budget: misurare un modello che non si e' scaricato](#--sweep-expert-budget-misurare-un-modello-che-non-si-e-scaricato)
  - [Il costo per miss migliora quando il budget si stringe](#il-costo-per-miss-migliora-quando-il-budget-si-stringe)
  - [Il margine vero, e perche' il primo numero era sbagliato](#il-margine-vero-e-perche-il-primo-numero-era-sbagliato)
  - [Lo store legge 896 KiB, e il formato permetterebbe 2,75 MiB](#lo-store-legge-896-kib-e-il-formato-permetterebbe-275-mib)
  - [Letture asincrone: un no, un si', e uno strumento rotto da me](#letture-asincrone-un-no-un-si-e-uno-strumento-rotto-da-me)
  - [Il terzo tentativo: emetti e continua](#il-terzo-tentativo-emetti-e-continua)
  - [E cosi' ho sbagliato due volte lo stesso verdetto](#e-cosi-ho-sbagliato-due-volte-lo-stesso-verdetto)
  - [argine experts report|simulate modello.gguf](#argine-experts-reportsimulate-modellogguf)
  - [argine store build|info|verify|bench modello.gguf](#argine-store-buildinfoverifybench-modellogguf)
  - [argine diskbench FILE](#argine-diskbench-file)
    - [Le due leve, e quale si puo' tirare davvero](#le-due-leve-e-quale-si-puo-tirare-davvero)
  - [argine ppl modello.gguf](#argine-ppl-modellogguf)
  - [argine quantize modello.gguf e argine imatrix modello.gguf](#argine-quantize-modellogguf-e-argine-imatrix-modellogguf)
  - [argine logs / argine doctor / argine setup](#argine-logs--argine-doctor--argine-setup)
- [API locale: due dialetti](#api-locale-due-dialetti)
  - [/v1/messages: il dialetto Anthropic](#v1messages-il-dialetto-anthropic)
- [Gestione della RAM](#gestione-della-ram)
  - [La cache dei prompt, che nessuno contava](#la-cache-dei-prompt-che-nessuno-contava)
  - [Il buffer di repack](#il-buffer-di-repack)
  - [Scala di sicurezza](#scala-di-sicurezza)
- [Modelli MoE ed expert cache](#modelli-moe-ed-expert-cache)
  - [Simulazione](#simulazione)
- [L'expert store](#lexpert-store)
  - [Perche' non bastava mmap](#perche-non-bastava-mmap)
  - [Il formato PSXS](#il-formato-psxs)
  - [Come si innesta](#come-si-innesta)
  - [I bug che e' costato](#i-bug-che-e-costato)
- [Risultati misurati](#risultati-misurati)
  - [Le due configurazioni utilizzabili](#le-due-configurazioni-utilizzabili)
  - [Una conversazione che cresce](#una-conversazione-che-cresce)
  - [Il controllo: serviva davvero un modello grande?](#il-controllo-serviva-davvero-un-modello-grande)
  - [Da dove viene la velocita'](#da-dove-viene-la-velocita)
  - [La curva qualita' / bit](#la-curva-qualita--bit)
  - [Come un testo sbagliato ha prodotto il risultato sbagliato](#come-un-testo-sbagliato-ha-prodotto-il-risultato-sbagliato)
- [Ipotesi smentite](#ipotesi-smentite)
- [Logging](#logging)
- [KPI](#kpi)
  - [Cosa decide la scalabilita'](#cosa-decide-la-scalabilita)
  - [La curva, misurata](#la-curva-misurata)
  - [Tre strati di lettura, un binario solo](#tre-strati-di-lettura-un-binario-solo)
  - [La scelta del modello, e il timore che non si e' avverato](#la-scelta-del-modello-e-il-timore-che-non-si-e-avverato)
  - [La previsione, scritta prima di misurare](#la-previsione-scritta-prima-di-misurare)
    - [Prima previsione, sbagliata in mezz'ora](#prima-previsione-sbagliata-in-mezzora)
    - [I numeri veri, letti dal file](#i-numeri-veri-letti-dal-file)
  - [La misura: 6,87 tok/s, cioe' previsione sbagliata del 37%](#la-misura-687-toks-cioe-previsione-sbagliata-del-37)
    - [Perche': il router concentra, e con 512 expert concentra meglio](#perche-il-router-concentra-e-con-512-expert-concentra-meglio)
    - [E il collo di bottiglia si e' spostato](#e-il-collo-di-bottiglia-si-e-spostato)
    - [Cosa resta vero, e cosa va riscritto](#cosa-resta-vero-e-cosa-va-riscritto)
    - [La curva dell'80B: esponente 0,38 contro 0,58](#la-curva-dell80b-esponente-038-contro-058)
- [Stato rispetto alle milestone](#stato-rispetto-alle-milestone)
  - [Cosa manca](#cosa-manca)
- [Compilare la build patchata](#compilare-la-build-patchata)
- [Struttura del progetto](#struttura-del-progetto)
- [Risoluzione problemi](#risoluzione-problemi)
  - [Variabili d'ambiente](#variabili-dambiente)

<!-- fine indice -->

---

## Migrare da PicoStar16

Il progetto si chiamava **PicoStar16**. Il nome descriveva l'ambizione iniziale —
una stella nana, piccola, 16 GB — e quell'ambizione si e' rivelata sbagliata: la
tesi di partenza era che un modello grande e compresso "sappia piu' cose", e le
misure l'hanno smentita (vedi [Ipotesi smentite](#ipotesi-smentite)). Quello che
il progetto fa davvero e' tenere un modello dentro un limite di memoria che non
puo' superare. Un argine non rende il fiume piu' grande: gli dice dove stare.

Se avevi l'installazione vecchia, dall'Anaconda Prompt nella cartella del
progetto:

```bat
pip uninstall -y picostar16
pip install -e .
```

Il comando ora si chiama `argine` (non piu' `picostar`). Tre cose **non** si
rifanno:

- **Lo store `.psxs` resta valido.** Il numero magico del formato non e'
  cambiato: e' il formato dei dati, non il nome del prodotto, e cambiarlo
  avrebbe voluto dire ricostruire 16,35 GiB per un motivo estetico.
- **Lo storico delle misure viene ereditato.** Alla prima esecuzione, Argine
  copia `runs.jsonl`, `perplexity.jsonl` e i testi di prova da
  `%LOCALAPPDATA%\PicoStar16\logs` verso `%LOCALAPPDATA%\Argine\logs`. Sono le
  firme che rendono confrontabili le perplessita' fra loro: ricominciare da zero
  avrebbe buttato due giorni di misure su questa macchina specifica.
- **Le vecchie variabili `PICOSTAR_*` continuano a funzionare**, con un avviso
  che dice come si chiamano adesso. Sia dal lato Python sia dal lato C++, e non
  per cortesia: ignorare in silenzio un `PICOSTAR_LLAMA_DIR` rimasto
  nell'ambiente avrebbe fatto partire il binario sbagliato, che e' esattamente
  il bug che una volta ha nascosto un fattore 1,74x (vedi [I bug che e'
  costato](#i-bug-che-e-costato)).

La cartella sul disco puo' restare chiamata come vuoi: `pip install -e .`
funziona da qualunque percorso. Se la rinomini e `ARGINE_LLAMA_DIR` puntava
dentro la vecchia, `argine setup` te lo dice e propone il percorso nuovo gia'
pronto da incollare — vedi sotto, perche' la prima volta non lo diceva.

### Un messaggio che mentiva, trovato dalla rinomina stessa

Alla prima installazione dopo il cambio di nome, `ARGINE_LLAMA_DIR` era rimasta a
puntare `...\picostar16\llama-noomp`, che non esisteva piu'. La compatibilita' ha
fatto il suo lavoro — l'ha letta dal nome vecchio, l'ha detto, e si e' rifiutata
di partire — ma poi `install.bat` ha stampato *"download non riuscito. Puoi
riprovare con `argine setup`, oppure indicare una copia locale con
ARGINE_LLAMA_DIR"*. Tre cose sbagliate in due righe: non era un guasto di rete,
ritentare il download non lo tenta nemmeno (finche' la variabile e' impostata
`setup` non scarica niente, usa solo quella), e il consiglio finale era
esattamente la cosa che era impostata e rotta.

Il rifiuto era giusto, il messaggio no. Adesso `setup` distingue i due guasti, e
quando la cartella puntata non c'e' cerca una build con lo stesso nome dentro il
progetto — che e' il caso quasi sempre vero, perche' il motivo per cui il percorso
e' morto e' che la cartella si e' spostata e la build e' ancora dentro. Se la
trova, stampa il comando completo:

```
[argine] Ho trovato una build con lo stesso nome ('llama-noomp') dentro il
         progetto: e' quasi certamente quella che cercavi.
    setx ARGINE_LLAMA_DIR "C:\Users\...\Argine\llama-noomp"
```

Vale la pena registrarlo perche' e' lo stesso genere di errore dei tre messaggi
diagnostici ancora da ricalibrare in [Cosa manca](#cosa-manca): non codice che
sbaglia, codice che spiega male cosa ha fatto. Costa piu' di quanto sembri —
manda a provare la cosa inutile, e chi legge non ha modo di sapere che il
suggerimento e' vuoto.

---

## Installazione

Dall'**Anaconda Prompt** (o da un normale prompt dei comandi Windows):

```bat
cd argine
install.bat
```

`install.bat` installa il pacchetto e scarica i binari `llama.cpp` gia'
compilati per CPU e Vulkan. Non serve Visual Studio, non serve compilare.

In alternativa, a mano:

```bat
python -m pip install -e .
python -m argine setup --backend both
```

Se preferisci non installare nulla, `argine.bat` esegue il progetto in loco:

```bat
argine.bat probe
```

Unica dipendenza Python: `psutil` (gia' presente in Anaconda).

### Uso offline

Dopo `argine setup` tutto funziona senza rete. Se il PC non ha accesso a
Internet, scarica a mano lo zip dalle release di llama.cpp
(`llama-bNNNNN-bin-win-cpu-x64.zip` e/o `...-win-vulkan-x64.zip`), estrailo e
indica la cartella:

```bat
set ARGINE_LLAMA_DIR=C:\llama\bin
argine run modello.gguf --offline
```

---

## Comandi

### `argine probe`

Mostra CPU, core/thread, RAM totale e libera, GPU, backend disponibili e spazio
disco. Con `--deep` interroga i binari `llama.cpp` ed elenca i device Vulkan
reali (sulla macchina target deve comparire *Intel(R) Iris(R) Xe Graphics*).

```bat
argine probe
argine probe --deep
argine probe --json
```

### `argine inspect modello.gguf`

Legge l'header GGUF senza caricare i pesi e riporta dimensione, architettura,
quantizzazione (con il mix dei tipi e i bit per peso), numero di parametri,
se il modello e' MoE, quanti expert ha, quanti parametri sono davvero attivi per
token e la RAM stimata. Stampa anche il piano di memoria completo.

```bat
argine inspect C:\modelli\Qwen3-30B-A3B-Q4_K_M.gguf --ram-limit 11G
```

`argine plan` e' un alias dello stesso comando.

### `argine run modello.gguf`

Avvia il modello ed espone l'API locale.

```bat
argine run C:\modelli\Qwen3-8B-Q4_K_M.gguf --backend auto --ram-limit 11G --ctx 4096
argine run C:\modelli\Qwen3-30B-A3B-Q4_K_M.gguf --auto
```

Opzioni principali:

| opzione | significato |
|---|---|
| `--backend auto\|cpu\|vulkan` | scelta del backend (auto preferisce Vulkan se presente) |
| `--ram-limit 11G` | budget RAM massimo |
| `--ctx 4096` | context in token |
| `--threads 10` | thread di generazione |
| `--gpu-layers N` | layer offloadati su GPU |
| `--kv-type f16\|q8_0\|q4_0` | precisione della KV cache |
| `--expert-cache 5G` | budget della expert cache (MoE) |
| `--port 8080` | porta dell'API |
| `--auto` | profilo automatico: ignora ctx/kv/gpu-layers e li calcola |
| `--dry-run` | mostra piano e comando senza avviare |

### `argine benchmark modello.gguf`

Misura prompt tokens/sec, generation tokens/sec, time to first token, RAM peak e
media, page fault, throughput di lettura disco, utilizzo CPU e — su Windows —
utilizzo GPU dai contatori prestazioni.

```bat
argine benchmark C:\modelli\Qwen3-8B-Q4_K_M.gguf --n-predict 128 --repeat 3
```

Sulle CPU ibride Intel (P-core + E-core) il numero di thread ottimale non si
deduce dal conteggio dei core, ed e' **la singola scelta che pesa di piu'**: ggml
sincronizza i thread a ogni nodo del grafo e aspetta il piu' lento, quindi un
E-core — o il fratello SMT di un P-core gia' occupato — aggiunge un partecipante
all'attesa senza aggiungere lavoro fatto. Argine sceglie **un thread per
P-core**, che sulla macchina di riferimento vale +40-46% rispetto al doppio.

Su una CPU diversa va verificato, ed e' la prima cosa da fare:

```bat
argine benchmark modello.gguf --sweep-threads 4,6,8,12
```

Ripete il benchmark per ogni valore, con una pausa di raffreddamento fra l'uno e
l'altro perche' la deriva termica non falsi il confronto, e stampa una tabella
con il valore consigliato.

Il benchmark avvisa anche quando il **binario** e' cambiato rispetto all'ultima
misura dello stesso modello: due velocita' prese su build diverse non sono
confrontabili, e la differenza puo' arrivare a 1,74x.

#### `--sweep-expert-budget`: misurare un modello che non si e' scaricato

```bat
argine benchmark modello.gguf --expert-store --sweep-expert-budget 8G,6G,4G,3G
```

Quello che decide se questa architettura scala non e' la taglia del modello, e'
la **frazione degli expert che sta nel budget**. Un modello con tre volte gli
expert e lo stesso tetto di RAM si comporta come questo modello con un terzo del
tetto: cambia il tasso di miss, e il tasso di miss e' l'unica variabile che conta.
Quindi la curva si puo' misurare *oggi*, sul modello che si ha gia' in casa,
invece di scaricare decine di GiB per scoprire la stessa cosa.

L'uscita riporta, per ogni budget, la frazione residente e la velocita', e chiude
con un numero solo: l'**esponente** d(log velocita') / d(log residenza), **con la
sua incertezza**.

| esponente | significato |
|---|---|
| ~0 | la residenza non conta: si puo' crescere quanto si vuole |
| ~0,5 | un modello 4 volte piu' grande va 2 volte piu' piano |
| ~1 | la velocita' e' proporzionale alla residenza: **il tetto esiste** |

Si confrontano i due estremi e non una regressione su tutti i punti, perche' con
tre o quattro misure una regressione darebbe una precisione che i dati non hanno.

**L'incertezza non e' decorativa, e serve a tacere.** Il denominatore e'
log(escursione), quindi un intervallo corto fra il budget piu' alto e il piu'
basso amplifica il rumore: misurato, 1,33x di escursione da' 0,47 **±0,15**,
mentre 2,67x con la stessa rumorosita' da' ±0,04. Un esponente di 0,47±0,15 vale
[0,32, 0,62], e quell'intervallo contiene la soglia 0,35: non distingue "regge" da
"scende". Quindi il verdetto si pronuncia solo se l'escursione e' almeno **2x** e
l'intervallo non scavalca una soglia; altrimenti il programma dice che non lo sa e
suggerisce quale punto basso servirebbe.

La prima versione non aveva l'incertezza e ha stampato *"la velocita' scende in
modo sensibile"* da quei 0,47. Un verdetto che i dati non sostengono e' peggio di
nessun verdetto, perche' viene creduto. L'errore per punto si prende dalla
dispersione fra le ripetizioni quando esiste, e quando non esiste si assume il 3%
dichiarandolo.

**Due difetti trovati al primo uso vero**, entrambi nel modo in cui gli errori
venivano raccontati, non nel calcolo:

- **Un percorso sbagliato non fermava lo sweep.** Il file non esisteva, la prima
  configurazione falliva, il programma dormiva 20 secondi di raffreddamento e si
  preparava a ripetere lo stesso fallimento altre tre volte. Un file che non
  esiste non esistera' nemmeno al giro dopo; una configurazione che non ci sta in
  RAM invece va saltata e si prosegue. Sono guasti diversi e ora sono trattati in
  modo diverso: il modello si valida una volta, prima del ciclo. Il test che lo
  fissa, col bug reintrodotto, impiega 60,22 secondi invece di 0,2.
- **L'elenco dei file "ci sono pero' questi .gguf" nascondeva quello giusto.** Era
  alfabetico e troncato agli otto *senza dirlo*, e nella cartella c'erano nove
  GGUF dai nomi quasi identici. Su Windows l'ordinamento ignora le maiuscole e
  `...-Q4_K_M-reordered.gguf` precede `...-Q4_K_M.gguf`, perche' il trattino viene
  prima del punto: il file utile era il nono, tagliato. Il messaggio nato per
  aiutare ha elencato otto file sbagliati e omesso l'unico giusto. Ora l'elenco e'
  ordinato per somiglianza al nome scritto — il file cercato risale al primo posto
  con 0,91 di affinita' — e se tronca lo dichiara.

Il secondo e' istruttivo oltre il caso: **un elenco che puo' essere troncato va
ordinato per pertinenza, non per alfabeto.** Chi sbaglia a scrivere un nome ha
scritto qualcosa di simile a quello giusto.

Al secondo uso vero ne e' uscito un terzo, dello stesso stampo: lo sweep scopriva
*una configurazione alla volta* che il budget non ci stava, con 20 secondi di
pausa in mezzo, quando `planner.plan()` lo sapeva in millisecondi senza avviare
nulla. Ora c'e' un **pre-volo** che calcola il piano per ogni budget e stampa
subito la tabella:

```
== PRE-VOLO: QUALI BUDGET SONO AVVIABILI -------------------------
  expert nel modello       15.19 GiB
      budget  residente   stima totale                      esito
     8.0 GiB        53%       10.2 GiB        oltre il budget RAM
     6.0 GiB        40%        8.2 GiB        oltre il budget RAM
     4.0 GiB        26%        6.2 GiB                  avviabile
     3.0 GiB        20%        5.2 GiB                  avviabile
```

Se i punti avviabili sono meno di due, si ferma: due misure su quattro
produrrebbero una pendenza costruita sul nulla. Qui saltare *e'* legittimo — un
budget piu' grande della RAM libera non ci sta, e non e' un errore — ma dirlo
subito e tutto insieme e' la differenza fra una tabella in due secondi e un minuto
di pause per scoprire la stessa cosa.

E al terzo uso ne e' uscito un quarto, il piu' insidioso dei quattro perche' non
produceva nessun messaggio: **`PICOSTAR_EXPERT_PRETOUCH` era rimasta
nell'ambiente** da un esperimento di giorni prima. `PRETOUCH` azzera ogni slice
(`memset`) prima di leggerla, per attribuire il costo dei page fault separatamente
da quello dell'I/O; e' uno strumento di diagnosi e costa una scrittura piena per
ogni miss. La prima curva velocita'/residenza e' stata misurata cosi'.

Non e' un rallentamento costante che si potrebbe sottrarre a posteriori: il memset
si paga **per ogni miss**, cioe' esattamente sulla variabile che lo sweep fa
variare. La curva ne risultava piu' ripida del vero, e la distorsione cadeva
proprio sulla quantita' da misurare. L'unica traccia era la riga di compatibilita'
che diceva "uso PICOSTAR_EXPERT_PRETOUCH (nome vecchio)" — scritta per un altro
motivo, e l'unica ragione per cui ce ne siamo accorti.

Adesso Argine impone `ARGINE_EXPERT_PRETOUCH=0` sempre, e dichiara quando sta
ignorando una variabile residua. Si attiva solo chiedendolo con `--pretouch`. La
regola e' la stessa di `--cache-ram`: **cio' che non si dichiara lo decide qualcun
altro**, e "qualcun altro" puo' essere il proprio ambiente di tre giorni prima.

**E la direzione della distorsione era l'opposto di quella prevista.** Avevo
scritto che il memset, pagandosi per ogni miss, rendeva la curva *piu' ripida del
vero*. La misura ripetuta senza `PRETOUCH` dice il contrario:

| budget | residente | con PRETOUCH | senza | differenza |
|---|---|---|---|---|
| 4 GiB | 24% | 5,34 tok/s | **6,16** | +15,4% |
| 3 GiB | 18% | 4,67 tok/s | **5,04** | +7,9% |
| | | esponente 0,47 | **0,70** | piu' ripida |

Il guadagno e' maggiore dove i miss sono *meno*, e togliere `PRETOUCH` ha reso la
curva piu' ripida, non meno. Quindi i dati contaminati facevano sembrare la
scalabilita' **migliore** di com'e'.

Il meccanismo plausibile e' che `PRETOUCH` non aggiunga lavoro ma lo **sposti**:
Windows azzera le pagine impegnate al primo tocco comunque, e farlo in un `memset`
prima della lettura significa che il DMA scrive in pagine gia' toccate. Costo fisso
piu' alto per ogni miss, costo *marginale* del miss successivo piu' basso: alza
l'intercetta e abbassa la pendenza.

**La previsione era**: senza `PRETOUCH`, il tempo di *lettura* per miss deve essere
piu' alto, e la differenza deve valere circa quanto, con `PRETOUCH`, valgono le
*pagine nuove*. **Misurata, regge:**

| budget | lettura/miss senza | con | differenza | pagine/miss con | recuperato |
|---|---|---|---|---|---|
| 4,0 GiB | 1,233 ms | 0,932 | 0,301 | 0,370 ms | 81% |
| 1,5 GiB | 1,063 ms | 0,773 | 0,290 | 0,395 ms | 73% |

L'azzeramento e' **lavoro spostato, non aggiunto**: il `memset` toglie 0,29-0,30 ms
da dentro il DMA e li ripaga 0,37-0,40 fuori. Costa il 23-37% in piu' farlo
esplicitamente invece di lasciarlo al gestore di page fault dentro la lettura, ed e'
tutto il motivo per cui `PRETOUCH` e' piu' lento (5,90 contro 6,14 tok/s a 4 GiB,
3,32 contro 3,61 a 1,5 GiB). Ipotesi confermata.

**E sull'esponente mi ero sbagliato di nuovo, per la ragione che avevo appena
costruito lo strumento per evitare.** Avevo scritto che togliere `PRETOUCH` rende la
curva piu' ripida, da 0,47 a 0,70. Quei due numeri venivano da un'escursione di
1,33x, che lo strumento stesso dichiarava insufficiente per un verdetto — e io ne
ho tratto uno in prosa. Rimisurato a 2,67x:

| | senza PRETOUCH | con |
|---|---|---|
| escursione 1,33x | 0,70 ± 0,20 | 0,47 ± 0,15 |
| escursione 2,67x | **0,54 ± 0,03** | **0,58 ± 0,06** |

A escursione adeguata `PRETOUCH` **non cambia la pendenza**: alza l'intercetta di
circa il 4% e basta. La "direzione invertita" era un artefatto del campione corto.
Avevo costruito il meccanismo che sospende i verdetti sugli intervalli stretti, e
poi ne ho pronunciato uno su quei dati nel testo, dove nessun controllo arriva.

I tre numeri servono gia': il lato C++ stampa `per miss: X ms di lettura, Y ms di
pagine nuove, Z ms di commit`. Mancava che qualcuno li leggesse — lo sweep non
chiamava nemmeno il resoconto della cache, quindi `--expert-stats` non produceva
nulla di visibile. Ora li estrae, li mette in tabella accanto alla curva e dice
quale voce cresce al calare del budget:

```
      budget  lettura/miss  pagine/miss  commit/miss
     4.0 GiB        1.337m       0.212m       0.370m
     1.5 GiB        1.684m       0.219m       0.371m
  cresce al calare del budget: lettura 1.337->1.684 ms (1.26x)
```

Se cresce la **lettura** il costo e' nel dispositivo; se crescono le **pagine
nuove** e' nel sistema di memoria; se non cresce niente, la velocita' scende solo
perche' i miss aumentano. Sono tre diagnosi diverse che portano a tre lavori
diversi — e la risposta vera e' una quarta, che non avevo previsto.

### Il costo per miss migliora quando il budget si stringe

Non cresce: **scende**. Da 4 a 1,5 GiB la lettura per miss passa da 1,233 a 1,063 ms
(0,86x), e con `PRETOUCH` da 0,932 a 0,773 (0,83x). Il motivo e' nella riga accanto:
la **profondita' di coda** sale da 2,09 a 2,49 e la banda per singola lettura da 739
a 858 MB/s (+16%). Piu' miss significa piu' letture in volo, e un NVMe rende in
funzione di quante richieste ha contemporaneamente.

La decomposizione, su 2,67x di residenza in meno:

| grandezza | fattore | esponente |
|---|---|---|
| numero di miss | ×2,20 | 0,80 |
| lettura per miss | ×0,86 | **−0,15** (migliora) |
| tempo di lettura totale | ×1,90 | 0,65 |
| velocita' di generazione | ×1,70 | **0,54** |

**La pendenza e' tutta nel numero di miss, e il dispositivo ne restituisce una
parte.** Senza l'effetto della coda, a 1,5 GiB si starebbe a 2,79 tok/s invece di
3,61: quell'effetto vale **+29%**, ed e' la ragione per cui l'esponente e' 0,54 e non
0,80.

Per la scalabilita' e' una buona notizia con una scadenza: la coda cresce da sola al
crescere del modello, quindi la leva continua a dare. Quanto margine resta e' nella
sezione seguente — e la prima risposta che mi ero dato, "circa 2x", era un confronto
fra cose diverse.

### Il margine vero, e perche' il primo numero era sbagliato

Avevo confrontato i nostri 2133 MB/s aggregati con i 4260 che `diskbench` misura **a
32 thread sul pattern `direct`**, cioe' senza la barriera per layer e a una
concorrenza che non raggiungiamo mai. Il confronto giusto e' con lo stesso pattern
(`moe`) alla stessa profondita' di coda.

Misura dedicata, slice da 896 KiB e 2,75 MiB per sei livelli di parallelismo:

| thread | 896 KiB | 2,75 MiB |
|---|---|---|
| 1 | 1338 | 2351 |
| 2 | 2335 | 3087 |
| 3 | 2777 | 3465 |
| 4 | 3069 | 3724 |
| 6 | 3398 | 3996 |
| 8 | 3563 | 4070 |

Alla nostra coda di 2,49 con letture da 896 KiB il dispositivo da' **2552 MB/s**, e
noi facciamo 2133: siamo all'**84%**, e il 16% che manca e' la barriera fra i thread,
che `diskbench` stesso quantifica (il pattern `moe` rende l'87% di `direct`). Non c'e'
un 2x da recuperare al punto in cui siamo: siamo quasi al massimo **per la
dimensione di lettura che usiamo**.

### Lo store legge 896 KiB, e il formato permetterebbe 2,75 MiB

Byte letti diviso miss: **934 KiB**, identico ai due budget. Ogni miss legge *una*
matrice, non la tripletta — mentre il formato PSXS tiene `gate`, `up` e `down` dello
stesso expert contigui proprio per poterli leggere in un colpo. La contiguita' c'e',
il lettore non la usa.

Da qui due strade, e la tabella dice quale:

| | effetto sulla coda | MB/s | |
|---|---|---|---|
| **A** unire le tre matrici in una lettura da 2,75 MiB | 2,49 → ~0,83 | 2351 | |
| **B** mandare le tre letture da 896 KiB in volo insieme | 2,49 → ~7,5 | **3522** | 1,50x meglio di A |

**B vince, e di molto.** Il motivo e' che le due leve non si sommano: a byte totali
fissi per nodo, letture piu' grandi sono anche *meno numerose*, quindi alzare la
dimensione abbassa la coda. Averle presentate come indipendenti e' stato il terzo
difetto di quel verdetto, e ora `diskbench` lo dice.

Guadagno atteso di **B** al nostro punto: **1,38x**, e a budget basso il tempo di
parete e' al 100% letture, quindi si trasferisce quasi uno a uno: 3,61 tok/s
diventerebbero circa **5,0**. Nel regime che decide i modelli grandi.

La strada **A** sarebbe comunque bloccata: le tre matrici sono contigue nel *file* ma
le loro destinazioni sono tre tensori ggml a indirizzi non correlati, lo stesso
vincolo che ha reso impossibile la lista di slab riutilizzabili. Servirebbe I/O
scatter (`ReadFileScatter`) con segmenti allineati alla pagina — fattibile, dato che
le slice sono allineate a 4096, ma e' un lavoro a parte e **B** rende piu' di **A**
da sola.

**E questa e' la riabilitazione di un'ipotesi che avevo dichiarato smentita.** "La
coda di lettura e' bassa perche' i thread sono pochi" era stata chiusa con "0,75 miss
per nodo: i miss sono rari, non serializzati". Ai budget che contano i miss sono 2,30
e 4,99 per nodo, e le letture *sono* serializzate: ogni thread aspetta la propria
invece di lasciarle volare insieme. La smentita era giusta nel regime in cui l'avevo
misurata e sbagliata in quello che conta.

### Letture asincrone: un no, un si', e uno strumento rotto da me

Scritte: `psx_cache_ensure_many` prenota n slice, emette tutte le letture insieme su
un handle aperto con `FILE_FLAG_OVERLAPPED`, attende e poi annuncia. Il collaudo C++
gira senza llama.cpp, quindi la correttezza si verifica prima di consegnare — e
reintroducendo il bug storico dell'annuncio anticipato conta 435 letture su memoria
non pronta su 480, cioe' prende davvero.

**Il primo tentativo, blocchi da 4, non ha dato niente.** Velocita' +0,5% e +3,0%,
dentro il rumore misurato (1,1-4,0%). Il motivo e' nel commento che il codice aveva
gia': su 45 slice per nodo, blocchi da 4 danno **12 lotti per 12 thread**, cioe' zero
margine per il bilanciamento dinamico. Con hit che costano zero e miss un
millisecondo, la distribuzione dinamica degenera in statica. Misurato a 1,5 GiB:
attesa fra thread 2,26x, lock 3,76x, decommit 1,76x, parete +8%.

**E la misura non poteva dirlo, perche' avevo rotto lo strumento.** `ns_read` si
accumulava una volta per slice; con le letture sovrapposte si accumula una volta per
blocco, cioe' il *massimo* delle letture e non la somma. Quindi "lettura per miss" in
calo e "profondita' di coda" in calo erano artefatti: avevo cambiato il codice e il
significato della metrica nello stesso passo. Ora lo store conta blocchi e letture, e
la coda stampata e' `(ns_read/parete) x (letture/blocco)`, con una riga che dichiara
la semantica.

**Blocchi da 2: +12,3% dove conta.**

| | base (sincrono) | blocco 4 | blocco 2 |
|---|---|---|---|
| 4 GiB (24%) | 6,09 tok/s | 6,12 (+0,5%) | **6,25** (+2,6%) |
| 1,5 GiB (9%) | 3,33 tok/s | 3,43 (+3,0%) | **3,74** (+12,3%) |
| coda a 1,5 GiB | 2,49 | n/d (metrica rotta) | **3,49** |
| attesa fra thread | 37 543 ms | 84 670 (2,26x) | 51 753 (1,38x) |
| parete | 120 091 ms | 130 067 (+8%) | 120 170 (+0,1%) |
| esponente | 0,62 | 0,59 | **0,52** |

A 2 i lotti sono 23 per 12 thread: il margine c'e', la coda sale davvero, e i costi
restano dove erano. A 4 GiB il guadagno e' dentro il rumore, ed e' coerente col
meccanismo: con il 91% di hit non ci sono abbastanza miss da sovrapporre.

L'esponente scende da 0,62 a **0,52**: un modello con 2,6 volte gli expert costa 1,65
volte invece di 1,80, e la proiezione per un MoE di classe 80B passa da ~5,3 a
**~5,9 tok/s**. Il default era 2, regolabile con `ARGINE_EXPERT_BATCH`, perche' il
numero giusto dipende da quante slice ha un nodo e da quanti thread girano.

> Questa versione e' stata poi **sostituita** dall'emetti-e-continua, che non ha una
> costante da scegliere, e la variabile non esiste piu'. Rimisurata con un binario
> solo, non vale niente: vedi
> [Tre strati di lettura](#tre-strati-di-lettura-un-binario-solo).

**Il blocco da 2, confermato su quattro punti**, dava +1,1% / +5,2% / +3,6% / +9,6%
ed esponente 0,53. Riprodotto su due esecuzioni (1,5% e 2,4% di scarto). Ma la riga
"letture per blocco" diceva **1,30 a 4 GiB e 1,48 a 1,5 GiB** su un massimo possibile
di 2,00: con l'80-91% di hit, due slice ne contengono poco piu' di una da leggere.

### Il terzo tentativo: emetti e continua

**Ingrossare il blocco non poteva funzionare, e l'aritmetica lo diceva prima di
misurare.** Per raccogliere piu' miss serve consumare piu' slice per lotto; ma su 45
slice e 12 thread i lotti sono 45 / blocco, e il margine per il bilanciamento
dinamico e' lotti / thread:

| blocco | lotti | margine |
|---|---|---|
| 1 | 45 | 3,75 |
| 2 | 23 | 1,92 |
| 4 | 12 | 1,00 ← misurato peggiore |

Le due cose sono legate, quindi 2 era il massimo sotto quel vincolo. **Il passo che
avevo scritto in questo stesso README — "raccogliere i miss su una finestra piu'
larga" — era irraggiungibile cosi'.**

Si slegano notando *quando* i byte servono: non al ritorno di `ensure`, ma alla
barriera prima del calcolo. Quindi il thread prende **una slice alla volta** —
distribuzione fine come il blocco 1 — emette la lettura, continua, e aspetta tutte le
sue letture insieme alla fine. La profondita' di coda diventa pari ai miss che quel
thread incontra, non a una costante scelta a priori: `ARGINE_EXPERT_BATCH` non esiste
piu' perche' non c'e' un numero da indovinare.

Misurato: **letture in volo da 2,36 a 3,71**, contro 1,30-1,48 del blocco.

Sulla velocita' la prima misura diceva +7,1% / +7,0% / +10,7% / +11,1% sul sincrono, e
**era sbagliata**: quei tre numeri venivano da tre binari diversi. Rimisurata con un
binario solo, il guadagno va da **−0,6% al budget piu' alto a +10,8% al piu' basso**,
e il blocco da 2 non vale niente. La tabella completa, e perche' la prima era
sbagliata, in [Tre strati di lettura](#tre-strati-di-lettura-un-binario-solo).

**E un bug mio, nel punto che avevo dichiarato non verificabile.** Avevo scritto, prima
di cominciare, che l'I/O asincrono e' Windows-only e che qui potevo solo compilarlo. Il
bug e' finito li' dentro: `ReadFile` riceve il puntatore a una struttura `OVERLAPPED`
che il kernel tiene per tutta la durata della lettura, e quella struttura viveva dentro
un `std::vector`. Al primo ingrandimento gli elementi si spostavano e tutte le letture
in volo restavano con puntatori pendenti. Dal vivo si vedeva come *"impossibile rendere
residenti gli expert"* con **21 MiB usati su un budget di 2048** — un messaggio che
accusava la memoria mentre il problema era un indirizzo.

Corretto con un `std::deque`, che garantisce indirizzi stabili in coda, piu' un
`static_assert` che impedisce il ritorno a un vector. Il collaudo C++ su Linux non
poteva vederlo, perche' la' le letture sono sincrone: la difesa vera e' il controllo a
tempo di compilazione, che vale su entrambe le piattaforme.

### E cosi' ho sbagliato due volte lo stesso verdetto

La misura qui sopra ha smentito una cosa che avevo scritto **lo stesso giorno**,
poche ore prima, correggendo `diskbench`. Avevo inchiodato nel codice una costante —
0,75 miss per nodo — e concluso che la profondita' di coda reale e' ~1, quindi la
leva del parallelismo non e' disponibile. Quella costante veniva da una misura al 49%
di residenza. Ai budget che contano per un modello piu' grande vale **2,30 miss per
nodo a 4 GiB e 4,99 a 1,5 GiB**, con coda 2,09 e 2,49: tre e sei volte tanto.

Avevo sostituito un verdetto tarato sul regime sbagliato con un verdetto tarato su
un altro regime sbagliato, nella stessa funzione, con un test a proteggerlo.

La correzione non e' un numero migliore: e' togliere il numero. **Il regime non e'
una costante del programma, e' un dato della misura.** Ora `diskbench` accetta
`--coda N` — il valore che lo store stampa con `--expert-stats` — e senza quel dato
non pronuncia il verdetto su come leggere, invece di inventarsi il regime in cui si
trova chi lo esegue.

La lezione che resta e' sulla forma dell'errore, non sul numero: avevo dedotto la
direzione da un ragionamento sul costo per miss, e il ragionamento sembrava
solido. Era sbagliato, e l'unico modo di saperlo e' stato rifare la misura.

### `argine experts report|simulate modello.gguf`

Analisi della expert cache per i modelli MoE (vedi piu' avanti).

### `argine store build|info|verify|bench modello.gguf`

L'**expert store**: un file `.psxs` che contiene gli stessi pesi degli expert del
GGUF, riorganizzati per essere letti invece che mappati. Serve al buffer type C++
descritto piu' avanti.

```bat
argine store build Qwen3-30B-A3B-Q4_K_M.gguf
argine store verify Qwen3-30B-A3B-Q4_K_M.gguf
```

`verify` confronta i byte dello store con quelli del GGUF uno per uno su un
campione, piu' il controllo degli allineamenti. Un byte sbagliato non fa rumore:
degrada il modello in silenzio.

`bench` misura quattro percorsi di lettura sugli stessi dati e rifiuta di
confrontare percorsi con esposizione diversa alla cache dell'OS — un GGUF letto
con `read()` bufferizzato sembra piu' veloce di uno store letto scavalcando la
cache, e il confronto non significherebbe niente.

### `argine diskbench FILE`

Misura il tetto dello storage sul pattern di accesso vero degli expert: una
barriera fra i thread a ogni ondata, come nel calcolo reale. Esiste per
rispondere a una domanda che decide dove va il lavoro — i MB/s che si vedono in
generazione sono il disco o il percorso software?

Sulla macchina di riferimento (NVMe): 4260-4470 MB/s con il pattern a ondate,
contro i 1139 MB/s raggiunti in produzione. La risposta e' "il percorso
software", e cambia dove ha senso ottimizzare.

#### Le due leve, e quale si puo' tirare davvero

Sul *come* leggere, `diskbench` concludeva: *"la profondita' di coda e' la leva,
quindi conviene emettere piu' letture insieme piuttosto che letture piu'
grandi"*. La misura dietro quella frase era corretta — da 1 a 32 thread il
dispositivo scala di 4,2x — e il consiglio era inservibile, perche' descriveva un
regime in cui lo store non si trova mai.

A profondita' di coda 1 non c'e' niente da sovrapporre, e passare da 896 KiB (un
tensore expert) a 2,75 MiB (la tripletta) rende il **76% in piu'** (1338 contro 2351 MB/s). E' la ragione
per cui il formato PSXS tiene contigue le tre matrici di un expert: decisione presa
*contro* il consiglio del nostro stesso strumento.

Il primo tentativo di correzione ha sbagliato di nuovo, inchiodando nel codice una
costante (0,75 miss per nodo, misurata al 49% di residenza) per dedurne che la coda
e' sempre ~1. Non lo e': ai budget bassi vale 2,09-2,49. Il racconto per intero e' in
[E cosi' ho sbagliato due volte lo stesso verdetto](#e-cosi-ho-sbagliato-due-volte-lo-stesso-verdetto).

Adesso `diskbench` misura entrambe le leve e **prende il regime da fuori**:
`--coda N`, col numero che lo store stampa con `--expert-stats`. Senza quel dato non
pronuncia il verdetto. Entrambi i consigli sbagliati hanno un test che impedisce loro
di ricomparire.

Due lezioni, e la seconda l'ho imparata solo al secondo errore: **una misura corretta
non e' ancora una raccomandazione** — fra le due c'e' la domanda "in quale regime si
lavora?"; e quella domanda non si risponde con una costante nel codice, perche' una
costante e' solo un regime scelto una volta e dimenticato.

### `argine ppl modello.gguf`

Perplessita': quanto il modello e' sorpreso da un testo. E' il numero che da' un
prezzo alla quantizzazione.

```bat
argine ppl modello.gguf --text testo-riferimento.txt --ctx 512 --chunks 60
```

Ogni misura porta con se' una **firma** (`ctx;blocchi;impronta del testo`) e il
confronto si rifiuta se non coincide: due perplessita' ottenute con parametri
diversi non si possono mettere nella stessa frase. Il riferimento e' il migliore
fra i modelli gia' misurati, non il piu' recente, e `--riferimento NOME` lo
forza.

Il testo conta piu' di quanto sembri. Vedi
[Come un testo sbagliato ha prodotto il risultato sbagliato](#come-un-testo-sbagliato-ha-prodotto-il-risultato-sbagliato).

### `argine quantize modello.gguf` e `argine imatrix modello.gguf`

Ci sono due modi di far stare un modello MoE in 16 GB di RAM: leggere gli expert
da disco mentre servono (l'expert store) oppure rimpicciolirli. Il primo e'
implementato e misurato; questo comando prova il secondo.

Il conto che decide tutto si fa senza produrre niente:

    argine quantize Qwen3-30B-A3B-Q4_K_M.gguf --experts IQ2_XXS --dry-run

    == COSA USCIRA' -------------------------------------------------
      expert -> IQ2_XXS        16.35 GiB -> 7.49 GiB  (-8.86 GiB)
      attenzione/altro         498.38 MiB -> 498.38 MiB
      embedding/output         410.35 MiB -> 410.35 MiB
      norm/router              48.80 MiB -> 48.80 MiB
      file totale              18.62 GiB -> 9.42 GiB
      bit per peso             4.54 -> 2.29

Se il file non ci sta nel budget, il comando lo dice adesso invece che dopo
un'ora di calcolo, e propone il tipo piu' grande che ci starebbe — il piu'
grande, non il piu' piccolo: ogni bit in meno costa qualita' e scendere sotto
il necessario la regala senza comprare niente. Sotto i 2 bit non propone
niente: un modello che entra in RAM e risponde a caso non ha risolto il
problema.

Solo gli expert cambiano tipo. Il resto (attenzione, embedding, router) resta
dov'e': sono il **5,4%** dei byte (957,53 MiB su 17,35 GiB) e non c'e' niente da
guadagnare a rovinarli. Il
dirottamento usa `--tensor-type` di `llama-quantize`, e i tre pattern sono
`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`.

**I tipi IQ2 non esistono senza matrice di importanza.** Non e' una
raccomandazione di qualita': `llama-quantize` carica il modello, arriva al primo
tensore e si interrompe. Argine lo verifica prima di avviare il processo, e la
matrice si calcola con:

    argine imatrix modello.gguf --chunks 128 --expert-store

La matrice misura quanto ogni colonna contribuisce all'uscita su un testo di
calibrazione, e la quantizzazione usa quel numero per spendere la precisione
dove serve. Su un MoE con 128 expert per layer servono molti blocchi, perche'
un expert mai attivato resta senza dati. L'expert store funziona anche qui: il
calcolo sta nel budget di RAM invece di affidarsi alla memoria mappata.

L'alternativa senza matrice e' `--experts Q2_K` (2,62 bit): produce un file piu'
grande e di qualita' peggiore a parita' di bit, ma non richiede un'ora di
calcolo preliminare.

Partire da un GGUF gia' quantizzato richiede `--allow-requantize` e arrotonda i
pesi una seconda volta: il risultato e' peggiore dello stesso tipo costruito dal
modello originale in BF16. Va bene per capire se la strada regge, non come file
definitivo.

Il numero che chiude il discorso e' `argine ppl` sul file prodotto, con gli
stessi identici parametri della misura di riferimento.

### `argine logs` / `argine doctor` / `argine setup`

`logs` mostra lo storico JSONL delle esecuzioni, `setup` scarica o aggiorna i
runtime, `doctor` verifica l'installazione.

Di `doctor` conta soprattutto una riga, perche' risponde alla domanda che non si
puo' fare a `--help`:

```
  ARGINE_LLAMA_DIR         C:\Users\...\Argine\llama-noomp
  binario in uso           llama-noomp@78ee4096 -> ...\llama-server.exe
  expert store nel binario presente
```

L'expert store si configura per variabili d'ambiente, quindi non compare fra le
opzioni, e **una build ufficiale le ignora in silenzio**: `--expert-store`
sembrerebbe attivo e non lo sarebbe. `doctor` cerca il marcatore dentro
l'eseguibile e lo dice. Riporta anche l'impronta della build (`@78ee4096`), che e'
l'unico modo per accorgersi di aver confrontato due misure prese su binari
diversi.

---

## API locale: due dialetti

`argine run` espone su `http://127.0.0.1:8080`:

```
POST /v1/chat/completions       OpenAI (anche in streaming SSE)
POST /v1/completions           OpenAI
POST /v1/messages              Anthropic (anche in streaming SSE)
GET  /v1/models
POST /completion                endpoint nativo llama.cpp
GET  /health
```

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"local","messages":[{"role":"user","content":"Ciao"}]}'
```

Da Python, con qualunque client OpenAI:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="non-serve")
print(client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Ciao"}],
).choices[0].message.content)
```

### `/v1/messages`: il dialetto Anthropic

```bash
curl http://127.0.0.1:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"local","max_tokens":256,
       "system":"Rispondi in una frase.",
       "messages":[{"role":"user","content":"Ciao"}]}'
```

```python
from anthropic import Anthropic
client = Anthropic(base_url="http://127.0.0.1:8080", api_key="non-serve")
print(client.messages.create(
    model="local", max_tokens=256,
    messages=[{"role": "user", "content": "Ciao"}],
).content[0].text)
```

Il backend parla solo OpenAI, quindi questo endpoint traduce in entrambe le
direzioni. Le differenze che contano, e come sono risolte:

| | Anthropic | OpenAI | come |
|---|---|---|---|
| istruzione di sistema | campo `system` in cima | messaggio con `role: system` | diventa il primo messaggio |
| contenuto | stringa o lista di blocchi | stringa | i blocchi `text` si appiattiscono |
| motivo di fine | `stop_reason` | `finish_reason` | tabella di conversione |
| sequenza incontrata | `stop_sequence` la nomina | non la distingue da `stop` | si ricava dal testo finale |
| conteggi | `input_tokens` / `output_tokens` | `prompt_tokens` / `completion_tokens` | rinominati |
| streaming | eventi tipizzati e nominati | chunk tutti uguali + `[DONE]` | sequenza completa ricostruita |
| `max_tokens` | obbligatorio | opzionale | manca → errore, non un default |

**Due scelte che vale la pena dire, perche' sono rifiuti e non traduzioni.**

Un blocco `image` o `document` viene **rifiutato** con un errore che spiega
perche': questo runtime serve modelli testuali su CPU, e accettare l'immagine per
poi scartarla darebbe una risposta plausibile a meta' della domanda. Lo stesso per
`max_tokens` mancante: inventare un default produrrebbe una risposta troncata senza
che il client sappia da dove viene il troncamento.

Gli errori escono nella forma Anthropic (`{"type":"error","error":{...}}`), compresi
quelli che arrivano dal backend: tradurre tutto tranne la parte che si legge quando
le cose vanno male e' il modo piu' sicuro di far sembrare l'integrazione rotta.

**La traduzione sta in `argine/anthropic_api.py` e sono funzioni pure**, da
dizionario a dizionario: nessun I/O, nessuno stato, nessuna piattaforma. E' una
scelta presa dopo un bug nell'expert store, dove il collaudo girava su una
piattaforma diversa da quella di destinazione e un puntatore pendente ci e' passato
in mezzo. Qui non c'e' una piattaforma di destinazione: c'e' il JSON, e i 13 test
lo coprono — compresi il flusso troncato a meta', quello vuoto, e le righe SSE
spezzate in mezzo a un chunk.

In piu' Argine aggiunge endpoint propri:

| endpoint | contenuto |
|---|---|
| `/argine/status` | stato, build llama.cpp, modello, RAM misurata in tempo reale |
| `/argine/plan` | piano di memoria e configurazione runtime |
| `/argine/memory` | budget, stima per componente, misura reale |
| `/argine/experts` | telemetria della expert cache |

---

## Gestione della RAM

Il budget e' calcolato riservando al sistema il 25% della RAM (minimo 3,5 GiB,
massimo 6 GiB): su 16 GB restano circa **12 GiB** per Argine. Con
`--ram-limit` lo imposti a mano; il valore viene comunque limitato al 90% della
RAM fisica per non mandare in swap il sistema.

Il Memory Planner contabilizza separatamente:

- **model weights** residenti (per i MoE: solo i tensori *shared*);
- **expert cache** (MoE);
- **KV cache**, calcolata analiticamente da layer, teste KV, dimensione delle
  teste, context e precisione;
- **activations** e buffer di calcolo, in funzione di ubatch e context;
- **runtime buffers** del processo;
- **GPU shared memory**: su una iGPU come la Iris Xe non e' memoria in piu', e'
  la stessa RAM di sistema. Argine la conta una volta sola e limita l'offload
  in modo che non superi il tetto di shared memory del driver (~meta' RAM).
- **cache dei prompt**, cioe' gli stati KV che `llama-server` conserva per
  riusare il prefisso dei turni precedenti.

### La cache dei prompt, che nessuno contava

La voce precedente e' l'ultima arrivata, e il motivo per cui e' arrivata e' che
mancava. `llama-server` tiene in RAM gli stati KV delle conversazioni passate per
poterne riusare il prefisso, e il tetto predefinito e' `--cache-ram 8192`, cioe'
**8 GiB**. Su una macchina da 16 GB e' meta' della memoria, non contabilizzata da
nessuna parte: il piano dichiarava di starci dentro, il server partiva, e poi la
RAM se ne andava in un posto che il piano non conosceva. Non si e' manifestato
come un crash ma come qualcosa di peggio — un piano che diceva il vero su tutte
le voci che misurava, e taceva su quella piu' grande.

Adesso la cache dei prompt e' una voce del budget come le altre: Argine la
calcola (`kv_cache_bytes() * stati`, default 2 stati), la somma al totale, e la
passa **sempre** esplicitamente a `llama-server` come `--cache-ram`, in MiB.
Esplicitamente anche quando il valore coincide col default, perche' un default e'
un numero che puo' cambiare sotto di noi in un aggiornamento di llama.cpp, e il
piano smetterebbe di descrivere la realta' senza che nessuno tocchi niente.

Con `--prompt-cache-states N` si cambia il numero di stati; `0` spegne il riuso
del prefisso e libera tutto. Vale la pena saperlo prima di metterlo a zero: su
una conversazione di sei turni il riuso del prefisso tiene il tempo al primo
token piatto intorno ai 5 secondi contro 40,12 senza, cioe' **7,8 volte**, vedi
[Una conversazione che cresce](#una-conversazione-che-cresce).

### Il buffer di repack

llama.cpp riscrive i pesi quantizzati in un layout interleaved per le istruzioni
SIMD, e quella copia e' un'allocazione **in aggiunta** al file mappato. Su un
modello Q4 vale circa due terzi dei pesi: ignorarla significa sottostimare la
RAM del 35-40%. Argine la calcola esattamente dalla tabella dei tensori
(tipi ripacchettabili, esclusi gli embedding e i tensori 1D). Verifica su
Qwen3-8B-Q4_K_M: 3199 MiB calcolati contro 3199,50 MiB dichiarati da llama.cpp.

**Per default Argine lo disattiva.** Misura su i5-1340P (AVX2, senza AVX-512
ne' AMX), Qwen3-8B-Q4_K_M, 8 thread, 3 ripetizioni:

| | con repack | senza repack |
|---|---|---|
| generazione sostenuta | 7,05 tok/s | **8,24 tok/s** |
| generazione a freddo | 7,64 tok/s | 8,84 tok/s |
| RAM peak | 8,16 GiB | **5,04 GiB** |
| RAM libera minima | 737 MiB | 3,93 GiB |

Non e' uno scambio fra RAM e velocita': su questa classe di CPU il repack e'
peggiore su entrambi i fronti, verosimilmente perche' i kernel che sfrutterebbero
quel layout richiedono estensioni assenti, mentre il costo in traffico di memoria
resta in un carico gia' limitato dalla banda. Su un processore con AMX il
risultato puo' essere diverso: `--repack` lo riattiva per rimisurarlo.

Su un MoE l'effetto e' drastico: con gli expert su CPU il repack duplicherebbe
l'intero set di expert, diverse decine di GiB su Qwen3-30B-A3B.

### Scala di sicurezza

Se la stima supera il budget, Argine **non avvia** il modello: prova prima a
rientrare, nell'ordine previsto dalla specifica:

```
disattivazione repack -> cache dei prompt a 1 stato -> cache dei prompt a 0
-> riduzione context -> riduzione batch
-> KV Q8 -> KV Q4 -> riduzione GPU offload
-> lazy loading / expert streaming -> errore
```

Il primo gradino non era nella specifica, che non poteva sapere del buffer di
repack. Viene prima di tutti perche' libera piu' RAM di qualsiasi altra
riduzione e, sulle CPU misurate, senza costi.

Il secondo e il terzo non c'erano nemmeno nel codice, finche' non si e' scoperto
che la cache dei prompt mangiava 8 GiB non contati. Vengono subito dopo il
repack, e prima del context, perche' il context decide *cosa* il modello riesce a
fare e la cache dei prompt solo *quanto in fretta* lo rifa': fra ricordare meno
turni e non poter leggere un documento intero, si rinuncia ai turni. La discesa
passa per **1 stato** prima di arrivare a 0, perche' con uno la conversazione in
corso resta coperta — e' il caso che conta — mentre si libera comunque meta'
della voce.

**Con l'expert store quel gradino sparisce.** In llama.cpp `--no-repack` non
spegne solo il repack: imposta `no_extra_bufts`, che disabilita tutti gli extra
buffer type, lo store compreso. Il risultato sarebbe uno store aperto a cui
nessun tensore viene assegnato — attivo all'apparenza e inutile nei fatti.
Argine lo rileva e tiene il repack acceso, che con lo store costa 229,50 MiB
invece dei due terzi dei pesi, perche' gli expert appartengono a noi e al repack
restano i soli tensori shared.

**Anche la scelta dei thread entra nel piano**, e sulle CPU ibride non e' ovvia:
un thread per P-core, non due. Vale +40-46%, vedi
[Da dove viene la velocita'](#da-dove-viene-la-velocita).

Su un MoE c'e' un passaggio in piu'. Una volta attivato lo streaming degli
expert si libera molta memoria, e restare con `--ctx 512` non avrebbe senso:
Argine esegue allora una **fase di relax** che riporta context, batch,
precisione KV e offload al valore migliore che ci sta ancora nel budget, e
assegna alla expert cache tutto cio' che avanza. Il registro delle decisioni
mostra sia le riduzioni sia il risultato finale.

Se anche cosi' non ci si sta, il modello non parte e viene scritto un record
`run_refused` nel log, con il motivo. Se invece il fallimento avviene al
caricamento (OOM reale del backend), Argine riprova automaticamente con una
configurazione ridotta (`--retries`, default 2).

---

## Modelli MoE ed expert cache

Su un modello come Qwen3-30B-A3B il file GGUF e' grande ma solo una frazione dei
pesi serve a ogni token. Argine separa la contabilita':

- **shared tensors** — attenzione, norm, embedding, output: servono sempre e
  restano residenti (circa 1 GiB su Qwen3-30B-A3B Q4);
- **expert tensors** — `ffn_*_exps`: sono la quasi totalita' del file e vengono
  serviti da una cache con budget rigido, alimentata da SSD.

`argine inspect` e `argine experts report` mostrano quanti expert entrano nel
budget e quanti restano "freddi" su disco:

```bat
argine experts report Qwen3-30B-A3B-Q4_K_M.gguf --cache-budget 5G
```

```
== BUDGET MEMORIA MoE --------------------------------------------
  shared tensors (residenti) 957.53 MiB
  expert cache             5.00 GiB
  KV cache                 384.00 MiB
  runtime                  384.00 MiB
  TOTALE                   6.69 GiB
== RESIDENZA -----------------------------------------------------
  expert totali            128 (8 attivi/token)
  layer MoE                48
  unita' di trasferimento  2.53 MiB
  slot in cache            2022 (42.12/layer)
  copertura hot            32.9%
  cold su SSD              10.19 GiB
```

### Simulazione

`argine experts simulate` fa passare una traccia di token attraverso la cache
**leggendo davvero le slice degli expert dal GGUF su NVMe**, cosi' i numeri di
throughput e hit rate sono misurati, non ipotizzati.

```bat
argine experts simulate Qwen3-30B-A3B-Q4_K_M.gguf --cache-budget 5G --tokens 256 --compute-tok-s 12
```

Parametri utili:

- `--skew` sbilanciamento del routing: `0` = uniforme (caso peggiore),
  `0.9`–`1.2` = realistico (pochi expert molto usati);
- `--policy lru|2q|lfu` politica di eviction;
- `--no-prefetch` per misurare quanto vale il prefetch del layer successivo;
- `--compute-tok-s` i tok/s misurati con `argine benchmark`, per proiettare la
  velocita' finale tenendo conto della sovrapposizione fra calcolo e I/O.

Esempio di confronto ottenuto sul modello di riferimento (fixture da 16 GiB,
30 token, top-8 su 48 layer):

| configurazione | hit rate | byte/token da SSD | tok/s proiettati |
|---|---|---|---|
| cache 5 GiB, prefetch on | 73,3% | 451 MiB | 2,63 |
| cache 5 GiB, prefetch off | 53,6% | 451 MiB | 2,53 |
| cache 9 GiB, prefetch on | 78,8% | 349 MiB | 2,95 |
| cache 5 GiB, routing uniforme | 53,4% | 690 MiB | 1,67 |

Si legge cosi': il prefetch alza molto l'hit rate a parita' di byte letti (la
deduplicazione delle richieste in volo evita letture doppie), mentre il fattore
che domina davvero la velocita' e' quanta parte degli expert caldi entra in
cache. Con routing uniforme il sistema e' irrimediabilmente I/O bound.

**Quei tok/s sono proiezioni, non misure.** Servono a decidere se vale la pena
provare un modello prima di scaricarlo. Le misure vere sulla stessa macchina e
sullo stesso modello stanno in [Risultati misurati](#risultati-misurati), e sono
circa tre volte piu' alte: il simulatore assume che calcolo e I/O si sovrappongano
meno di quanto facciano davvero. Serve a confrontare configurazioni fra loro, non
a predire un numero assoluto.

---

## L'expert store

La simulazione sopra risponde a "quanto sarebbe veloce"; l'**expert store**
risponde a "fallo". E' una patch C++ a llama.cpp che registra un buffer type
aggiuntivo: i tensori degli expert non vengono piu' mappati in memoria, ma letti
da un file con letture esplicite e tenuti in una cache LRU con un tetto deciso da
noi.

### Perche' non bastava mmap

Con `mmap` la residenza la decide il sistema operativo. Misurato: picco di RAM al
**105-111% del budget dichiarato**, RAM libera scesa a 15-260 MiB. Il "limite di
RAM" era un auspicio.

Con lo store: picco al **79-88% del budget**, RAM libera minima 0,7-1,3 GiB. Il
tetto e' rigido perche' la memoria la impegniamo e la rilasciamo noi.

C'e' anche un motivo di velocita'. Misurato con `diskbench`, a parita' di cache
fredda e un solo thread:

| percorso | MB/s |
|---|---|
| page fault su memoria mappata | 310 |
| letture esplicite non bufferizzate | 1324 |

Il percorso dei page fault di Windows costa **4,3 volte** la lettura diretta.

### Il formato PSXS

Intestazione di 256 byte, tabella dei layer, poi `n_layer x n_expert` voci da 36
byte ciascuna: tre offset e tre dimensioni, uno per `ffn_gate`, `ffn_up`,
`ffn_down`. **Ogni slice e' allineata a 4096 byte da sola**, non solo il gruppo:
e' la condizione perche' ogni lettura atterri direttamente nella memoria del suo
tensore, senza staging e senza `memcpy`, e perche' `FILE_FLAG_NO_BUFFERING` la
accetti.

L'indice da' liberta' di disposizione fisica **senza permutare i pesi del
router** — differenza non ovvia rispetto a un riordino del GGUF, dove spostare un
expert obbliga a spostare la riga corrispondente della matrice di routing.

### Come si innesta

`compute_forward` non calcola: legge gli id scelti dal router, garantisce che le
slice di quegli expert siano in memoria, sincronizza i thread e poi ritorna
`false`, cosi' ggml esegue il suo kernel standard sulla memoria appena riempita.
Nessun kernel duplicato da tenere allineato con llama.cpp a ogni rebase.

La patch a llama.cpp e' di **15 righe** (4 nel CMakeLists, 11 in `ggml-cpu.cpp`);
tutto il resto vive in file nostri. Si applica pulita sul master di llama.cpp a
distanza di settimane.

### I bug che e' costato

Tutti di concorrenza, tutti silenziosi:

- **`ensure` tornava prima che i byte fossero arrivati.** Invisibile finche' non
  ho aggiunto un ritardo artificiale di 300 µs sulle letture: 430 letture sporche
  su 480 (il numero esatto cambia a ogni esecuzione: 435 in un'altra), poi 0.
- **Uno stato letto prima di un'attesa e usato dopo.** `cv.wait_for` rilascia il
  mutex, quindi la decisione "la chiave non c'e'" precedeva l'attesa: due
  caricatori per la stessa chiave, un nodo orfano, sfratto di memoria in uso.
  Trovato solo con un anello di eventi, dopo cinque ipotesi sbagliate. Gli hit
  sono passati da 232 a 2232 su 4800.
- **Un handle usato fuori dal lock.** Trovato da ThreadSanitizer.
- **Lo sharding mandava tutte le chiavi al settore 0.** Gli offset sono multipli
  di 4096 e le basi allineate a 64 KiB, quindi `% 16` faceva sempre 0. Sintomo:
  310 MiB residenti su 4982, e la velocita' quasi dimezzata. Risolto mescolando i
  bit alti (fmix64).
- **`--no-repack` spegneva lo store.** In llama.cpp imposta `no_extra_bufts`, che
  disabilita *tutti* gli extra buffer type. Lo store restava aperto senza
  ricevere un tensore.

---

## Risultati misurati

Macchina di riferimento: Windows 11, **Intel Core i5-1340P** (12 core fisici, 16
logici: 4 P-core + 8 E-core), **16 GB di RAM** (15,72 GiB utilizzabili), NVMe,
iGPU Intel Iris Xe e nessuna GPU dedicata.

Fino a poco prima di questa riga il documento diceva "Intel i7 ibrido". Era falso,
e in un documento la cui tesi e' *"dire su quale macchina"* e' il tipo di errore
peggiore: l'ho letto dal log delle esecuzioni (`runs.jsonl`, campo
`hardware.cpu`), che registra il modello esatto da sempre. Nessuno dei numeri
cambia — la macchina e' quella, era il nome a essere sbagliato — ma chiunque
avesse provato a riprodurli su un i7 vero avrebbe confrontato due macchine diverse
credendo di averne una sola.

Modello principale: Qwen3-30B-A3B, 30,5 miliardi di parametri totali e 3,3 attivi,
128 expert per layer di cui 8 usati per token, 48 layer. Modello grande:
Qwen3-Next-80B-A3B, 79,7 miliardi totali e 3,87 attivi, 512 expert di cui 10 usati,
48 layer.

### Le due configurazioni utilizzabili

| | Q4_K_M + store | IQ2_S residente |
|---|---|---|
| perplessita' | **12,06** | 13,63 (+13%) |
| generazione | 9,62 tok/s | **12,97 tok/s** |
| prompt | 23,5 tok/s | **39,4 tok/s** |
| TTFT | 3,31 s | **1,97 s** |
| file su disco | 17,35 GiB | **9,65 GiB** |
| RAM al picco | 7,54 GiB | 7,13 GiB |

Non c'e' un vincitore: la prima tiene la qualita', la seconda e' piu' rapida su
tutti e tre gli assi di velocita'. La scelta dipende da cosa serve.

### Una conversazione che cresce

Ogni misura di velocita' qui sopra usa **prompt indipendenti**. Una chat vera e'
un prefisso che si allunga, ed e' il caso per cui esiste il riuso della KV cache:
misurarlo serve un banco a parte, `benchmark --conversazione N`, che manda turno
1, poi 1+2, poi 1+2+3, con testi diversi fra loro — turni identici renderebbero
il prefisso innaturalmente comprimibile.

Sei turni su Qwen3-30B-A3B IQ2_S, la stessa conversazione due volte:

| turno | token totali | con riuso | senza riuso | rapporto |
|---|---|---|---|---|
| 1 | 188 | 5,03 s | 5,11 s | 1,0x |
| 2 | 444 | 5,62 s | 10,96 s | 1,9x |
| 3 | 697 | 4,78 s | 18,19 s | 3,8x |
| 4 | 950 | 5,05 s | 24,57 s | 4,9x |
| 5 | 1207 | 4,86 s | 32,15 s | 6,6x |
| 6 | 1467 | **5,12 s** | **40,12 s** | **7,8x** |

**Il TTFT resta piatto** mentre il prefisso cresce: al sesto turno llama.cpp
riusa 1293 token e ne elabora 174. Il rapporto cresce con la conversazione e non
ha ancora smesso di crescere a 1467 token.

I ~5 secondi che restano non sono un difetto della cache: sono i ~170 token nuovi
di ogni turno, elaborati a 38 tok/s. Per abbassarli serve un prompt piu' veloce,
che e' un problema diverso.

**Questa misura ha cancellato un progetto.** La cache KV su disco era in cima
alla lista delle cose da costruire, con una stima di "oltre cento volte" sul
TTFT. Quella stima confrontava il disco con *nessuna* cache; il confronto onesto
e' con quella che llama.cpp ha gia', e li' il disco non aggiunge niente al caso
normale. Resterebbe utile solo per riprendere dopo un riavvio e per alternare fra
conversazioni diverse — reali, ma non abbastanza per un formato su disco con
tetto rigido e cache LRU.

### Il controllo: serviva davvero un modello grande?

Il progetto e' nato da un'assunzione mai verificata — *su 16 GB conviene un
modello piu' grande, perche' sa di piu*. Il controllo e' un **Qwen3-14B denso in
Q4_K_M**, intatto, che occupa lo stesso spazio del 30B compresso. Stessa famiglia,
quindi stesso tokenizer, quindi perplessita' confrontabili; stesso testo, stessa
firma.

| a parita' di RAM | Qwen3-14B denso Q4_K_M | Qwen3-30B-A3B a 2,56 bit |
|---|---|---|
| perplessita' | 13,2766 ± 0,3176 | 13,6283 ± 0,3290 |
| generazione | 3,83 tok/s | **12,97 tok/s** |
| prompt | 13,9 tok/s | **39,4 tok/s** |
| TTFT | 5,56 s | **1,97 s** |
| file | 8,38 GiB | 9,65 GiB |

**La qualita' e' la stessa** — 0,35 di differenza contro 0,91 di due sigma, cioe'
indistinguibile — **e la velocita' e' 3,4 volte.**

Quindi l'assunzione era sbagliata nella sua forma originale: il modello grande
compresso **non sa di piu**. Ma la conclusione pratica regge, per un motivo
diverso e migliore:

> A parita' di qualita', il modello grande e sparso costa un terzo del tempo per
> token. Il 14B fa passare ogni token per tutti i 14 miliardi di pesi; il
> 30B-A3B ne attiva 3,3. Si paga in spazio su disco e si incassa in velocita',
> non in conoscenza.

La prima misura aveva il 14B in svantaggio (7,75 GiB liberi per 8,38 GiB di pesi:
paginava, 16,57 GiB letti da disco). Rifatta a macchina sgombra la lettura scende
a 1,38 GiB e **la velocita' non cambia**, 3,81 → 3,83: la lentezza e' calcolo, non
paginazione.

Una nota sul limite della misura. La perplessita' confronta la previsione del
token successivo. Fra due quantizzazioni **dello stesso modello** e' un ottimo
indicatore del danno; fra **due modelli diversi**, come qui, il legame con
ragionamento, istruzioni e uso di strumenti si allenta. Il numero dice che i due
prevedono il testo ugualmente bene, non che siano assistenti equivalenti.

### Da dove viene la velocita'

Partenza: 4,23 tok/s. Due fattori, entrambi indipendenti dall'architettura del
progetto.

**I thread.** Il planner sceglieva `P-core x 2`. Misurato su due quantizzazioni
diverse, tre ripetizioni per punto:

| thread | Q4_K_M + store | IQ2_XXS residente |
|---|---|---|
| **4** | **5,93** | **7,08** |
| 6 | — | 5,19 |
| 8 | 4,23 | 4,84 |
| 12 | — | 4,01 |

Monotona decrescente: +40% e +46% passando da 8 a 4. Non e' un effetto del tipo
di pesi, e' la barriera. ggml sincronizza i thread a ogni nodo del grafo e
aspetta il piu' lento; un E-core, o il fratello SMT di un P-core gia' occupato,
aggiunge un partecipante all'attesa senza aggiungere lavoro fatto. **Un thread
per P-core.**

**Il binario.** La build patchata era compilata male. Stesso modello, stessi
flag, stessi 4 thread:

| build | generazione | CPU media |
|---|---|---|
| vecchia patchata | 7,45 tok/s | 296% |
| llama.cpp ufficiale b11214 | 13,53 tok/s | 462% |
| **nuova patchata** (llama.cpp aggiornato, senza OpenMP, link statico) | **12,97 tok/s** | 470% |

**1,74x** dalla sola ricompilazione. Resta irrisolto se il fattore venga da
OpenMP o dalla versione di llama.cpp: la variante di controllo avrebbe richiesto
tre DLL in piu' sul PC dell'utente e l'ho abbandonata.

Lo stesso salto vale per lo store (5,93 → 9,62, cioe' 1,62x), e il suo throughput
di lettura e' passato da ~432 a **1139 MB/s** senza toccare una riga del codice
di I/O: i thread che emettono le letture sono semplicemente schedulati meglio,
quindi ce ne sono di piu' in volo.

### La curva qualita' / bit

Expert quantizzati con matrice di importanza calibrata su prosa reale, misurati
tutti con la stessa firma (`ctx=512`, 60 blocchi):

| expert | bit/peso | file | perplessita' | vs Q4 | pendenza locale |
|---|---|---|---|---|---|
| Q4_K_M | 4,88 | 17,35 GiB | 12,06 | — | |
| IQ2_S | 2,56 | 9,65 GiB | 13,63 | +12,96% | −0,68 ppl/bit ↑ |
| IQ2_XS | 2,31 | 8,80 GiB | 14,36 | +19,01% | −2,92 ppl/bit |
| IQ2_XXS | 2,06 | 7,90 GiB | 15,19 | +25,95% | −3,32 ppl/bit |

**2,56 bit e' il ginocchio.** Sopra, la curva e' quasi piatta e i bit in piu'
rendono poco; sotto, e' quattro volte piu' ripida. Il gradino successivo che
migliorerebbe la qualita' (IQ3_XXS, 3,06 bit) uscirebbe a 11,34 GiB e non
lascerebbe spazio per KV cache e contesto: non e' una scelta, e' un muro.

### Come un testo sbagliato ha prodotto il risultato sbagliato

Il primo confronto Q4 contro IQ2_XXS diceva **+2,06%, non distinguibile dal
rumore**. Il confronto giusto dice **+36%**.

Due errori, entrambi nello strumento di misura:

1. **Il testo di prova era troppo facile.** Frasi composte da 15 soggetti, 13
   verbi, 11 oggetti e 8 code: tutte distinte, ma con lo stesso schema. Dopo tre
   frasi il modello indovina la struttura, e una quantizzazione che rompe il
   ragionamento difficile non si vede. La perplessita' usciva 3,43 contro i 12,06
   su prosa vera — lo stesso modello, un metro diverso.
2. **La matrice di importanza era calibrata sulla stessa distribuzione.** Aveva
   imparato quali pesi servono per quelle frasi, e la misura chiedeva esattamente
   quelle frasi. Calibrare sul set di test.

Correggendo solo il testo di calibrazione — stessa procedura, stesso tipo, stesso
numero di bit — il degrado e' sceso da **+36,3% a +26,0%**. Dieci punti dalla
sola scelta del corpus.

Il progetto distribuisce due testi disgiunti, `testo-calibrazione.txt` (Stoker,
Dickens, Thoreau, Twain, Wells, Carroll) e `testo-riferimento.txt` (Melville,
Austen, Conan Doyle, Federalist Papers), verificati come privi di frasi in
comune. Non sono wikitext-2-raw, quindi i numeri **non** sono confrontabili con
quelli pubblicati; servono a confrontare due quantizzazioni fra loro, che e' cio'
che serve qui.

---

## Ipotesi smentite

Ognuna sembrava ovvia, ognuna e' costata tempo, ognuna e' caduta su una misura.
Stanno qui perche' sono il risultato piu' riutilizzabile del progetto.

| ipotesi | misura | esito |
|---|---|---|
| Il collo di bottiglia e' la banda del disco | tetto del dispositivo 4260-4470 MB/s sul pattern reale, in produzione 562 **allora** (un solo thread leggeva); 1139 dopo la correzione, 2133 con le letture asincrone | smentita |
| I page fault valgono le letture esplicite | 310 contro 1324 MB/s, un thread, cache fredda in entrambi | smentita (4,3x) |
| Riordinare il GGUF per co-attivazione degli expert | 4,69 contro 5,17 tok/s | peggiora |
| La decodifica speculativa a n-gram aiuta | nessun guadagno | smentita |
| Letture grandi battono letture numerose | nessun guadagno | smentita |
| `MEM_COMMIT` e' il costo nascosto | 0,37 ms su 1,44 ms di miss, e sta dentro la lettura | in parte |
| Lo streaming residuo limita il modello residente | dimezzando le letture la velocita' non cambia (+0,4%) | smentita |
| Il guadagno dell'IQ2 e' del +65% | a parita' di thread e' +19%; il resto erano i thread | smentita |
| Il binario non c'entra con la velocita' | 1,74x fra due compilazioni dello stesso sorgente | smentita |
| **Un modello piu' grande compresso sa piu' di uno piccolo intatto** | 13,63 contro 13,28 di perplessita': indistinguibili | **smentita** |
| La coda di lettura e' bassa perche' i thread sono pochi | 0,75 miss per nodo: i miss sono rari, non serializzati | ~~smentita~~ **vera**: ai budget bassi sono 2,30-4,99 per nodo, e le letture asincrone a blocchi di 2 valgono +12,3% |
| Blocchi di lettura piu' grandi sono meglio (il mio primo tentativo) | a 4: velocita' invariata, attesa fra thread 2,26x, parete +8%. A 2: +9,6% | **smentita: il massimo non e' il piu' grande** |
| Il margine sulla sovrapposizione si prende allargando il blocco (scritto da me in questo README) | i lotti sono 45/blocco e il margine di bilanciamento lotti/thread: le due cose sono legate, 2 era il massimo | **smentita dall'aritmetica, prima di misurare** |
| Un esponente piu' basso significa che va meglio | il blocco da 2 ha esponente 0,53 contro 0,58, ed e' piu' lento a tutti e quattro i budget | **smentita: era piu' lento in alto, non piu' veloce in basso** |
| Le letture asincrone valgono +7-11%, uniformemente | rimisurate con un binario solo: −0,6% al budget alto, +10,8% al basso | **mezza smentita: la cifra al regime che conta era giusta, l'uniformita' no** |
| Il blocco da 2 vale +3,6% sul sincrono | con lo stesso binario e' indistinguibile a tutti e quattro i budget | **smentita: era differenza fra build, non fra algoritmi** |
| La riproducibilita' e' l'1-4% (misurato fra ripetizioni) | due esecuzioni della stessa curva a sei ore di distanza differiscono del 5% | **vera dentro una sessione, falsa fra sessioni** |
| Una lista di slab riutilizzabili toglie commit e azzeramento | gli indirizzi delle slice sono fissati da ggml: non sono intercambiabili | impossibile |
| **Il prefill domina l'esperienza in una conversazione lunga** | con il riuso del prefisso il TTFT resta piatto a ~5 s da 188 a 1467 token | **smentita** |
| **Con lo store la dimensione non ha un tetto** | esponente 0,62 ± 0,04: 2,6x di expert costano 1,81x di tempo | **vera solo per la memoria** |
| `PRETOUCH` cambia la pendenza della curva (mia previsione, due volte) | a escursione 2,67x: 0,54 senza contro 0,58 con. Alza l'intercetta del 4%, non la pendenza | **smentita: era un artefatto del campione corto** |
| L'azzeramento delle pagine e' lavoro aggiunto da `PRETOUCH` | toglie 0,29 ms dal DMA e li ripaga 0,37 fuori: spostato, non aggiunto | **confermata (la mia ipotesi sul meccanismo)** |
| Il costo per miss peggiora quando il budget si stringe | scende: 1,233 -> 1,063 ms, perche' la coda sale da 2,09 a 2,49 | **smentita, +29% di velocita' regalati** |
| La coda di lettura dello store e' ~1 (costante 0,75 miss/nodo) | 2,30 miss per nodo a 4 GiB, 4,99 a 1,5 GiB; coda 2,09 e 2,49 | **smentita lo stesso giorno in cui l'avevo scritta** |
| Una cache KV su disco varrebbe "oltre cento volte" sul TTFT | il confronto era con *nessuna* cache; contro quella che c'e' gia', non aggiunge niente | smentita |
| **Il tasso di miss dipende dalla residenza** (ipotesi implicita di due previsioni di fila) | l'11,7% di residenza da' il 91,2% di hit: dipende anche da quanto il router concentra, e quello e' del modello | **smentita dalla prima misura su un modello diverso** |
| Piu' expert significa meno localita', quindi piu' miss | 512 expert hanno meno miss di 128 a pari residenza: conta la taglia dell'insieme caldo, non la frequenza d'uso di ciascuno | **smentita, al contrario** |
| Un MoE di classe 80B starebbe a ~5,7 tok/s (extrapolazione), poi 4,0-5,0 (previsione raffinata) | misurato 6,87 | **smentite entrambe, entrambe in difetto** |
| Commit e decommit valgono il 7% del tempo: la riscrittura della residenza vale +13% | su Qwen3-Next sono il 39% del secchio di I/O | **vera per quel modello, scritta come se fosse dello strato** |
| Il disco e' il limite dello store | sull'80B la lettura vera e' il 44% del tempo di I/O e la banda 1082 MB/s su 2500 disponibili | **smentita: il limite e' la contabilita' della residenza** |

Due conclusioni che ho dichiarato e poi dovuto ritirare, perche' sono il tipo di
errore piu' insidioso — misurare con precisione la cosa sbagliata:

- *"La micro-ottimizzazione e' esaurita: eliminando tutto tranne la lettura si
  guadagnerebbe l'8-9%."* L'analisi dei tempi era corretta **dentro un binario
  che lasciava fuori il 45% della macchina**. Va rifatta.
- *"Il collo di bottiglia e' l'I/O, non il calcolo."* Vero in Q4_K_M, dove ogni
  token voleva 110 MB dal disco. Falso nel regime residente, dove il messaggio ha
  continuato a comparire per una soglia cablata.

---

## Logging

Ogni esecuzione aggiunge una riga JSONL in
`%LOCALAPPDATA%\Argine\logs\runs.jsonl` con hardware, modello,
quantizzazione, parametri runtime, RAM peak e media, tok/s, throughput disco,
backend, page fault ed eventuali OOM.

```bat
argine logs --last 20
argine logs --json > risultati.jsonl
```

Serve a confrontare le configurazioni fra loro: e' il materiale grezzo per
rispondere al KPI del progetto.

---

## KPI

> Quanto puo' essere grande un modello MoE utilizzabile su una macchina Windows
> con 16 GB di RAM, mantenendo una velocita' interattiva accettabile?

**Risposta misurata: un MoE da 80 miliardi di parametri, a 6,87 tok/s.**
Qwen3-Next-80B-A3B in Q3_K_M: 34,31 GiB di expert serviti da un budget di 4 GiB,
picco di RAM 6,11 GiB, 871 MiB sempre liberi per il resto del sistema. Vedi
[La misura](#la-misura-687-toks-cioe-previsione-sbagliata-del-37).

Se si vuole la qualita' piena su un modello piu' piccolo: **30,5 miliardi a 9,62
tok/s** (o 12,97 accettando il 13% di perplessita' in piu'). I due numeri
rispondono a due domande diverse — "quanto grande" e "quanto veloce" — e il
secondo non e' il tetto del primo.

**E vale la pena, ma non per il motivo che credevamo.** Un Qwen3-14B denso e
intatto, nello stesso spazio, ha la stessa perplessita' (13,28 contro 13,63,
indistinguibili) e va **3,4 volte piu' piano** (3,83 contro 12,97 tok/s). Il
modello grande compresso non sa di piu': costa molto meno per token a parita' di
qualita'. Vedi [Il controllo](#il-controllo-serviva-davvero-un-modello-grande).

**E crescere costa meno di quanto la curva prevedesse.** Sul 30B la curva
velocita'/residenza da' un esponente di **0,58 ± 0,02**, e da quella avevo
estrapolato ~5,7 tok/s per un 80B. Misurato: **6,87**. L'extrapolazione sbaglia di
1,2 volte in difetto, e la previsione piu' accurata che le e' succeduta (4,0-5,0,
costruita sui byte per token del modello vero) sbaglia di 1,5 volte sul suo centro,
sempre in difetto.

Il motivo e' una variabile che nessuna delle due considerava: **quanto il router
concentra le sue scelte.** Con 512 expert, una cache che ne tiene l'11,7% serve il
91,2% delle richieste. La residenza da sola non predice la velocita' di un modello
che non si e' misurato; serve anche il tasso di miss, e quello dipende dal modello.
Vedi [La misura](#la-misura-687-toks-cioe-previsione-sbagliata-del-37).

Il percorso per riprodurla su un'altra macchina:

1. `argine probe` — fotografa la macchina.
2. `argine inspect modello.gguf` — verifica che il piano sia fattibile.
3. `argine benchmark modello.gguf --sweep-threads 4,6,8,12` — **da fare per
   primo su una CPU nuova**: la scelta dei thread vale piu' di qualunque altra
   cosa in questo elenco.
4. `argine store build` + `benchmark --expert-store` — la configurazione che
   tiene la qualita'.
5. `argine imatrix` + `quantize --experts IQ2_S` + `ppl` — l'alternativa piu'
   veloce, con il suo prezzo misurato invece che stimato.

### Cosa decide la scalabilita'

Non la dimensione del modello: **quanti byte di expert servono per token.** Su
Qwen3-30B-A3B sono 8 expert su 128, cioe' 1/16 del totale. Un MoE con molti
parametri e pochi attivi e' il bersaglio giusto per questa architettura; un
modello con la stessa taglia ma molti parametri attivi per token sarebbe
inutilizzabile a parita' di hardware.

Con tutto residente il tetto e' ~39 miliardi di parametri a 2,24 bit/peso.
Con lo store la dimensione **non ha un tetto di memoria**, e adesso si sa anche
quanto costa in velocita', perche' e' stato misurato invece che previsto.

### La curva, misurata

Qwen3-30B-A3B Q4_K_M, store, backend CPU, 4 thread, tre ripetizioni per punto:

| budget | residente | gen tok/s | coda | TTFT | RAM peak |
|---|---|---|---|---|---|
| 4,0 GiB | 24% | **6,52** | 2,58 | 3,73 s | 5,5 GiB |
| 3,0 GiB | 18% | 5,18 | 2,80 | 4,04 s | 4,5 GiB |
| 2,0 GiB | 12% | 4,35 | 3,49 | 3,97 s | 3,5 GiB |
| 1,5 GiB | 9% | 3,70 | 3,43 | 4,17 s | 3,0 GiB |

Residenza ridotta 2,67x, velocita' ridotta 1,76x: **esponente 0,58 ± 0,02**.

Questa curva e' stata **rimisurata** sei ore dopo, dentro il confronto fra gli
strati di lettura: 6,33 / 5,19 / 4,30 / 3,89, esponente 0,496. Le due esecuzioni
differiscono fino al 5% sul punto piu' basso. Vedi
[Tre strati di lettura](#tre-strati-di-lettura-un-binario-solo) per cosa significa
quello scarto.

### Tre strati di lettura, un binario solo

La prima versione di questo confronto aveva un difetto che non era nei numeri ma
nel metodo: i tre valori venivano da **tre binari diversi, costruiti in tre
sessioni diverse**, e due di quei binari non esistevano piu'. Nessuno — nemmeno
chi l'aveva scritto — poteva rifarlo.

Da qui lo strato di lettura si sceglie a runtime (`ARGINE_EXPERT_READ`, o
`--read-mode`), i tre percorsi stanno uno accanto all'altro nel codice, e
`--sweep-read-mode` li alterna **dentro ogni budget**, cosi' una deriva termica
colpisce tutti e tre invece di premiarne uno. Dodici punti, un binario, un
pomeriggio:

| budget | residente | sincrono | blocco da 2 | emetti e continua | vs sincrono | *la prima volta* |
|---|---|---|---|---|---|---|
| 4,0 GiB | 24,5% | **6,37** | 6,23 | 6,33 | **−0,6%** | *+7,1%* |
| 3,0 GiB | 18,4% | 4,98 | 4,89 | **5,19** | +4,2% | *+7,0%* |
| 2,0 GiB | 12,2% | 4,10 | 4,09 | **4,30** | +4,9% | *+10,7%* |
| 1,5 GiB | 9,2% | 3,51 | 3,52 | **3,89** | **+10,8%** | *+11,1%* |
| esponente | | 0,608 | 0,582 | **0,496** | | |

**Il "+7-11%" era sbagliato come cifra unica, e giusto dove conta.** Il guadagno
non e' uniforme: cresce da zero a +10,8% mentre il budget si stringe. Al budget
piu' alto l'emetti-e-continua **non e' piu' veloce** del sincrono; al piu' basso —
il regime in cui vive un modello grande — lo e' dell'11%, che e' quasi esattamente
la cifra misurata la prima volta.

Il perche' si legge in una colonna sola. Il `sincrono` di oggi fa 6,37 / 4,98 /
4,10 / 3,51 contro i 6,09 / 4,84 / 3,93 / 3,33 di allora: **e' migliorato del 4-5%
a ogni budget.** Oggi e' lo stesso binario dell'asincrono e cambia solo il percorso
di codice; allora era un binario diverso. Quel 4-5% di differenza fra build era
stato attribuito all'ottimizzazione, e non era suo.

Due conseguenze sul resto del documento:

- **Il blocco da 2 non vale niente.** 6,23 / 4,89 / 4,09 / 3,52 contro il sincrono:
  indistinguibile a tutti e quattro i budget. Sembrava +3,6%, ed era la stessa
  differenza fra build.
- **L'esponente dell'asincrono e' il migliore dei tre** (0,496 contro 0,608 e
  0,582), mentre nella prima misura sembrava il peggiore. Avevo scritto una
  riflessione giusta su un dato sbagliato: *"un esponente piu' basso non significa
  che va meglio"* resta vera in generale, ma qui non serviva, perche' l'asincrono
  ha sia l'esponente migliore sia i valori migliori dove il budget stringe.

Due esecuzioni della stessa curva asincrona, a sei ore di distanza, danno
6,52/5,18/4,35/3,70 e 6,33/5,19/4,30/3,89: fino al **5% di scarto** sul punto piu'
basso, contro l'1-4% misurato fra ripetizioni dentro una stessa esecuzione. La
riproducibilita' **fra** sessioni e' peggiore di quella **dentro** una sessione, ed
e' la ragione per cui un confronto va interlacciato invece che messo in fila.

Dettagli e le due strade sbagliate percorse prima in
[Letture asincrone](#letture-asincrone-un-no-un-si-e-uno-strumento-rotto-da-me).

**La risposta e' "sublineare, ma non gratis".** Un modello con 2,6 volte gli expert
e lo stesso budget non costa 2,6 volte: costa **1,76 volte** (1,63 nella seconda
esecuzione della stessa curva: e' il rapporto fra due punti misurati, non una
costante). Lo store compra
qualcosa di reale — senza di lui quel modello semplicemente non parte — ma
l'affermazione *"la dimensione non ha un tetto"*, letta come "si puo' crescere
senza pagare", era falsa. Il tetto non e' di memoria, e' di tempo.

Tradotto nel caso che interessava: un MoE di classe 80B con la stessa frazione
attiva, su questa macchina con la RAM liberata, starebbe intorno a **5,7 tok/s** —
al limite dell'interattivo. Resta un'extrapolazione da un esponente, e lo resta
finche' un modello grande non gira davvero.

> Un modello grande poi ha girato, e questa previsione e' risultata bassa di 1,2
> volte. Le due sezioni che seguono sono il prima e il dopo, lasciati entrambi.

**La coda cresce con la pressione, ed e' la firma dell'effetto giusto**: 2,58 al 24%
di residenza, 3,43 al 9%. Piu' miss significa piu' letture in volo, e un NVMe rende
in funzione di quante richieste ha contemporaneamente. La riga "letture in volo per
attesa" lo dice direttamente: da 2,36 a 3,71, mentre col blocco da 2 era fissa fra
1,30 e 1,48.

Per la proiezione: con la pendenza di 0,58 un MoE di classe 80B su questa macchina
starebbe intorno a **5,7 tok/s**, partendo dai 9,62 misurati con la RAM liberata.

Quel numero era un'extrapolazione, e poi un 80B ha girato davvero: **6,87 tok/s**,
cioe' 1,2 volte piu' veloce della proiezione. Vedi
[La misura](#la-misura-687-toks-cioe-previsione-sbagliata-del-37).

**Il tetto rigido, invece, regge esattamente.** A tutti e quattro i budget la RAM
di picco supera il budget degli expert di **1,5 GiB costanti** — nessuna deriva su
un intervallo di 2,67x. E' la promessa centrale dello store, e questa e' la prima
volta che viene verificata su piu' punti anziche' su uno.

Poi e' stata verificata una seconda volta, su un'altra architettura e un modello
grande: su Qwen3-Next-80B la costante e' **2,11 GiB**, altrettanto ferma su quattro
budget. Il valore cambia col modello — e' quasi tutto nei tensori shared, 1,35 GiB
contro ~1,0 — ma la *costanza* no, ed e' quella la promessa.

Non e' stato necessario scaricare niente: la curva si misura sul modello che si ha
gia' in casa, con
[`--sweep-expert-budget`](#--sweep-expert-budget-misurare-un-modello-che-non-si-e-scaricato),
perche' un modello con 2,6 volte gli expert e lo stesso tetto si comporta come
questo modello con un tetto 2,6 volte piu' piccolo.

### La scelta del modello, e il timore che non si e' avverato

La forma giusta esiste: **Qwen3-Next-80B-A3B** ha 512 expert con 10 attivi piu'
uno condiviso, cioe' **3,87 miliardi di parametri attivi su 79,7** — frazione
attiva del 4,9% contro il 10,8% di Qwen3-30B-A3B. (I 3 miliardi che diceva la
prima stesura venivano dalla scheda del modello; 3,87 e' il valore nei metadati
del GGUF, che e' quello che gira.) Legge *meno* byte di expert per token del
modello che gia' facciamo girare, pur essendo 2,6 volte piu' grande. Sulla carta
e' il bersaglio perfetto.

**E la prima conclusione e' stata che non andava usato per questa misura**, per un motivo che non ha niente a che
vedere con noi: la sua attenzione non e' quella standard ma una ibrida con Gated
DeltaNet, e i kernel relativi su CPU sono lenti e in movimento. Su un EPYC a 48
core fa 11,76 tok/s contro i 63,10 di Qwen3-30B-A3B *a parita' di 3B attivi*; su
un Ryzen a 12 core con il modello interamente in RAM, 7,74 tok/s. Un issue lo da'
per risolto da PR #20334, un altro riporta una regressione da 11 a 3 tok/s.

Misurare il nostro store su quel modello avrebbe prodotto un numero
ininterpretabile: davanti a 2 tok/s non avremmo saputo dire se il colpevole era
la residenza degli expert — la cosa che volevamo misurare — o l'attenzione
lineare di qualcun altro. La regola che ne esce vale oltre questo caso: **per
misurare una variabile serve un modello noioso su tutte le altre.**

**Il timore era fondato e non si e' avverato.** Il modello e' stato misurato
comunque, qualche giorno dopo, e fa **6,87 tok/s** con lo store: non 2, e nemmeno
i 7,74 di un Ryzen a 12 core *con tutto in RAM*. L'attenzione lineare non ha
dominato niente, e il tasso di hit ha spiegato la velocita' da solo. Il
ragionamento qui sopra resta quello giusto da fare prima — un numero
ininterpretabile era un rischio reale — ma la risposta era misurabile, e il costo
di scoprirlo era un pomeriggio. Vedi [La misura](#la-misura-687-toks-cioe-previsione-sbagliata-del-37).

Le forme con attenzione standard restano candidabili, con il vincolo che decide
davvero, cioe' il disco: lo store duplica i byte degli expert, quindi servono
circa due volte la dimensione del GGUF. Con 87 GiB liberi, il GGUF deve stare
sotto i ~44 GiB. (Il Q3_K_M scelto ne occupa 35,67, e lo store 34,31.) (Un modo per togliere la duplicazione sarebbe leggere gli expert
dal GGUF invece che da un file a parte — costerebbe le letture allineate, che
sono il 19,9% del tempo di thread. Non misurato.)

### La previsione, scritta prima di misurare

Il modello scelto per smettere di extrapolare e' **Qwen3-Next-80B-A3B** in
Q3_K_M, con il 30B-A3B nello **stesso** Q3_K_M come controllo: cosi' l'unica
differenza fra i due e' la dimensione.

Leggendone la scheda e' venuta fuori una cosa che cambia l'interpretazione, e
che va detta prima della misura: **l'80B non e' "il 30B con 2,6 volte gli
expert a pari frazione attiva".** Attiva 10 expert su 512 (1,95%) dove il 30B ne
attiva 8 su 128 (6,25%): e' tre volte piu' sparso.

#### Prima previsione, sbagliata in mezz'ora

La prima versione di questo paragrafo diceva **5,2 tok/s**, da un conto fatto
assumendo una quantizzazione uniforme sui tre tensori. E' durata il tempo del
download. Leggendo l'intestazione del file scaricato — i tipi veri, tensore per
tensore — `ffn_down_exps` non e' Q3_K come gli altri due: e' **Q4_K su 45 layer e
Q5_K su 3**. Gli expert non sono 31,1 GiB ma 34,3, e i byte per token non sono
621 ma 684. Stimare la quantizzazione invece di leggerla ha spostato il bersaglio
del 10%, in mezz'ora, sulla stessa pagina.

Vale la pena tenerlo scritto: non per penitenza, ma perche' e' la quarta volta
che un numero assunto al posto di uno letto cambia una conclusione, e le altre
tre volte me ne sono accorto dopo.

#### I numeri veri, letti dal file

| | 30B-A3B Q4_K_M | 80B-A3B Q3_K_M |
|---|---|---|
| expert totali | 16,35 GiB | **34,31 GiB** (2,10x) |
| non-expert (sempre in RAM) | ~1,0 GiB | **1,35 GiB** |
| byte per token | 1046 MiB | **684 MiB** (0,65x) |
| letture per token | 1152 | **1440** (1,25x) |
| byte per lettura | 930 KiB | 440 / 576 / 704 KiB |
| residenza con budget 4 GiB | 24% | **11,7%** |

Il modello grande ha due volte gli expert e ne legge **un terzo in meno per
token**, perche' e' piu' sparso. Le due leve di
[diskbench](#le-due-leve-e-quale-si-puo-tirare-davvero) si muovono in direzioni
opposte: il 25% di letture in piu', ognuna piu' piccola di un terzo. E' lo stesso
scambio misurato la', su un caso che non abbiamo scelto noi.

Un dettaglio che non era scontato: **tutte e tre le dimensioni di slice sono
multipli esatti di 4096** (440, 576 e 704 KiB). Non e' fortuna, e' struttura —
ogni slice contiene esattamente 4096 blocchi di quantizzazione, quindi i byte
sono 4096 x (byte per blocco) qualunque sia il tipo. Senza questo l'I/O senza
cache non sarebbe possibile e il budget non sarebbe un tetto.

**Previsione, da falsificare: fra 4,0 e 5,0 tok/s a budget 4 GiB**, con il centro
intorno a 4,5. Derivata cosi': il 30B al 12% di residenza — praticamente lo stesso
11,7% — fa 4,35 tok/s, cioe' 230 ms per token, di cui ~104 sono il calcolo
(la velocita' a residenza piena, 9,62 tok/s) e ~126 l'attesa del disco. L'80B
legge 0,65 volte i byte con letture piu' piccole, quindi l'attesa scende a ~95 ms;
il calcolo resta l'incognita, perche' 36 dei suoi 48 layer usano la Gated
DeltaNet. A calcolo uguale sono 5,0 tok/s; se la DeltaNet costa una volta e mezza,
4,0.

Cosa smentirebbe cosa:

- **sotto i 4,0 con la quota di ggml cresciuta:** il costo non e' lo store ma
  l'attenzione lineare. Risposta sul modello, non sul metodo.
- **sotto i 4,0 con le quote invariate:** la curva della residenza non si
  trasferisce da un modello all'altro, e l'extrapolazione per classe di
  dimensione va buttata.
- **dentro l'intervallo:** residenza e byte per token bastano a prevedere la
  velocita' di un modello mai misurato, che e' la cosa che
  `--sweep-expert-budget` prometteva di saper fare.

Previsione scritta il 2026-10-03, prima di costruire lo store.

### La misura: 6,87 tok/s, cioe' previsione sbagliata del 37%

| | valore |
|---|---|
| generazione | **6,87 tok/s** (6,96 / 6,87 / 6,81 su tre ripetizioni) |
| prompt | 15,5 tok/s, TTFT 5,00 s (0,64 s con prefisso in cache) |
| budget expert | 4,00 GiB su 34,31 = **11,7% di residenza** |
| RAM di picco | 6,11 GiB, minimo libero 871 MiB |
| hit della cache | **91,2%** (1.771.302 hit, 170.088 miss) |
| letto dallo store | 81,3 GiB a 1082 MB/s aggregati |

**Previsione 4,0-5,0; misurato 6,87.** Fuori dall'intervallo del 37%, e in alto.
Il 30B alla stessa residenza (12%) fa 4,35 tok/s: il modello con **due volte gli
expert e' 1,58 volte piu' veloce**, non piu' lento.

#### Perche': il router concentra, e con 512 expert concentra meglio

La cache tiene l'11,7% degli expert e ne serve il **91,2%**. Se il routing fosse
uniforme, l'11,7% di residenza darebbe l'11,7% di hit. Questo e' il numero che
spiega tutto il resto, ed e' il numero che non avevo previsto: ho corretto
l'aritmetica dei byte due volte e non ho messo in dubbio l'ipotesi implicita che
il tasso di miss dipendesse solo dalla residenza.

Dipende anche da **quanto il router e' concentrato**, e con 512 expert fra cui
scegliere lo e' di piu' che con 128: su 33,7 slice chieste per nodo i miss sono
2,95. Il 30B, a residenza comparabile, ne mancava circa quattro su 24.

L'intuizione sbagliata era: "piu' expert significa ciascuno usato meno spesso,
quindi meno localita'". Vale il contrario, perche' cio' che conta non e' quante
volte si usa un expert ma quanto e' piccolo l'insieme caldo rispetto alla cache.

#### E il collo di bottiglia si e' spostato

Dentro il tempo speso nello strato di I/O (126 s sommati sui thread):

| | 80B-A3B | 30B-A3B |
|---|---|---|
| lettura vera | **44%** | 48% |
| decommit | **31%** | 6% |
| lock | **17%** | 3% |
| commit | 8% | 1% |
| contabilita' in totale | **56%** | 10% |

Su questo modello **leggere e' meno di meta' del lavoro**: il resto e' impegnare
e rilasciare memoria e aspettare i mutex. La causa e' aritmetica: 161.511 sfratti
su slice da 440-704 KiB invece di 930, quindi quasi tre volte la contabilita' per
byte utile. La banda aggregata e' 1082 MB/s, cioe' meno della meta' di quello che
`diskbench` misura sul dispositivo: **il disco non e' piu' il limite.**

Questo riapre una decisione archiviata. Il
[+13% della riscrittura della residenza](#cosa-manca) era stato quantificato e
declinato sul 30B, dove commit e decommit valevano il 7% del tempo di thread. Qui
valgono il 39% del secchio di I/O. Quella stima era corretta **per quel modello**,
e l'ho scritta come se fosse una proprieta' dello strato di residenza: e' lo stesso
errore di forma — una costante misurata in un punto e applicata altrove — comparso
altre quattro volte in questo progetto.

#### Cosa resta vero, e cosa va riscritto

- **Vero:** il tetto di RAM. 34,31 GiB di expert, 4 GiB di budget, 6,11 GiB di
  picco. La residenza e' nostra.
- **Vero:** la risposta al KPI, e migliore di quanto sperato. Un MoE da **80
  miliardi di parametri** gira a **6,87 tok/s** su una macchina da 16 GB senza GPU.
- **Da riscrivere:** l'extrapolazione per classe di dimensione. Non sbaglia di
  poco e non sbaglia in modo prudente: sbaglia di 1,4-1,7 volte, perche' usa una sola
  variabile (la residenza) dove ce ne sono due, e la seconda — la concentrazione
  del router — dipende dal modello e non si deduce dalla sua taglia.
#### La curva dell'80B: esponente 0,38 contro 0,58

Quattro budget, tre ripetizioni ciascuno, lettura asincrona:

| budget | residente | gen tok/s | TTFT | prompt | RAM di picco | picco − budget |
|---|---|---|---|---|---|---|
| 4,0 GiB | 11,7% | **7,10** | 4,97 s | 15,5 | 6,11 GiB | 2,11 GiB |
| 3,0 GiB | 8,7% | 6,32 | 5,36 s | 14,4 | 5,11 GiB | 2,11 GiB |
| 2,0 GiB | 5,8% | 5,68 | 6,20 s | 12,4 | 4,11 GiB | 2,11 GiB |
| 1,5 GiB | 4,4% | 4,87 | 6,34 s | 12,2 | 3,61 GiB | 2,11 GiB |

**Esponente 0,384 ± 0,043**, contro lo 0,58 ± 0,02 del 30B. La curva del modello
grande e' piu' piatta: perde velocita' piu' lentamente di quella del piccolo
mentre il budget si stringe. Stessa causa del resto — il router concentrato tiene
alto il tasso di hit anche quando la cache si riduce.

Lo strumento, per conto suo, **rifiuta di emettere un verdetto** su questo
esponente: l'intervallo [0,341, 0,427] contiene la soglia di 0,35 che separa "la
velocita' regge" da "scende in modo sensibile", e una soglia dentro l'intervallo
significa che la misura non distingue i due casi. Ha ragione lui.

Due cose che vale la pena guardare da sole:

- **picco − budget = 2,11 GiB, costante su tutti e quattro i punti.** Il tetto di
  RAM e' verificato su un secondo modello e una seconda architettura (sul 30B la
  costante era 1,50 GiB; la differenza e' quasi tutta nei tensori shared, 1,35 GiB
  contro ~1,0). Questa e' la promessa centrale del progetto, e regge dove non era
  mai stata provata.
- **a 1,5 GiB di budget l'80B fa 4,87 tok/s, il 30B ne faceva 3,70 e 3,89** nelle sue
  due esecuzioni. Con la stessa
  RAM dedicata agli expert — un gigabyte e mezzo — il modello da 80 miliardi e' piu'
  veloce di quello da 30. Non "quasi usabile": piu' veloce.

Il primo punto della curva (7,10) e il benchmark isolato di mezz'ora prima (6,87)
sono lo stesso punto misurato due volte, a 3,3% di distanza, su due binari diversi e
in due sessioni diverse: dentro il **5%** che separa due esecuzioni della stessa curva
(l'1-4% vale fra ripetizioni *dentro* un'esecuzione, non fra esecuzioni — vedi
[Tre strati di lettura](#tre-strati-di-lettura-un-binario-solo)). E' la prima volta in
questo progetto che un numero viene riprodotto per caso invece che per decisione.



---

## Stato rispetto alle milestone

**Milestone 1 — completa.** Launcher stabile, llama.cpp precompilato scaricato
in automatico, probe hardware, run di GGUF, selezione CPU/Vulkan, RAM limit,
benchmark, API compatibile OpenAI. Verificata end-to-end: caricamento modello,
generazione, `/v1/chat/completions` anche in streaming, telemetria live.

**Milestone 2 — completa.** Detection MoE dai metadati e dai tensori, placement
CPU/GPU degli expert, budget di memoria separato per shared ed expert,
telemetria di residenza (`inspect`, `experts report`, `/argine/experts`).

**Milestone 3 — completa.** L'expert store e' dentro llama.cpp: formato PSXS,
buffer type C++, cache LRU sharded con tetto rigido, letture non bufferizzate,
verifica byte per byte contro il GGUF. Il tetto di RAM e' verificato (79-88% del
budget contro il 105-111% di `mmap`) e l'uscita del modello e' identica bit per
bit con e senza store, anche sotto sfratto.

**Milestone 4 — completa.** Quantizzazione degli expert con matrice di
importanza, misura della perplessita' con firma, curva qualita'/bit su quattro
punti.

**Milestone 5 — completa.** API locale nei due dialetti: OpenAI e Anthropic
(`/v1/messages`), con streaming in entrambi e la traduzione verificata da 13 test
che non avviano niente.

### Cosa manca

- ~~**Confermare l'emetti-e-continua su una seconda esecuzione.**~~ **Fatto**, e la
  cifra era sbagliata: non +7-11% uniforme ma da −0,6% a +10,8% secondo il budget.
  Vedi [Tre strati di lettura](#tre-strati-di-lettura-un-binario-solo).
- **Riprodurre la riga "letture in volo" su un'altra CPU.** 2,36-3,71 dipende da
  quanti miss incontra un thread, e quello dipende da quanti thread sono e da quante
  slice ha un nodo: su una macchina diversa il numero cambia, e con lui il guadagno.
- **Una passata sulle costanti di regime.** Lo stesso errore e' uscito quattro volte
  in un giorno: un numero misurato in un punto di lavoro, inchiodato nel codice o in
  una conclusione, e applicato come universale. Due volte nel verdetto di
  `diskbench`, una nella smentita del pool di lettura, una nel confronto fra
  percentuali con denominatori diversi. Non so quante altre ce ne siano, e questo e'
  il problema: vanno cercate.
- **Il tempo dello store, misurato ma non recuperato.** Il profilo, **misurato sul
  binario sincrono** (cioe' prima dell'emetti-e-continua: le percentuali qui sotto
  sono quelle, non quelle di adesso), dice: lettura vera 19,9% del tempo di thread,
  attesa fra thread 10,2%, azzeramento delle pagine 7,4%, decommit 2,6%, lock 1,1%,
  commit 0,5% — e il calcolo di ggml il 58,3%. Anche azzerando tutto il nostro
  codice il tetto e' 1,7x. La parte raggiungibile (azzeramento + commit +
  decommit, il 10,5% del totale) richiederebbe di scollegare la memoria fisica
  dall'indirizzo virtuale (`VirtualAlloc2` con placeholder e `MapViewOfFile3`, un
  `memfd` su POSIX), perche' smappare e rimappare una sezione non riazzera il
  contenuto. Vale circa **+13%**, in cambio di una riscrittura dello strato di
  residenza con due implementazioni separate, nel punto del codice che ha gia'
  prodotto quattro bug di concorrenza silenziosi. Misurato, quantificato, non
  fatto.

  La stima regge anche sul binario attuale, ma va riespressa: con l'emetti-e-continua
  l'azzeramento non e' piu' una voce a se', perche' finisce dentro l'attesa della
  lettura. In quota sul *nostro* codice (41,7% del tempo di thread, ggml escluso) la
  parte raggiungibile era 10,5/41,7 = **25,2%** prima, ed e' commit + decommit =
  **27,2%** ora: sostanzialmente la stessa. Il **+13%** non cambia; i numeri assoluti
  citati sopra invece sono vecchi, e questo e' il motivo per cui sono etichettati.

**Cancellato:** la cache KV su disco. Era in cima alla lista; la misura su una
conversazione che cresce dice che il riuso in RAM, con un solo stato, copre gia'
il caso d'uso. Vedi [Una conversazione che cresce](#una-conversazione-che-cresce).

**Sospeso:** il kernel a doppia precisione (expert caldi ad alta precisione,
freddi a 2 bit). Era in lista perche' il disco era il collo di bottiglia e tenere
meno byte caldi valeva molto; la misura dice che dimezzare le letture non cambia
la velocita'. La premessa non c'e' piu', e il costo — implementare `MUL_MAT_ID`
per conto nostro — e' di settimane.

---

## Compilare la build patchata

Lo store richiede la patch; una build ufficiale di llama.cpp ignora le variabili
d'ambiente in silenzio. `argine doctor` e il benchmark dicono se il binario in
uso la contiene.

Questa frase e' stata falsa per un po': il benchmark controllava, `doctor` no —
guardava solo i runtime nella cartella dati e non la variabile che decide quale
binario parte, quindi dichiarava "runtime non installato" davanti a una build
indicata a mano e perfettamente funzionante. Trovata facendo la rinomina, con
`doctor` eseguito su una macchina vera. Vale la pena notare *come* e' nata: avevo
scritto la riga del README descrivendo l'intenzione, non il codice, e per mesi
nessuno dei due l'ha contraddetta ad alta voce.

Cross-compilazione da Linux per Windows, con i flag che contano:

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git && cd llama.cpp
git apply ../argine/patches/0002-argine-expert-store.patch
cp ../argine/patches/argine-{store,experts}.{h,cpp} ggml/src/ggml-cpu/
git apply ../argine/patches/0001-argine-route-log.patch   # serve: vedi sotto

cmake -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=mingw.cmake \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF \
  -DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON \
  -DGGML_F16C=ON -DGGML_BMI2=ON -DGGML_OPENMP=OFF \
  -DCMAKE_EXE_LINKER_FLAGS="-static -static-libgcc -static-libstdc++"
ninja -C build llama-server llama-perplexity llama-quantize llama-imatrix
```

Tre dettagli pagati cari:

- **`0001` non e' opzionale, nonostante il nome.** Si chiama `route-log` perche'
  nasce come strumentazione del router — quale expert viene scelto, quante volte
  — ma porta anche uno shim di compatibilita' mingw su
  `THREAD_POWER_THROTTLING_STATE`, che gli header di mingw-w64 non dichiarano.
  Senza, `ggml-cpu.c` non compila e l'errore non ha niente a che vedere col log
  del router: ci ho perso una compilazione a cercarlo nella patch sbagliata.
  Applicarla **dopo** `0002`, altrimenti i contesti non combaciano.
- **`-DGGML_OPENMP=OFF`.** Le release ufficiali per Windows non usano OpenMP, e
  su un carico dominato dalla barriera fra i thread la differenza e' grossa. La
  build con OpenMP era 1,74 volte piu' lenta (resta aperto se il merito sia tutto
  di questo flag o in parte della versione di llama.cpp).
- **Il link statico.** Senza, gli eseguibili dipendono da `libstdc++-6.dll` e
  `libgcc_s_seh-1.dll`, e su un PC con Anaconda nel PATH Windows risolve quelle
  copie — compilate con un modello di thread diverso. Il risultato e' un errore
  su `__gthr_win32_self` all'avvio. Verificare sempre con
  `objdump -p llama-server.exe | grep "DLL Name"`: devono comparire solo DLL di
  sistema.

Poi:

```bat
set ARGINE_LLAMA_DIR=C:\percorso\della\build
```

La variabile vale **solo per quella finestra del terminale**. Aprendone una nuova
si torna alla build ufficiale senza che nulla lo dica, ed e' successo: ha
prodotto un confronto fra due binari diversi interpretato come una differenza fra
due quantizzazioni. Da allora l'etichetta della build include un'impronta del
binario e il benchmark avvisa quando cambia.

---

## Struttura del progetto

```
argine/
  argine/
    cli.py           tutti i comandi
    hardware.py      rilevamento CPU/RAM/GPU/Vulkan/disco e risoluzione del backend
    gguf.py          parser GGUF puro-Python, detection MoE, contabilita' tensori
    planner.py       budget RAM, stima per componente, scala di sicurezza, fase di relax
    runtime.py       binari llama.cpp: download, copia locale, impronta della build
    server.py        avvio llama-server + gateway HTTP con API OpenAI
    bench.py         misure tok/s, TTFT, utilizzo GPU
    monitor.py       campionamento RAM/CPU/disco/page fault del processo
    expert_cache.py  indice expert, cache LRU, prefetch, simulatore
    expert_store.py  formato PSXS: costruzione, indice, verifica, benchmark
    diskbench.py     tetto dello storage sul pattern di accesso degli expert
    quality.py       perplessita', firma della misura, confronto fra modelli
    quantize.py      quantizzazione degli expert, matrice di importanza, stima a secco
    reorder.py       riordino degli expert nel GGUF (esperimento, smentito)
    telemetry.py     logging JSONL
  patches/
    0002-...patch        l'innesto: 15 righe in due file di llama.cpp
    0001-...patch        strumentazione del router + compatibilita' mingw
    argine-store.*       lettore del formato, residenza, cache LRU sharded
    argine-experts.*     il buffer type ggml
    DESIGN-expert-cache.md
  tests/
    test_argine.py       287 test (nessuna rete, nessun modello vero)
    make_fixtures.py     GGUF sintetici sparsi (8B dense, 30B MoE)
    make_tiny_model.py   GGUF llama minuscolo ma realmente caricabile
  strumenti/
    controlla_readme.py  verifica i link interni e rigenera l'indice di questo file
  testo-riferimento.txt    prosa reale per la perplessita'
  testo-calibrazione.txt   prosa reale per la matrice di importanza (disgiunta)
  install.bat / argine.bat
```

Molti test non verificano che il codice funzioni, ma che un errore gia' commesso
non torni: il riferimento della perplessita' scelto male, i thread scelti al
doppio del giusto, i byte per blocco divergenti fra due tabelle, un comando
stampato che il parser rifiuterebbe. L'intestazione di ciascuno racconta il caso
che lo ha generato.

Test:

```bat
python -m unittest discover -s tests -v
```

---

## Risoluzione problemi

**`argine setup` non scarica niente.** La release `latest` di llama.cpp puo'
essere un tag senza binari: Argine scorre le release `bNNNNN` piu' recenti
finche' non trova l'asset giusto. Se non hai rete usa `ARGINE_LLAMA_DIR`.

**Il backend Vulkan non viene rilevato.** Serve `vulkan-1.dll` e un driver
grafico aggiornato. `argine probe --deep` interroga direttamente il binario e
dice se i device vengono davvero enumerati; se non lo sono, l'esecuzione ricade
su CPU.

**`/v1/chat/completions` risponde con un errore di template.** Il modello ha un
chat template che llama.cpp non riesce ad applicare: riprova con `--no-jinja`.

**Il modello non parte per budget insufficiente.** L'output elenca le riduzioni
gia' tentate. Chiudi le applicazioni che occupano RAM, alza `--ram-limit`,
abbassa `--ctx`, oppure passa a una quantizzazione piu' piccola.

**`Impossibile trovare il punto di ingresso __gthr_win32_self`** all'avvio di
`llama-server.exe`. Gli eseguibili non sono linkati staticamente e Windows sta
risolvendo `libstdc++-6.dll` / `libgcc_s_seh-1.dll` contro le copie di Anaconda,
compilate con un modello di thread diverso. Ricompila con
`-static -static-libgcc -static-libstdc++` e verifica con `objdump -p`.

**`[argine] NON attivo: llama-server non ha assegnato nessun tensore al buffer
dello store`.** Il binario non contiene la patch, oppure hai passato
`--no-repack` (che in llama.cpp disabilita *tutti* gli extra buffer type, store
compreso), oppure `ARGINE_LLAMA_DIR` si e' persa aprendo un terminale nuovo.
Argine distingue tre stati — non aperto, aperto, in uso — proprio perche' i
primi due si assomigliano.

**Due misure di velocita' non tornano fra loro.** Controlla la riga
`build=...` di entrambe: dalla versione corrente l'etichetta include un'impronta
del binario, e il benchmark avvisa quando cambia rispetto all'ultima misura dello
stesso modello. Due numeri presi su binari diversi non si confrontano.

### Variabili d'ambiente

| variabile | effetto |
|---|---|
| `ARGINE_HOME` | cartella dati (default `%LOCALAPPDATA%\Argine`) |
| `ARGINE_LLAMA_DIR` | usa una copia locale di llama.cpp invece di scaricarla |
| `ARGINE_HTTP_LOG` | log delle richieste HTTP del gateway |
| `ARGINE_FAKE_RAM`, `ARGINE_FAKE_GPU`, `ARGINE_FAKE_VULKAN`, `ARGINE_FAKE_CORES` | simulano hardware diverso, per provare i piani di memoria di un'altra macchina |

Ogni variabile e' accettata anche col vecchio prefisso `PICOSTAR_`, con un avviso
su stderr: vedi [Migrare da PicoStar16](#migrare-da-picostar16).

Lette dal lato C++, impostate da Argine quando passi `--expert-store`:

| variabile | effetto |
|---|---|
| `ARGINE_EXPERT_STORE` | percorso del file `.psxs`; senza, il buffer type non si registra |
| `ARGINE_EXPERT_BUDGET` | byte di RAM per gli expert. **0 o assente = nessun tetto**, e la cache cresce fino a contenerli tutti |
| `ARGINE_EXPERT_DIRECT` | `1` (default) scavalca la cache dell'OS, `0` la usa |
| `ARGINE_EXPERT_SHARDS` | settori della cache; `0` = automatico |
| `ARGINE_EXPERT_VERBOSE` | statistiche della cache ogni 5 secondi |
| `ARGINE_EXPERT_PRETOUCH` | tocca le pagine prima della lettura, per separare il costo delle pagine fresche da quello dell'I/O |
| ~~`ARGINE_EXPERT_BATCH`~~ | non esiste piu': il blocco di dimensione fissa e' stato sostituito da emetti-e-continua, che non ha una costante da scegliere |
