# Patch per llama.cpp

Patch che Argine applica a una copia di llama.cpp per abilitare funzioni che
il progetto ufficiale non espone. Non servono all'utente finale: Argine
funziona con i binari ufficiali, e queste aggiungono capacita' di misura e
ottimizzazione.

## 0001-argine-route-log.patch

Registra le scelte del router MoE su file, attivandosi con la variabile
d'ambiente `ARGINE_ROUTE_LOG`. Senza la variabile il costo e' una sola
lettura di puntatore per operazione.

Formato prodotto, una riga per chiamata a `MUL_MAT_ID`:

    <nome_tensore> <n_token> <n_expert_usati> <id> <id> ...

Serve a `argine reorder`, che dalla traccia ricava quali expert si attivano
insieme e riscrive il GGUF in modo che siano contigui su disco.

### Applicazione

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp
git apply ../patches/0001-argine-route-log.patch
cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja
cmake --build build --target llama-server -j
```

Poi si indica a Argine dove trovare i binari:

```bash
export ARGINE_LLAMA_DIR=/percorso/llama.cpp/build/bin
```

### Verifica

Il riordino e' una rietichettatura esatta: a parita' di prompt e temperatura 0,
il modello riordinato produce output identico byte per byte all'originale.
Verificato su un qwen3moe di prova (8 expert, 2 attivi, 2 layer).

## argine-store.{h,cpp} — lettore dell'expert store e residenza

Modulo autonomo: non include niente di llama.cpp o di ggml, cosi' si compila e
si collauda da solo. Il buffer type che lo usera' e' un altro file, ancora da
scrivere.

Fa tre cose:

1. **legge l'expert store** prodotto da `argine store build`, con letture
   posizionate esplicite e, se richiesto, `FILE_FLAG_NO_BUFFERING`;
2. **gestisce la residenza**: riserva lo spazio di indirizzamento di un tensore
   per intero e impegna la memoria fisica una slice per volta, cosi' il
   puntatore che ggml riceve resta valido e contiguo mentre la RAM occupata e'
   solo quella delle slice in cache;
3. **applica un budget rigido** con una LRU, i cui slot in uso sono bloccati.

### Perche' esiste

Misurato su i5-1340P, 15.72 GiB, Qwen3-30B-A3B Q4_K_M, sullo stesso disco e con
lo stesso pattern di accesso degli expert:

| percorso | MB/s dal disco |
|---|---|
| memoria mappata, come fa llama.cpp oggi | 562 in generazione |
| letture esplicite, offset allineati, senza cache dell'OS | 4264 |

La differenza non e' il disco: e' il percorso di page fault. A un thread, sulla
stessa slice da 896 KiB e con cache fredda in entrambi i casi, la memoria
mappata rende 310 MB/s e una `ReadFile` 1324 — 4.3x, con la latenza che passa da
2.96 ms a 0.69. Windows non fault-a 896 KiB in una volta: ne prende un pezzo per
volta e paga un giro sull'NVMe ogni volta.

### Collaudo

```bash
g++ -O2 -std=c++17 -Wall -Wextra argine-store.cpp argine-store-test.cpp -o psxtest
./psxtest <store.psxs> <atteso.bin> <n_layers> <n_experts>

g++ -O2 -std=c++17 -I. argine-store.cpp argine-store-stress.cpp -o psxstress
NT=16 DLY=50 SLOTS=3 ./psxstress <store.psxs> x <n_layers> <n_experts>
```

`atteso.bin` sono le slice lette **dal GGUF** e concatenate nell'ordine
(layer, expert, kind): il confronto e' contro la sorgente vera, non contro lo
store stesso. Lo produce il test Python `TestExpertStore`.

Entrambi puliti sotto ThreadSanitizer e AddressSanitizer/UBSan.

### Quattro difetti che il collaudo ha trovato, e che vale la pena ricordare

Sono tutti e quattro invisibili a occhio e tutti e quattro fatali dal vivo.

**1. `ensure` tornava prima che i byte fossero arrivati.** La slice veniva
annunciata nell'indice e letta dopo; un altro thread trovava un hit e calcolava
su memoria non inizializzata. Con `MUL_MAT_ID`, che tutti i thread eseguono
sullo stesso tensore, succedeva a ogni miss. Non si vedeva perche' leggere
8 KiB dalla page cache dura un microsecondo e avviare un thread cinquanta:
la finestra c'era ma non ci si inciampava mai. Da qui
`psx_debug_set_read_delay_us`, che la allarga a comando: con 300 us il test
contava 430 letture sporche su 480. Corretto con tre stati (assente, in
caricamento, pronta) e attesa su condition variable.

**2. `ensure` non teneva la slice.** Tornava 1 e un altro thread poteva
sfrattarla un istante dopo: l'accesso finiva su memoria decommitata, cioe' un
segfault. Trovato dal test da stress alla prima esecuzione. Corretto con il
blocco (`psx_cache_unpin`), e da qui il vincolo che il budget non puo' essere
piu' piccolo del working set di una singola operazione
(`psx_cache_min_budget`). Nota per il dimensionamento: durante il prompt un
ubatch tocca molti piu' expert che durante la generazione, nel caso peggiore
tutti, quindi il minimo si calcola sul caso prompt.

**3. Uno stato letto prima di un'attesa e usato dopo.** Il ciclo di sfratto
aspettava spazio con `wait_for`, che rilascia il mutex, ma la decisione "questa
chiave e' assente" era stata presa prima di aspettare. Al risveglio si inseriva
un secondo slot per la stessa chiave: `index[key]` sovrascriveva il primo, il
suo nodo restava orfano nella lista, gli unpin finivano sul nodo sbagliato e si
arrivava a sfrattare memoria in uso. Il test da stress lo mostrava come
segfault intermittente (2 volte su 3 con 16 thread) e il diario degli eventi
come due `load` consecutivi sullo stesso offset. Corretto ricominciando dalla
ricerca dopo ogni attesa. Effetto collaterale della correzione: gli hit sono
passati da 232 a 2232 su 4800, perche' ogni caricamento duplicato contava come
miss e raddoppiava le letture.

**4. Un descrittore usato fuori dal lock.** `read_with` dereferenziava
`s->handles[idx]` senza il mutex, mentre un altro thread apriva un descrittore e
faceva riallocare il vettore. Segnalato da ThreadSanitizer come corsa in
`_M_realloc_insert`; dal vivo sarebbe stato un use-after-free raro. Corretto
copiando il valore del descrittore sotto il lock.

---

## Argine2 — `0003-argine-research-trace.patch` and `argine-research.{h,cpp}` (English)

Research instrumentation for Phase 1 of Argine2 (observation only). The patch itself only adds the two source files to `ggml/src/ggml-cpu/CMakeLists.txt`; the hooks live in `argine-store.cpp` and `argine-experts.cpp`, which are copied into llama.cpp as before.

Activated by `ARGINE_RESEARCH_TRACE=<path>` (`%p` = process id). Writes a tab-separated raw log (format in the header of `argine-research.h`): router decisions per forward pass and layer, every slice access with outcome (hit / miss / waited), every read with a unique request id and issue / completion / wait timestamps (completion flagged exact or upper bound), evictions with resident time, the residency phase of every MoE node, and cache snapshots including a hard-budget violation counter (`violazioni`, must be 0).

**Format v2** (instrumentation id `argine2-research-v2`, `H` record version 2, 2026-10-03, EXP-021). It adds what offline replay needs to reproduce the store decision by decision:
- `A` (access) records carry the shard id and a per-shard sequence number. The number is assigned under the shard mutex, so the per-shard order is exactly the order of the store's decisions.
- `E` (eviction) records carry shard, sequence and the number of pinned or loading slots skipped before the victim.
- A new `U` (undo) record marks a reservation removed after a failed read.

The parser (`argine/research/rawlog.py`) reads v1 and v2. The v1 binaries in `llama-research/` remain the provenance of the EXP-007 runs; the v2 binaries are in `llama-research-v2/`. Cache behaviour is identical: only the records differ.

`argine-store-fidelity-test.cpp` drives the real store deterministically: one thread, gate/up/down nodes as `argine-experts.cpp` issues them, every 25th step a prompt step. It writes a v2 trace that `tests/test_fidelity.py` replays with `argine/research/storereplay.py` and compares decision by decision:

```bash
g++ -O1 -std=c++17 argine-store.cpp argine-research.cpp argine-store-fidelity-test.cpp -o fidtest -lpthread
./fidtest <store.psxs> <n_layers> <n_experts> <trace> <budget> <shards> <steps> <seed> [pools]
```

`pools = 2` runs prompt steps on one persistent thread and decode steps on another, strictly in turn. This reproduces llama.cpp with `--threads ≠ --threads-batch`, where each pool keeps its pins while the other computes (EXP-021, Windows v2 follow-up).

**Trace tail at termination.** Each thread writes its buffer into the file at the end of every node, and the file is `fflush`ed at most every 250 ms. Records written after the last flush are lost when llama-server is terminated. The flush-sentinel request exists so that this tail is never workload. `argine research fidelity` checks it: missing sequence numbers are classified, and node completeness is checked against the router records.

Order of application on llama.cpp `254b177307b14d055bbe4c050c9c441a1e06d8a4`:

```bash
git apply ../patches/0002-argine-expert-store.patch
cp ../patches/argine-{store,experts,research}.{h,cpp} ggml/src/ggml-cpu/
git apply ../patches/0001-argine-route-log.patch
git apply ../patches/0003-argine-research-trace.patch
```

Newer llama.cpp (from `631109b`, 2026-10-02) adds `alloc_buffer_n` to the buffer-type interface; `argine-experts.cpp` must then initialise that member (not done yet: the commit is pinned instead).

### Test

```bash
g++ -O1 -g -std=c++17 -fsanitize=thread argine-store.cpp argine-research.cpp argine-research-test.cpp -o arxtest -lpthread
./arxtest <store.psxs> <n_layers> <n_experts> <trace-file>
```

Checks: identical cache statistics with and without the trace (observation only); per-shard sequence numbers gapless and covering every shard (v2); exactly one read record per miss with the same request id; issue ≤ completion and wait start ≤ wait end; every completion flagged exact xor upper bound; every eviction logged; no budget violation with 6 threads. Clean under ThreadSanitizer and AddressSanitizer/UBSan; the existing store test (79 checks) and stress test also pass with tracing on. `tests/test_research.py` compiles and runs it automatically when `g++` is available.

### What has and has not been executed

- Linux, synchronous POSIX read path: unit tests + end-to-end llama-server run (EXP-019).
- Windows overlapped-I/O path (precise completion stamps with `WaitForMultipleObjects`, polling with `HasOverlappedIoCompleted`): compiled with MinGW-w64, **not yet executed** *(historical wording, 2026-10-03)*. [2026-10-12] It has since been executed on Windows: the 2026-10-03 smoke run (66.6% exact completions) and the EXP-019 validation of 2026-10-12 (63.4% exact, the rest upper bounds).
