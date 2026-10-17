// Collaudo del lettore dello store, indipendente da llama.cpp.
//
// Verifica che il C++ legga esattamente cio' che il Python ha scritto, che la
// residenza a pezzi funzioni (indirizzi riservati, memoria impegnata e
// rilasciata), e che il budget rigido sia davvero rigido.
//
//   g++ -O2 -std=c++17 argine-store.cpp argine-store-test.cpp -o psxtest
//   ./psxtest <store.psxs> <atteso.bin> <n_layers> <n_experts>
//
// `atteso.bin` e' prodotto dal test Python: le slice concatenate nell'ordine
// (layer, expert, kind), cosi' il confronto e' byte per byte contro la
// sorgente vera, che e' il GGUF.

#include "argine-store.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static int fallimenti = 0;

static void check(bool ok, const char * cosa) {
    printf("  [%s] %s\n", ok ? "ok" : "NO", cosa);
    if (!ok) {
        fallimenti++;
    }
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "uso: %s <store.psxs> <atteso.bin> <n_layers> <n_experts>\n", argv[0]);
        return 2;
    }
    const char * store_path = argv[1];
    const char * atteso_path = argv[2];
    const uint32_t n_layers  = (uint32_t) atoi(argv[3]);
    const uint32_t n_experts = (uint32_t) atoi(argv[4]);

    // --- apertura -----------------------------------------------------------
    printf("apertura\n");
    // su Linux O_DIRECT su un filesystem qualunque puo' fallire: si prova
    // prima senza, che e' cio' che conta per la correttezza dei byte
    psx_store * s = psx_open(store_path, 0);
    if (psx_error(s)) {
        printf("  errore: %s\n", psx_error(s));
        return 1;
    }
    check(psx_n_layers(s) == n_layers, "numero di layer");
    check(psx_n_experts(s) == n_experts, "numero di expert");
    check(psx_alignment(s) == 4096, "allineamento 4096");

    // --- confronto dei byte con l'atteso -----------------------------------
    printf("confronto dei byte con il GGUF\n");
    FILE * fa = fopen(atteso_path, "rb");
    if (!fa) {
        printf("  errore: manca %s\n", atteso_path);
        return 1;
    }
    std::vector<unsigned char> mio, suo;
    size_t confrontate = 0, diverse = 0;
    for (uint32_t l = 0; l < n_layers; l++) {
        for (uint32_t x = 0; x < n_experts; x++) {
            for (int k = 0; k < (int) PSX_N_KINDS; k++) {
                const psx_entry * e = psx_get(s, l, x);
                if (!e) {
                    printf("  errore: manca l'indice per (%u,%u)\n", l, x);
                    return 1;
                }
                const size_t sz = e->size[k];
                mio.assign(psx_slice_read_bytes(s, l, x, (psx_kind) k), 0);
                if (psx_read_slice(s, l, x, (psx_kind) k, mio.data()) != sz) {
                    printf("  errore: lettura fallita su (%u,%u,%d)\n", l, x, k);
                    return 1;
                }
                suo.assign(sz, 0);
                if (fread(suo.data(), 1, sz, fa) != sz) {
                    printf("  errore: atteso troncato a (%u,%u,%d)\n", l, x, k);
                    return 1;
                }
                if (memcmp(mio.data(), suo.data(), sz) != 0) {
                    diverse++;
                }
                confrontate++;
            }
        }
    }
    fclose(fa);
    printf("  %zu slice confrontate\n", confrontate);
    check(diverse == 0, "tutte identiche al GGUF");

    // --- residenza ----------------------------------------------------------
    printf("residenza a pezzi\n");
    const size_t slice = psx_slice_read_bytes(s, 0, 0, PSX_GATE);
    const size_t n_exp = n_experts;
    psx_region * r = psx_region_new(slice * n_exp, slice, n_exp);
    check(r != nullptr, "regione riservata");
    if (!r) {
        return 1;
    }
    check(psx_region_committed(r) == 0, "nulla impegnato all'inizio");
    check(psx_region_commit(r, 0, slice) == 1, "impegno di una slice");
    check(psx_region_committed(r) >= slice, "contabilita' dell'impegnato");
    // scrivere nella slice impegnata deve funzionare
    memset(psx_region_base(r), 0x5A, slice);
    check(((unsigned char *) psx_region_base(r))[slice - 1] == 0x5A, "slice scrivibile");
    check(psx_region_decommit(r, 0, slice) == 1, "rilascio della slice");
    check(psx_region_committed(r) == 0, "contabilita' dopo il rilascio");

    // --- cache con budget rigido -------------------------------------------
    printf("cache LRU con budget rigido\n");
    // budget per 4 slice: la quinta deve sfrattare la prima
    psx_cache * c = psx_cache_new((uint64_t) slice * 4);
    for (uint32_t x = 0; x < 4; x++) {
        check(psx_cache_ensure(c, s, r, x * slice, 0, x, PSX_GATE) == 1, "slice residente");
        psx_cache_unpin(c, r, x * slice);
    }
    struct psx_cache_stats st;
    psx_cache_get_stats(c, &st);
    check(st.bytes_resident == (uint64_t) slice * 4, "quattro slice residenti");
    check(st.evictions == 0, "nessuno sfratto finche' il budget basta");
    check(st.misses == 4 && st.hits == 0, "quattro miss");

    // riletture: devono essere hit, senza toccare il disco
    const uint64_t letti_prima = st.bytes_read;
    for (uint32_t x = 0; x < 4; x++) {
        psx_cache_ensure(c, s, r, x * slice, 0, x, PSX_GATE);
        psx_cache_unpin(c, r, x * slice);
    }
    psx_cache_get_stats(c, &st);
    check(st.hits == 4, "quattro hit");
    check(st.bytes_read == letti_prima, "nessuna lettura in piu' sui hit");

    // la quinta sfratta, e il budget NON viene superato
    check(psx_cache_ensure(c, s, r, 4 * slice, 0, 4, PSX_GATE) == 1, "quinta slice");
    psx_cache_unpin(c, r, 4 * slice);
    psx_cache_get_stats(c, &st);
    check(st.evictions == 1, "uno sfratto");
    check(st.bytes_resident <= st.budget, "budget rispettato: e' il requisito del progetto");
    check(psx_region_committed(r) <= (size_t) st.budget, "RAM impegnata entro il budget");

    // contenuto corretto anche dopo lo sfratto: la slice 4 deve contenere i suoi byte
    {
        std::vector<unsigned char> ref(psx_slice_read_bytes(s, 0, 4, PSX_GATE), 0);
        psx_store * s2 = psx_open(store_path, 0);
        psx_read_slice(s2, 0, 4, PSX_GATE, ref.data());
        const unsigned char * in_cache = (unsigned char *) psx_region_base(r) + 4 * slice;
        check(memcmp(in_cache, ref.data(), psx_get(s, 0, 4)->size[PSX_GATE]) == 0,
              "la slice letta dopo lo sfratto e' quella giusta");
        psx_close(s2);
    }

    // budget piu' piccolo di una slice: deve rifiutare, non mentire
    {
        psx_cache * piccola = psx_cache_new(slice / 2);
        check(psx_cache_ensure(piccola, s, r, 0, 0, 0, PSX_GATE) == 0,
              "budget sotto una slice: rifiuta invece di sforare");
        psx_cache_free(piccola);
    }

    // --- concorrenza --------------------------------------------------------
    printf("accesso da piu' thread\n");
    {
        psx_cache *  cc = psx_cache_new((uint64_t) slice * n_exp);
        psx_region * rr = psx_region_new(slice * n_exp, slice, n_exp);
        std::vector<std::thread> ts;
        std::vector<int>         esiti(8, 1);
        for (int t = 0; t < 8; t++) {
            ts.emplace_back([&, t] {
                for (int giro = 0; giro < 40; giro++) {
                    const uint32_t x = (uint32_t) ((t * 7 + giro * 3) % n_exp);
                    if (!psx_cache_ensure(cc, s, rr, x * slice, 0, x, PSX_GATE)) {
                        esiti[t] = 0;
                        return;
                    }
                    psx_cache_unpin(cc, rr, x * slice);
                }
            });
        }
        for (auto & th : ts) {
            th.join();
        }
        bool tutti = true;
        for (int e : esiti) {
            tutti = tutti && e;
        }
        check(tutti, "8 thread, nessun errore");
        psx_cache_get_stats(cc, &st);
        check(st.bytes_resident <= st.budget, "budget rispettato anche in concorrenza");
        // i byte in cache devono essere quelli giusti, per ogni slice residente
        std::vector<unsigned char> ref;
        size_t sbagliate = 0;
        for (uint32_t x = 0; x < n_exp; x++) {
            ref.assign(psx_slice_read_bytes(s, 0, x, PSX_GATE), 0);
            psx_read_slice(s, 0, x, PSX_GATE, ref.data());
            const unsigned char * p = (unsigned char *) psx_region_base(rr) + x * slice;
            if (memcmp(p, ref.data(), psx_get(s, 0, x)->size[PSX_GATE]) != 0) {
                sbagliate++;
            }
        }
        check(sbagliate == 0, "contenuto corretto dopo l'accesso concorrente");
        psx_cache_free(cc);
        psx_region_free(rr);
    }

    // --- distribuzione fra i settori ----------------------------------------
    //
    // La cache e' divisa in settori per togliere la contesa sul mutex. Se le
    // chiavi non si distribuiscono, i settori esistono ma se ne usa uno solo e
    // la cache vale 1/16 di quella dichiarata: e' successo, con gli hit crollati
    // dal 94% al 67% e la generazione dimezzata, e nessun messaggio d'errore.
    //
    // La causa era aritmetica: gli offset delle slice sono multipli di 4096 e la
    // base della regione e' allineata a 64 KiB, quindi i quattro bit bassi della
    // chiave sono sempre zero, e il modulo 16 di zero e' zero. Questo test usa
    // esattamente quegli offset.
    printf("distribuzione delle chiavi fra i settori\n");
    {
        psx_cache *  cs = psx_cache_new((uint64_t) 5 << 30);  // 5 GiB -> 16 settori
        const unsigned n = psx_cache_shards(cs);
        check(n > 1, "con un budget grande i settori sono piu' di uno");
        std::vector<int> visti(n, 0);
        // offset realistici: tre tensori per layer, slice allineate alla pagina
        const size_t slab[3] = { 884736, 884736, 1290240 };
        size_t base = 0;
        for (uint32_t l = 0; l < 48; l++) {
            for (int k = 0; k < 3; k++) {
                for (uint32_t x = 0; x < 128; x++) {
                    const size_t off = base + (size_t) x * slab[k];
                    visti[psx_debug_shard_index(cs, r, off)] = 1;
                }
                base += 128 * slab[k];
                base = (base + 4095) & ~(size_t) 4095;  // ogni tensore allineato
            }
        }
        size_t usati = 0;
        for (int v : visti) {
            usati += (size_t) v;
        }
        printf("  %zu settori su %u ricevono chiavi\n", usati, n);
        check(usati == n, "ogni settore riceve chiavi: altrimenti la cache vale "
                          "una frazione di quella dichiarata");
        psx_cache_free(cs);
    }

    printf("equilibrio dei settori sotto carico\n");
    {
        const size_t s3 = slice;
        psx_cache *  cs = psx_cache_new((uint64_t) s3 * n_exp * 4);
        psx_region * r3 = psx_region_new(s3 * n_exp, s3, n_exp);
        for (uint32_t x = 0; x < n_exp; x++) {
            psx_cache_ensure(cs, s, r3, x * s3, 0, x, PSX_GATE);
            psx_cache_unpin(cs, r3, x * s3);
        }
        const unsigned n3 = psx_cache_shards(cs);
        std::vector<uint64_t> carico(n3, 0);
        psx_cache_shard_load(cs, carico.data(), n3);
        uint64_t somma = 0;
        for (uint64_t v : carico) {
            somma += v;
        }
        psx_cache_stats st3;
        psx_cache_get_stats(cs, &st3);
        check(somma == st3.bytes_resident,
              "la somma dei settori e' il totale residente");
        psx_cache_free(cs);
        psx_region_free(r3);
    }

    // --- la corsa che conta -------------------------------------------------
    //
    // In llama.cpp MUL_MAT_ID viene eseguita da tutti i thread sullo stesso
    // tensore: su un miss decine di thread chiedono la STESSA slice nello
    // stesso istante. Se `ensure` torna 1 prima che i byte siano arrivati, ggml
    // calcola su memoria non inizializzata e il modello sbaglia in silenzio.
    // Questo e' il test che deve accorgersene.
    printf("corsa su una slice fredda richiesta da tutti i thread\n");
    {
        std::vector<unsigned char> ref(psx_slice_read_bytes(s, 1, 3, PSX_DOWN), 0);
        psx_read_slice(s, 1, 3, PSX_DOWN, ref.data());
        const size_t utili = psx_get(s, 1, 3)->size[PSX_DOWN];

        // finestra allargata: senza questo la corsa non si vede mai
        psx_debug_set_read_delay_us(300);
        size_t sporche = 0;
        const int giri = 40;
        for (int giro = 0; giro < giri; giro++) {
            psx_cache *  cc = psx_cache_new((uint64_t) ref.size() * 2);
            psx_region * rr = psx_region_new(ref.size() * 2, ref.size(), 2);
            const int    nt = 12;
            std::vector<std::thread> ts;
            std::vector<int>         visto_sbagliato(nt, 0);
            for (int t = 0; t < nt; t++) {
                ts.emplace_back([&, t] {
                    if (!psx_cache_ensure(cc, s, rr, 0, 1, 3, PSX_DOWN)) {
                        visto_sbagliato[t] = 1;
                        return;
                    }
                    // appena ensure dice "residente", i byte devono esserci
                    const unsigned char * p = (unsigned char *) psx_region_base(rr);
                    if (memcmp(p, ref.data(), utili) != 0) {
                        visto_sbagliato[t] = 1;
                    }
                    psx_cache_unpin(cc, rr, 0);
                });
            }
            for (auto & th : ts) {
                th.join();
            }
            for (int v : visto_sbagliato) {
                sporche += (size_t) v;
            }
            psx_cache_free(cc);
            psx_region_free(rr);
        }
        psx_debug_set_read_delay_us(0);
        printf("  %d giri x 12 thread: %zu letture su memoria non pronta\n", giri, sporche);
        check(sporche == 0,
              "ensure non torna prima che i byte siano arrivati");
    }

    // --- il pin protegge dallo sfratto --------------------------------------
    //
    // Il difetto trovato dal test da stress: `ensure` diceva "residente" e un
    // altro thread poteva sfrattare la slice un istante dopo, lasciando il
    // chiamante a leggere memoria decommitata. Segfault, non dati sbagliati.
    printf("il blocco protegge dallo sfratto\n");
    {
        psx_cache *  cc = psx_cache_new((uint64_t) slice * 2);
        psx_region * rr = psx_region_new(slice * 8, slice, 8);
        // due slice bloccate saturano il budget
        check(psx_cache_ensure(cc, s, rr, 0, 0, 0, PSX_GATE) == 1, "prima bloccata");
        check(psx_cache_ensure(cc, s, rr, slice, 0, 1, PSX_GATE) == 1, "seconda bloccata");
        // la terza non ha spazio e non puo' sfrattare nulla: deve rifiutare,
        // non sfrattare una slice in uso
        check(psx_cache_ensure(cc, s, rr, 2 * slice, 0, 2, PSX_GATE) == 0,
              "rifiuta invece di sfrattare una slice in uso");
        // le due bloccate sono ancora leggibili: se fossero state sfrattate,
        // questo accesso sarebbe un crash
        volatile unsigned char tocco = ((unsigned char *) psx_region_base(rr))[0];
        tocco = (unsigned char) (tocco + ((unsigned char *) psx_region_base(rr))[slice]);
        (void) tocco;
        check(true, "le slice bloccate sono ancora accessibili");
        // sbloccata la prima, la terza entra
        psx_cache_unpin(cc, rr, 0);
        check(psx_cache_ensure(cc, s, rr, 2 * slice, 0, 2, PSX_GATE) == 1,
              "dopo lo sblocco lo sfratto e' possibile");
        psx_cache_unpin(cc, rr, slice);
        psx_cache_unpin(cc, rr, 2 * slice);
        psx_cache_free(cc);
        psx_region_free(rr);
    }

    // --- lettura in blocco ---------------------------------------------------
    //
    // La ragione d'essere del blocco e' avere piu' letture in volo insieme: a
    // byte totali fissi per nodo la profondita' di coda rende piu' della
    // dimensione della lettura. Qui non si misura la velocita', si verifica che
    // il risultato sia identico a quello di n chiamate singole — compresi i casi
    // in cui storicamente si rompeva: fallimento a meta' del blocco e pressione
    // di sfratto *dentro* un blocco.
    printf("lettura in blocco\n");
    {
        const size_t  N = 6;
        psx_cache *   cc = psx_cache_new((uint64_t) slice * 16);
        psx_region *  rr = psx_region_new(slice * 16, slice, 16);
        psx_region *  regs[N];
        size_t        offs[N];
        uint32_t      lays[N], exps[N];
        enum psx_kind kinds[N];
        for (size_t i = 0; i < N; i++) {
            regs[i]  = rr;
            offs[i]  = i * slice;
            lays[i]  = 0;
            exps[i]  = (uint32_t) (i / 3);
            kinds[i] = (enum psx_kind) (i % 3);
        }
        check(psx_cache_ensure_many(cc, s, N, regs, offs, lays, exps, kinds) == 1,
              "sei slice in un blocco");
        // il contenuto deve coincidere con la lettura diretta, slice per slice
        bool uguali = true;
        std::vector<unsigned char> una(slice, 0);
        for (size_t i = 0; i < N; i++) {
            psx_read_slice(s, lays[i], exps[i], kinds[i], una.data());
            const unsigned char * p = (unsigned char *) psx_region_base(rr) + offs[i];
            if (memcmp(p, una.data(), psx_slice_read_bytes(s, lays[i], exps[i], kinds[i])) != 0) {
                uguali = false;
            }
        }
        check(uguali, "ogni slice del blocco contiene i byte giusti");

        psx_cache_stats st;
        psx_cache_get_stats(cc, &st);
        check(st.misses == N, "un miss per slice, non uno per blocco");
        check(st.bytes_resident == (uint64_t) slice * N, "residenza contata per slice");

        // ripetere lo stesso blocco deve essere tutto hit, e non toccare la
        // residenza: e' il percorso in cui un conteggio doppio si vedrebbe
        for (size_t i = 0; i < N; i++) {
            psx_cache_unpin(cc, rr, offs[i]);
        }
        check(psx_cache_ensure_many(cc, s, N, regs, offs, lays, exps, kinds) == 1,
              "il secondo blocco identico riesce");
        psx_cache_get_stats(cc, &st);
        check(st.hits == N, "tutte hit al secondo giro");
        check(st.bytes_resident == (uint64_t) slice * N, "la residenza non e' raddoppiata");
        for (size_t i = 0; i < N; i++) {
            psx_cache_unpin(cc, rr, offs[i]);
        }
        psx_cache_free(cc);
        psx_region_free(rr);
    }

    // Un blocco piu' grande del budget: deve rifiutare **senza** lasciare slice
    // prenotate a meta'. Una slice che resta "in caricamento" per sempre blocca
    // chiunque la aspetti, ed e' il guasto peggiore possibile qui: non un errore
    // ma un blocco permanente.
    printf("blocco piu' grande del budget\n");
    {
        const size_t  N = 4;
        psx_cache *   cc = psx_cache_new((uint64_t) slice * 2);  // ne stanno due
        psx_region *  rr = psx_region_new(slice * 8, slice, 8);
        psx_region *  regs[N];
        size_t        offs[N];
        uint32_t      lays[N], exps[N];
        enum psx_kind kinds[N];
        for (size_t i = 0; i < N; i++) {
            regs[i]  = rr;
            offs[i]  = i * slice;
            lays[i]  = 0;
            exps[i]  = (uint32_t) i;
            kinds[i] = PSX_GATE;
        }
        const int esito = psx_cache_ensure_many(cc, s, N, regs, offs, lays, exps, kinds);
        check(esito == 0, "rifiuta un blocco che non ci sta");
        // Nessuna slice deve essere rimasta in caricamento: lo si verifica
        // chiedendone una da sola, cosa che si bloccherebbe per sempre se un
        // fantasma fosse rimasto nell'indice con ready=false.
        psx_cache_stats st;
        psx_cache_get_stats(cc, &st);
        check(st.bytes_resident <= (uint64_t) slice * 2, "budget rispettato dopo il rifiuto");
        check(psx_cache_ensure(cc, s, rr, 0, 0, 0, PSX_GATE) == 1,
              "dopo il rifiuto la cache e' ancora usabile");
        psx_cache_unpin(cc, rr, 0);
        psx_cache_free(cc);
        psx_region_free(rr);
    }

    // Blocchi concorrenti sulle stesse slice, con budget stretto: e' la
    // combinazione che ha prodotto i quattro bug silenziosi della versione a
    // slice singola. Qui si verifica solo che nessuno veda byte sbagliati.
    // Blocchi concorrenti con sfratto fra un'operazione e l'altra.
    //
    // Due premesse sbagliate prima di arrivare qui, e vale la pena che restino
    // scritte. La prima dava a (expert, kind) diversi lo stesso offset: la chiave
    // della cache e' (regione, offset), quindi il test chiedeva due contenuti
    // dallo stesso indirizzo e poi si stupiva. La seconda dava 8 thread x 3 slice
    // bloccate insieme con un budget da 6: sotto il minimo per costruzione, e il
    // risultato era uno stallo di due secondi per tentativo, non un errore.
    //
    // Il pattern vero: il budget tiene piu' di un'operazione ma non tutto, quindi
    // lo sfratto avviene fra un blocco e il successivo, mentre dentro un blocco
    // tutte le slice restano residenti. E' esattamente cio' che garantisce
    // psx_cache_min_budget.
    printf("blocchi concorrenti con sfratto\n");
    {
        const size_t N  = 3;   // un expert: gate, up, down
        const int    NT = 8;
        // 2 layer x 8 expert x 3 kind = 48 slot, tutti con offset distinto
        psx_region * rr = psx_region_new(slice * 48, slice, 48);
        // il budget tiene 27 slice: piu' dei 24 che NT thread bloccano insieme,
        // meno delle 48 totali, quindi si sfratta fra un giro e l'altro
        psx_cache *  cc = psx_cache_new((uint64_t) slice * 27);
        std::vector<int> sbagliate(NT, 0);
        std::vector<int> rifiutati(NT, 0);
        std::vector<std::thread> ts;
        for (int t = 0; t < NT; t++) {
            ts.emplace_back([&, t] {
                std::vector<unsigned char> una(slice, 0);
                for (int giro = 0; giro < 40; giro++) {
                    // Tutte e 16 le coppie (layer, expert) devono essere
                    // toccate, altrimenti le slice distinte stanno nel budget e
                    // lo sfratto non viene mai esercitato: la prima versione ne
                    // copriva 8 su 16 e il test passava senza provare niente.
                    const uint32_t idx = (uint32_t) ((t * 7 + giro) % 16);
                    const uint32_t lay = idx / 8;
                    const uint32_t x   = idx % 8;
                    psx_region *  regs[N];
                    size_t        offs[N];
                    uint32_t      lays[N], exps[N];
                    enum psx_kind kinds[N];
                    for (size_t i = 0; i < N; i++) {
                        regs[i]  = rr;
                        // unico per (layer, expert, kind)
                        offs[i]  = (((size_t) lay * 8 + x) * N + i) * slice;
                        lays[i]  = lay;
                        exps[i]  = x;
                        kinds[i] = (enum psx_kind) i;
                    }
                    if (psx_cache_ensure_many(cc, s, N, regs, offs, lays, exps, kinds) != 1) {
                        rifiutati[t]++;
                        continue;  // budget momentaneamente bloccato da altri
                    }
                    for (size_t i = 0; i < N; i++) {
                        const size_t utili = psx_slice_read_bytes(s, lays[i], exps[i], kinds[i]);
                        psx_read_slice(s, lays[i], exps[i], kinds[i], una.data());
                        const unsigned char * p =
                            (unsigned char *) psx_region_base(rr) + offs[i];
                        if (memcmp(p, una.data(), utili) != 0) {
                            sbagliate[t]++;
                        }
                    }
                    for (size_t i = 0; i < N; i++) {
                        psx_cache_unpin(cc, rr, offs[i]);
                    }
                }
            });
        }
        for (auto & th : ts) {
            th.join();
        }
        size_t tot = 0, rif = 0;
        for (int i = 0; i < NT; i++) {
            tot += (size_t) sbagliate[i];
            rif += (size_t) rifiutati[i];
        }
        psx_cache_stats st;
        psx_cache_get_stats(cc, &st);
        printf("  8 thread x 40 blocchi da 3: %zu slice sbagliate, %zu blocchi rifiutati, "
               "%llu sfratti\n", tot, rif, (unsigned long long) st.evictions);
        check(tot == 0, "nessun blocco consegna byte sbagliati sotto contesa");
        check(st.bytes_resident <= (uint64_t) slice * 27, "budget rispettato in concorrenza");
        check(st.evictions > 0, "lo sfratto e' stato esercitato: altrimenti il test e' vuoto");
        psx_cache_free(cc);
        psx_region_free(rr);
    }

    // Un blocco piu' grande del budget deve fallire **subito**, non dopo due
    // secondi di attese inutili: la sua impossibilita' e' calcolabile in partenza.
    printf("blocco oltre il budget: rifiuto immediato\n");
    {
        const size_t  N = 4;
        psx_cache *   cc = psx_cache_new((uint64_t) slice * 2);
        psx_region *  rr = psx_region_new(slice * 8, slice, 8);
        psx_region *  regs[N];
        size_t        offs[N];
        uint32_t      lays[N], exps[N];
        enum psx_kind kinds[N];
        for (size_t i = 0; i < N; i++) {
            regs[i] = rr; offs[i] = i * slice; lays[i] = 0;
            exps[i] = (uint32_t) i; kinds[i] = PSX_GATE;
        }
        const auto t0 = std::chrono::steady_clock::now();
        const int esito = psx_cache_ensure_many(cc, s, N, regs, offs, lays, exps, kinds);
        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t0).count();
        printf("  rifiutato in %lld ms\n", (long long) ms);
        check(esito == 0, "rifiuta un blocco che non ci sta");
        check(ms < 100, "rifiuta subito: calcolabile senza provare");
        psx_cache_free(cc);
        psx_region_free(rr);
    }

    // Tutto o niente quando la prenotazione fallisce **a meta'**, non nel
    // controllo preventivo. Lo spazio basterebbe per il blocco, ma una slice
    // bloccata da qualcun altro lo occupa: le prime due del blocco entrano, la
    // terza no. Se le prime due restassero bloccate, il budget resterebbe
    // occupato da pin che nessuno sblocchera' mai — e chi chiama non saprebbe
    // nemmeno quali slice sbloccare.
    printf("tutto o niente a meta' blocco\n");
    {
        const size_t  N = 3;
        psx_cache *   cc = psx_cache_new((uint64_t) slice * 3);  // tre slice esatte
        psx_region *  rr = psx_region_new(slice * 8, slice, 8);
        // una slice bloccata da parte: ne restano due libere su tre
        check(psx_cache_ensure(cc, s, rr, 7 * slice, 0, 7, PSX_GATE) == 1,
              "slice di disturbo bloccata");
        psx_region *  regs[N];
        size_t        offs[N];
        uint32_t      lays[N], exps[N];
        enum psx_kind kinds[N];
        for (size_t i = 0; i < N; i++) {
            regs[i] = rr; offs[i] = i * slice; lays[i] = 0;
            exps[i] = (uint32_t) i; kinds[i] = PSX_GATE;
        }
        // la somma sta nel budget, quindi il controllo preventivo passa: il
        // fallimento avviene dentro, alla terza prenotazione
        check(psx_cache_ensure_many(cc, s, N, regs, offs, lays, exps, kinds) == 0,
              "il blocco fallisce perche' una slice e' bloccata da altri");
        // Ora libero il disturbo: se il blocco fallito non ha lasciato pin, le tre
        // slice entrano. Se li ha lasciati, questo non riesce.
        psx_cache_unpin(cc, rr, 7 * slice);
        check(psx_cache_ensure_many(cc, s, N, regs, offs, lays, exps, kinds) == 1,
              "dopo il fallimento non resta bloccato niente");
        for (size_t i = 0; i < N; i++) {
            psx_cache_unpin(cc, rr, offs[i]);
        }
        psx_cache_free(cc);
        psx_region_free(rr);
    }

    // Lo stesso contratto quando il guasto e' nella LETTURA, non nella
    // prenotazione. E' il caso che conta davvero e che il test precedente non
    // raggiunge: le prime slice del blocco sono state lette e bloccate per
    // davvero, e sono quelle il cui sblocco va verificato. Senza l'iniezione di
    // guasto questo percorso resterebbe affermato e non provato — l'avevo scritto
    // "coperto" quando non lo era.
    printf("tutto o niente quando fallisce la lettura\n");
    {
        const size_t  N = 3;
        psx_cache *   cc = psx_cache_new((uint64_t) slice * 3);
        psx_region *  rr = psx_region_new(slice * 8, slice, 8);
        psx_region *  regs[N];
        size_t        offs[N];
        uint32_t      lays[N], exps[N];
        enum psx_kind kinds[N];
        for (size_t i = 0; i < N; i++) {
            regs[i] = rr; offs[i] = i * slice; lays[i] = 0;
            exps[i] = (uint32_t) i; kinds[i] = PSX_GATE;
        }
        psx_debug_fail_next_reads(1);
        check(psx_cache_ensure_many(cc, s, N, regs, offs, lays, exps, kinds) == 0,
              "il blocco fallisce se una lettura non arriva");
        psx_debug_fail_next_reads(0);
        // Ora si chiedono slice DIVERSE. Il budget e' di tre slice esatte e due
        // sono rimaste residenti dal blocco fallito: per entrare, le nuove devono
        // sfrattarle. Se le due avessero conservato il pin, lo sfratto sarebbe
        // impossibile e questo blocco fallirebbe — ed e' esattamente cosi' che la
        // prima versione del test non distingueva niente: riusava le stesse slice,
        // che erano hit e non chiedevano spazio.
        size_t   offs2[N];
        uint32_t exps2[N];
        for (size_t i = 0; i < N; i++) {
            offs2[i] = (i + 3) * slice;
            exps2[i] = (uint32_t) (i + 3);
        }
        check(psx_cache_ensure_many(cc, s, N, regs, offs2, lays, exps2, kinds) == 1,
              "nessun pin sopravvive a una lettura fallita");
        for (size_t i = 0; i < N; i++) {
            psx_cache_unpin(cc, rr, offs2[i]);
        }
        psx_cache_free(cc);
        psx_region_free(rr);
    }

    // --- emetti e continua ---------------------------------------------------
    //
    // Il percorso che slega la distribuzione del lavoro dalla profondita' di coda:
    // si prende una slice alla volta, la lettura parte e il thread va avanti; si
    // aspetta tutto alla fine. Qui non si misura la velocita', si verifica che i
    // byte ci siano e che il contratto tenga — compreso il caso in cui una lettura
    // non arriva.
    printf("emetti e continua\n");
    {
        const size_t N = 6;
        psx_cache *  cc = psx_cache_new((uint64_t) slice * 16);
        psx_region * rr = psx_region_new(slice * 16, slice, 16);
        int avviate = 0, hit = 0;
        for (size_t i = 0; i < N; i++) {
            const int r = psx_cache_avvia(cc, s, rr, i * slice, 0,
                                          (uint32_t) (i / 3), (enum psx_kind) (i % 3));
            if (r == 2) { avviate++; } else if (r == 1) { hit++; }
            check(r != 0, "avvio riuscito");
        }
        check(avviate == (int) N, "tutte da leggere: la cache era vuota");
        check(psx_cache_attendi(cc, s) == 1, "tutte le letture sono arrivate");
        bool uguali = true;
        std::vector<unsigned char> una(slice, 0);
        for (size_t i = 0; i < N; i++) {
            const uint32_t x = (uint32_t) (i / 3);
            const enum psx_kind k = (enum psx_kind) (i % 3);
            psx_read_slice(s, 0, x, k, una.data());
            if (memcmp((unsigned char *) psx_region_base(rr) + i * slice, una.data(),
                       psx_slice_read_bytes(s, 0, x, k)) != 0) {
                uguali = false;
            }
        }
        check(uguali, "i byte corrispondono al GGUF, slice per slice");
        // seconda passata: tutte hit, nessuna lettura da attendere
        for (size_t i = 0; i < N; i++) {
            psx_cache_unpin(cc, rr, i * slice);
        }
        int hit2 = 0;
        for (size_t i = 0; i < N; i++) {
            if (psx_cache_avvia(cc, s, rr, i * slice, 0,
                                (uint32_t) (i / 3), (enum psx_kind) (i % 3)) == 1) {
                hit2++;
            }
        }
        check(hit2 == (int) N, "al secondo giro sono tutte hit");
        check(psx_cache_attendi(cc, s) == 1, "attendi senza niente in volo e' innocuo");
        for (size_t i = 0; i < N; i++) {
            psx_cache_unpin(cc, rr, i * slice);
        }
        psx_cache_free(cc);
        psx_region_free(rr);
    }

    // Una lettura che non arriva deve essere riportata, e la slice non deve restare
    // annunciata come pronta: byte sbagliati in silenzio sono il guasto peggiore.
    printf("emetti e continua: lettura fallita\n");
    {
        psx_cache *  cc = psx_cache_new((uint64_t) slice * 8);
        psx_region * rr = psx_region_new(slice * 8, slice, 8);
        check(psx_cache_avvia(cc, s, rr, 0, 0, 0, PSX_GATE) == 2, "prima avviata");
        check(psx_cache_avvia(cc, s, rr, slice, 0, 1, PSX_GATE) == 2, "seconda avviata");
        psx_debug_fail_next_reads(1);
        const int esito = psx_cache_attendi(cc, s);
        psx_debug_fail_next_reads(0);
        check(esito == 0, "una lettura mancata viene riportata");
        psx_cache_stats st;
        psx_cache_get_stats(cc, &st);
        check(st.bytes_resident <= (uint64_t) slice * 2,
              "la slice fallita non resta contata come residente");
        psx_cache_free(cc);
        psx_region_free(rr);
    }

    // Molte letture sospese insieme: e' la scala a cui il bug si manifestava.
    //
    // Con un `std::vector` per le sospese, il primo ingrandimento spostava gli
    // OVERLAPPED mentre le letture erano in volo, e il kernel restava con puntatori
    // pendenti. Su Windows si vedeva come "impossibile rendere residenti gli expert"
    // con il budget quasi vuoto; su Linux, dove le letture sono sincrone, non si
    // vedeva per niente — ed e' proprio il limite che avevo dichiarato prima di
    // scrivere il codice, dentro il quale e' poi finito il bug.
    //
    // Qui si verifica la contabilita' a quella scala: su Linux non riproduce il
    // guasto, su Windows si'.
    printf("molte letture sospese\n");
    {
        const size_t N = 40;
        psx_cache *  cc = psx_cache_new((uint64_t) slice * (N + 4));
        psx_region * rr = psx_region_new(slice * (N + 4), slice, N + 4);
        size_t avviate = 0;
        for (size_t i = 0; i < N; i++) {
            const uint32_t lay = (uint32_t) (i / 24);
            const uint32_t x   = (uint32_t) ((i % 24) / 3);
            const enum psx_kind k = (enum psx_kind) (i % 3);
            const int r = psx_cache_avvia(cc, s, rr, i * slice, lay, x, k);
            if (r == 0) { break; }
            if (r == 2) { avviate++; }
        }
        check(avviate == N, "quaranta letture avviate senza aspettarne nessuna");
        check(psx_cache_attendi(cc, s) == 1, "arrivano tutte");
        bool uguali = true;
        std::vector<unsigned char> una(slice, 0);
        for (size_t i = 0; i < N; i++) {
            const uint32_t lay = (uint32_t) (i / 24);
            const uint32_t x   = (uint32_t) ((i % 24) / 3);
            const enum psx_kind k = (enum psx_kind) (i % 3);
            psx_read_slice(s, lay, x, k, una.data());
            if (memcmp((unsigned char *) psx_region_base(rr) + i * slice, una.data(),
                       psx_slice_read_bytes(s, lay, x, k)) != 0) {
                uguali = false;
            }
        }
        check(uguali, "i byte di tutte e quaranta sono quelli del GGUF");
        psx_cache_stats st;
        psx_cache_get_stats(cc, &st);
        check(st.letture >= N, "le letture sono contate tutte");
        for (size_t i = 0; i < N; i++) {
            psx_cache_unpin(cc, rr, i * slice);
        }
        psx_cache_free(cc);
        psx_region_free(rr);
    }

    printf("budget minimo\n");
    {
        const uint64_t min8 = psx_cache_min_budget(s, 8);
        const uint64_t min1 = psx_cache_min_budget(s, 1);
        check(min1 > 0, "minimo per un expert");
        check(min8 == min1 * 8, "scala col numero di expert per operazione");
    }

    psx_cache_free(c);
    psx_region_free(r);
    psx_close(s);

    printf("\n%s\n", fallimenti == 0 ? "TUTTO OK" : "CI SONO FALLIMENTI");
    return fallimenti == 0 ? 0 : 1;
}
