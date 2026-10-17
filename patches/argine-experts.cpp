// Argine — buffer type per gli expert MoE serviti da disco.
//
// Registra in ggml un tipo di buffer che prende possesso dei tensori
// `*_exps.weight` e li serve dall'expert store con letture esplicite, invece di
// lasciarli alla memoria mappata.
//
// Misurato su i5-1340P, 15.72 GiB, Qwen3-30B-A3B Q4_K_M, stesso disco:
//
//   memoria mappata, come fa llama.cpp oggi   562 MB/s in generazione
//   letture esplicite allineate, senza cache  4264 MB/s
//
// Non e' il disco a essere lento: e' il percorso di page fault. A un thread,
// sulla stessa slice e con cache fredda in entrambi i casi, mmap rende
// 310 MB/s e ReadFile 1324, con la latenza che passa da 2.96 ms a 0.69.
//
// Configurazione, per variabili d'ambiente, per non toccare il parsing degli
// argomenti di llama.cpp:
//
//   ARGINE_EXPERT_STORE    percorso del file .psxs (obbligatorio per attivarsi)
//   ARGINE_EXPERT_BUDGET   byte di RAM per gli expert (es. 6G); 0 = nessun tetto
//   ARGINE_EXPERT_DIRECT   1 (default) scavalca la cache dell'OS, 0 la usa
//   ARGINE_EXPERT_VERBOSE  1 stampa le statistiche della cache a fine grafo
//
// Senza ARGINE_EXPERT_STORE il buffer type non si registra e llama.cpp si
// comporta esattamente come prima.
//
// Come si innesta
// ---------------
// `compute_forward` NON calcola: legge gli id del router, garantisce le slice
// degli expert scelti, sincronizza i thread e poi ritorna `false`, cosi' ggml
// esegue il suo kernel standard sulla memoria che abbiamo appena riempito. Il
// vantaggio e' che non c'e' nessun kernel duplicato da mantenere allineato con
// upstream.
//
// I pin si rilasciano all'inizio dell'operazione successiva, non dopo il
// calcolo: nel kernel di ggml un thread puo' uscire prima degli altri, quindi
// una barriera dopo il calcolo sarebbe un deadlock. I nodi del grafo sono
// rigorosamente sequenziali (tutti i thread stanno sullo stesso nodo), quindi
// rilasciare i pin del nodo N all'inizio del nodo N+1 e' sicuro.

#include "argine-experts.h"
#include "argine-store.h"
#include "argine-research.h"

#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "traits.h"

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <atomic>
#include <string>
#include <unordered_map>
#include <vector>

// Numero massimo di expert per layer che sappiamo gestire in una bitmap su
// stack. Qwen3-30B-A3B ne ha 128; il tetto evita allocazioni nel percorso caldo.
#define PSX_MAX_EXPERTS 512

namespace {

// --------------------------------------------------------------- config ----

uint64_t parse_size(const char * s) {
    if (!s || !*s) {
        return 0;
    }
    char *   fine = nullptr;
    double   v    = strtod(s, &fine);
    uint64_t mult = 1;
    if (fine && *fine) {
        switch (*fine) {
            case 'k': case 'K': mult = 1ull << 10; break;
            case 'm': case 'M': mult = 1ull << 20; break;
            case 'g': case 'G': mult = 1ull << 30; break;
            case 't': case 'T': mult = 1ull << 40; break;
            default: break;
        }
    }
    return (uint64_t) (v * (double) mult);
}

// Legge una variabile ARGINE_*, accettando anche il vecchio nome PICOSTAR_*.
//
// Il progetto si chiamava PicoStar16. Se il lato Python mandasse
// ARGINE_EXPERT_STORE e il binario leggesse solo PICOSTAR_EXPERT_STORE (o il
// contrario, con un eseguibile vecchio in una cartella dimenticata), il buffer
// type semplicemente non si registrerebbe: nessun errore, nessun avviso, e gli
// expert tornerebbero su mmap. E' esattamente il guasto silenzioso che ha gia'
// reso inutile la prima misura dello store, quando `--no-repack` lo spegneva
// senza dirlo. Accettare entrambi i nomi costa otto righe e toglie di mezzo una
// classe intera di misure sbagliate.
const char * env_due_nomi(const char * nome) {
    const char * v = getenv(nome);
    if (v && *v) {
        return v;
    }
    if (strncmp(nome, "ARGINE_", 7) == 0) {
        char vecchio[128];
        snprintf(vecchio, sizeof(vecchio), "PICOSTAR_%s", nome + 7);
        v = getenv(vecchio);
        if (v && *v) {
            fprintf(stderr, "argine: uso %s (nome vecchio); rinominala in %s\n",
                    vecchio, nome);
            return v;
        }
    }
    return nullptr;
}

bool env_flag(const char * nome, bool def) {
    const char * v = env_due_nomi(nome);
    if (!v || !*v) {
        return def;
    }
    return !(v[0] == '0' || v[0] == 'n' || v[0] == 'N' || v[0] == 'f' || v[0] == 'F');
}

// ---------------------------------------------------- identita' tensore ----

struct TensorId {
    bool          valido = false;
    uint32_t      layer  = 0;
    enum psx_kind kind   = PSX_GATE;
};

// "blk.12.ffn_gate_exps.weight" -> layer 12, kind gate
TensorId identifica(const char * nome) {
    TensorId id;
    if (!nome) {
        return id;
    }
    const char * b = strstr(nome, "blk.");
    if (!b) {
        return id;
    }
    char *        fine  = nullptr;
    unsigned long layer = strtoul(b + 4, &fine, 10);
    if (!fine || *fine != '.') {
        return id;
    }
    if (strstr(nome, ".ffn_gate_exps.weight")) {
        id.kind = PSX_GATE;
    } else if (strstr(nome, ".ffn_up_exps.weight")) {
        id.kind = PSX_UP;
    } else if (strstr(nome, ".ffn_down_exps.weight")) {
        id.kind = PSX_DOWN;
    } else {
        return id;
    }
    id.layer  = (uint32_t) layer;
    id.valido = true;
    return id;
}

// --------------------------------------------------------- modo lettura ----

// Quale strato di lettura usare per portare le slice in memoria.
//
// Esiste per una ragione di misura, non di configurazione: le tre versioni sono
// state confrontate in tre sessioni diverse, con tre binari diversi, in ore
// diverse. Il risultato (+7-11% per l'asincrono) viene da un run per punto
// contro un rumore misurato dell'1-4%: il segno e' solido, la cifra no, e
// soprattutto non e' riproducibile da chi legge il README, perche' due dei tre
// binari non esistono piu'.
//
// Con il modo selezionabile a runtime il confronto si fa con lo STESSO binario,
// nello stesso pomeriggio, alternando i modi a ogni budget. Quello che prima era
// un'affermazione d'autore diventa un comando.
enum ModoLettura {
    // Una slice per volta, aspettandola: la prima versione. La profondita' di
    // coda e' 1 per thread.
    LETTURA_SINCRONA = 0,
    // Blocchi di n slice con le letture in volo insieme, aspettando la fine del
    // blocco prima del prossimo: la seconda versione.
    LETTURA_BLOCCO = 1,
    // Emetti e continua: la lettura parte, il thread prende la slice dopo, si
    // aspetta tutto alla barriera. La terza, e la piu' veloce misurata.
    LETTURA_ASINCRONA = 2,
};

#define ARGINE_BLOCCO_MAX 8u

// "sincrono" | "blocco" | "blocco:4" | "asincrono" (sinonimi: sync, async).
// Qualsiasi altra cosa e' un errore detto, non un silenzio: una variabile
// d'ambiente scritta male che ripiega sul default falsa la misura e non lascia
// traccia.
inline ModoLettura leggi_modo(const char * v, uint32_t * blocco_out) {
    *blocco_out = 2;
    if (!v || !*v) {
        return LETTURA_ASINCRONA;
    }
    if (strncmp(v, "sincrono", 8) == 0 || strncmp(v, "sync", 4) == 0) {
        return LETTURA_SINCRONA;
    }
    if (strncmp(v, "blocco", 6) == 0) {
        const char * due = strchr(v, ':');
        if (due && due[1]) {
            const long n = atol(due + 1);
            if (n >= 1 && n <= (long) ARGINE_BLOCCO_MAX) {
                *blocco_out = (uint32_t) n;
            } else {
                fprintf(stderr,
                        "argine: ARGINE_EXPERT_READ=%s: il blocco va fra 1 e %u, "
                        "si usa %u\n",
                        v, ARGINE_BLOCCO_MAX, *blocco_out);
            }
        }
        return LETTURA_BLOCCO;
    }
    if (strncmp(v, "asincrono", 9) == 0 || strncmp(v, "async", 5) == 0) {
        return LETTURA_ASINCRONA;
    }
    fprintf(stderr,
            "argine: ARGINE_EXPERT_READ=%s non riconosciuto (sincrono | blocco[:n] | "
            "asincrono): si usa asincrono\n",
            v);
    return LETTURA_ASINCRONA;
}

inline const char * nome_modo(ModoLettura m) {
    switch (m) {
        case LETTURA_SINCRONA:  return "sincrono";
        case LETTURA_BLOCCO:    return "blocco";
        default:                return "asincrono";
    }
}

// ------------------------------------------------------------- contesto ----

struct Contesto {
    psx_store * store  = nullptr;
    psx_cache * cache  = nullptr;
    uint64_t    budget = 0;
    bool        direct = true;
    bool        verbose = false;
    bool        attivo = false;
    ModoLettura modo   = LETTURA_ASINCRONA;
    uint32_t    blocco = 2;
    // messaggi da dare una sola volta: in mezzo alla generazione un avviso
    // ripetuto per ogni token e' rumore che nasconde il primo
    bool        detto_disallineato = false;
    bool        detto_fuso = false;
    std::mutex  mtx;

    // pin dell'operazione precedente, da rilasciare all'inizio della prossima
    struct Pin {
        psx_region * region;
        size_t       offset;
    };
    // I pin sono PER THREAD: ogni thread sblocca quelli che ha preso lui.
    //
    // Prima erano in un vettore condiviso protetto da mutex, e il thread 0 li
    // sblocchava tutti all'inizio del nodo successivo — il che richiedeva una
    // barriera in piu' perche' nessuno cominciasse a caricare prima che lo
    // sblocco fosse finito. Con le liste locali non c'e' ne' il mutex ne' la
    // barriera: nessuno tocca i pin di un altro.
    static std::vector<Pin> & miei_pin() {
        static thread_local std::vector<Pin> v;
        return v;
    }
    bool             detto_in_uso = false;

    // Dove vive ogni tensore degli expert, per layer e per tipo.
    //
    // Serve a caricare gate, up e down di un layer in un colpo solo. I tre
    // usano gli **stessi** expert — lo decide il router una volta per layer —
    // ma ggml li visita in tre nodi separati, e ogni nodo ha la sua barriera.
    // Caricandoli uno per volta si pagano 144 round sincronizzati per token
    // invece di 48, e la misura dice che li' sta il collo di bottiglia della
    // generazione: alzando la cache da 6 a 8 GiB il traffico e' sceso del 32% e
    // la velocita' non e' cambiata di un punto. Quando meno lavoro nello stesso
    // tempo non produce piu' velocita', il limite non e' la banda: sono le
    // attese in fila.
    struct Posizione {
        psx_region * region = nullptr;
        size_t       t_off  = 0;
        size_t       slab   = 0;
        bool         nota   = false;
    };
    std::unordered_map<uint64_t, Posizione> posizioni;  // chiave: layer * 8 + kind

    static uint64_t chiave(uint32_t layer, enum psx_kind kind) {
        return (uint64_t) layer * 8 + (uint64_t) kind;
    }
    bool             detto_troppi_expert = false;
    // Argine2 research trace state. Touched only by thread 0 of each node, and
    // nodes run one at a time, so no lock is needed.
    uint64_t arx_graph       = 0;   // forward passes seen (one per ubatch)
    int64_t  arx_last_layer  = -1;  // layer of the previous MoE node
    uint64_t arx_route_graph = UINT64_MAX;
    int64_t  arx_route_layer = -1;
    // quanto tempo passa dentro il nostro compute_forward, e quanti nodi:
    // se e' una frazione piccola del tempo per token, il collo di bottiglia non
    // e' qui e cercarlo qui e' tempo perso
    std::atomic<uint64_t> ns_nostro{ 0 };
    std::atomic<uint64_t> nodi{ 0 };
    std::chrono::steady_clock::time_point ultimo_resoconto{};
    // Istante del primo nodo MoE servito: serve un riferimento di tempo di
    // PARETE, perche' tutti gli altri contatori sono somme sui thread. Il
    // rapporto fra i due da' la concorrenza media, che e' il numero che dice
    // quante letture sono in volo insieme — e quindi se il disco e' lento o se
    // gli stiamo semplicemente chiedendo una cosa alla volta.
    std::chrono::steady_clock::time_point inizio{};
    // indice condiviso per la distribuzione dinamica delle letture
    std::atomic<uint32_t> prossima_slice{ 0 };

    // Contiene un mutex, quindi non e' copiabile ne' spostabile: si inizializza
    // sul posto, una volta sola, alla prima chiamata.
    void inizializza() {
        const char * path = env_due_nomi("ARGINE_EXPERT_STORE");
        if (!path || !*path) {
            return;  // non configurato: il buffer type non si registra
        }
        Contesto & c = *this;
        {
            c.direct  = env_flag("ARGINE_EXPERT_DIRECT", true);
            psx_debug_set_pretouch(env_flag("ARGINE_EXPERT_PRETOUCH", false) ? 1 : 0);
            c.verbose = env_flag("ARGINE_EXPERT_VERBOSE", false);
            c.modo    = leggi_modo(env_due_nomi("ARGINE_EXPERT_READ"), &c.blocco);
            c.budget  = parse_size(env_due_nomi("ARGINE_EXPERT_BUDGET"));
            c.store   = psx_open(path, c.direct ? 1 : 0);
            if (const char * err = psx_error(c.store)) {
                GGML_LOG_ERROR("argine: expert store non utilizzabile: %s\n", err);
                psx_close(c.store);
                c.store = nullptr;
                return;
            }
            {
                // ARGINE_EXPERT_SHARDS: 0 o assente = automatico
                const char * sh = env_due_nomi("ARGINE_EXPERT_SHARDS");
                const unsigned n = sh && *sh ? (unsigned) atoi(sh) : 0;
                c.cache = psx_cache_new_n(c.budget, n);
            }
            c.attivo = true;
            if (c.direct && !psx_direct(c.store)) {
                // chiesta ma non concessa: succede su certi filesystem e certi
                // dispositivi. Non e' fatale, ma cambia la natura del budget e
                // va detto, non subito.
                GGML_LOG_WARN(
                    "argine: questo dispositivo non accetta letture senza cache: si usa la "
                    "cache dell'OS. Il budget resta un tetto sulla nostra cache, ma la RAM "
                    "totale sara' maggiore perche' l'OS ne tiene una copia sua, e quella non "
                    "la controlliamo.\n");
                c.direct = false;
            }
            char budget_txt[40];
            if (!c.budget) {
                snprintf(budget_txt, sizeof(budget_txt), "illimitato");
            } else if (c.budget >= (1ull << 20)) {
                snprintf(budget_txt, sizeof(budget_txt), "%" PRIu64 " MiB", c.budget >> 20);
            } else {
                // "budget 0 MiB" per 200 KiB e' una riga che inganna: sembra
                // "nessun budget" quando invece e' un budget strettissimo
                snprintf(budget_txt, sizeof(budget_txt), "%" PRIu64 " KiB", c.budget >> 10);
            }
            // Su stderr e non via GGML_LOG_INFO: senza -v llama-server filtra i
            // log di caricamento, e allora questa riga sparisce proprio quando
            // serve. E' la sola prova che lo store si sia aperto, e uno
            // strumento che non puo' provare cio' che afferma e' inutile.
            char modo_txt[32];
            if (c.modo == LETTURA_BLOCCO) {
                snprintf(modo_txt, sizeof(modo_txt), "blocco:%u", c.blocco);
            } else {
                snprintf(modo_txt, sizeof(modo_txt), "%s", nome_modo(c.modo));
            }
            // Il modo di lettura va in questa riga perche' e' l'unica che
            // compare in ogni log: una misura archiviata senza sapere quale
            // strato di lettura l'ha prodotta non e' confrontabile con niente.
            fprintf(stderr,
                    "argine: expert store aperto: %s, %u layer x %u expert, budget %s, "
                    "%u settori di cache, cache dell'OS %s, lettura %s\n",
                    path, psx_n_layers(c.store), psx_n_experts(c.store), budget_txt,
                    psx_cache_shards(c.cache), c.direct ? "scavalcata" : "attiva", modo_txt);
            if (arx_enabled()) {
                // Argine2: the configuration travels inside the trace, so that a
                // raw log can never be separated from the conditions that made it.
                char num[32];
                arx_config("instrumentation", "argine2-research-v2");
                arx_config("store", path);
                snprintf(num, sizeof(num), "%" PRIu64, c.budget);
                arx_config("budget_bytes", num);
                arx_config("read_mode", modo_txt);
                arx_config("direct_io", c.direct ? "1" : "0");
                snprintf(num, sizeof(num), "%u", psx_cache_shards(c.cache));
                arx_config("cache_shards", num);
                snprintf(num, sizeof(num), "%u", psx_n_layers(c.store));
                arx_config("store_layers", num);
                snprintf(num, sizeof(num), "%u", psx_n_experts(c.store));
                arx_config("store_experts", num);
                arx_config("pretouch", env_flag("ARGINE_EXPERT_PRETOUCH", false) ? "1" : "0");
                arx_flush_thread();
                arx_flush_file(0);
            }
            if (c.budget) {
                const uint64_t minimo = psx_cache_min_budget(c.store, psx_n_experts(c.store));
                if (c.budget < minimo) {
                    GGML_LOG_WARN(
                        "argine: budget %" PRIu64 " MiB sotto il minimo teorico %" PRIu64
                        " MiB (tutti gli expert di un layer). Durante l'elaborazione del prompt "
                        "un ubatch puo' toccarli tutti: se succede, la generazione si ferma con "
                        "un errore invece di sforare il tetto.\n",
                        c.budget >> 20, minimo >> 20);
                }
            }
        }
    }

    // A fine processo, con ARGINE_EXPERT_VERBOSE=1, si stampano le statistiche
    // della cache: sono il modo di sapere se il budget e' stato rispettato e
    // quanto e' costato in miss.
    // Il resoconto va stampato DURANTE l'esecuzione, non alla chiusura.
    //
    // Nella prima versione lo stampava il distruttore: su Windows Argine
    // termina llama-server invece di lasciarlo uscire, i distruttori statici
    // non vengono eseguiti, e quelle righe non sono mai comparse in nessuna
    // misura. Uno strumento che funziona solo se il processo muore nel modo
    // giusto non e' uno strumento.
    // Quanti tensori di expert ha preso lo store, su quanti ce ne sono.
    //
    // Serve perche' fra "in uso" e "in uso per tutto" c'e' una differenza che
    // non si vede: se una parte dei tensori finisce sul percorso normale — un
    // `ffn_gate_up_exps` fuso, una slice non allineata di un solo tipo, un
    // conteggio di expert oltre il limite della build — la riga "IN USO" compare
    // comunque, la cache riporta hit e miss plausibili, il budget risulta
    // rispettato, e nel frattempo una fetta degli expert sta in memoria mappata.
    // Il tetto di RAM, che e' la promessa centrale del progetto, sarebbe falso di
    // quella fetta, e nessuna riga lo direbbe.
    //
    // Il conto e' banale: tre tensori per layer, e `posizioni` sa quali sono
    // passati per il nostro buffer.
    void copertura(uint32_t * presi, uint32_t * attesi) const {
        *attesi = psx_n_layers(store) * PSX_N_KINDS;
        uint32_t n = 0;
        for (const auto & kv : posizioni) {
            if (kv.second.nota) {
                n++;
            }
        }
        *presi = n;
    }

    void resoconto(const char * quando) {
        if (!attivo) {
            return;
        }
        psx_cache_stats st;
        psx_cache_get_stats(cache, &st);
        const uint64_t tot = st.hits + st.misses;
        fprintf(stderr,
                "argine: expert cache (%s): %llu hit, %llu miss (%.1f%% hit), %llu "
                "sfratti, %llu MiB letti dallo store, %llu/%llu MiB residenti\n",
                quando, (unsigned long long) st.hits, (unsigned long long) st.misses,
                tot ? 100.0 * (double) st.hits / (double) tot : 0.0,
                (unsigned long long) st.evictions,
                (unsigned long long) (st.bytes_read >> 20),
                (unsigned long long) (st.bytes_resident >> 20),
                (unsigned long long) (st.budget >> 20));

        // Dove se ne va il tempo. I millisecondi sono sommati su tutti i
        // thread, quindi il totale supera il tempo reale: contano i rapporti.
        const double lettura = (double) st.ns_read / 1e6;
        const double commit  = (double) st.ns_commit / 1e6;
        const double dec     = (double) st.ns_decommit / 1e6;
        const double att     = (double) st.ns_attesa / 1e6;
        const double lock    = (double) st.ns_lock / 1e6;
        const double fault   = (double) st.ns_fault / 1e6;
        const double nostro  = (double) ns_nostro.load() / 1e6;
        const double somma   = lettura + commit + dec + att + lock + fault;
        fprintf(stderr,
                "argine: tempo (ms, somma sui thread): lettura %.0f (%.0f%%), pagine "
                "nuove %.0f (%.0f%%), commit %.0f (%.0f%%), decommit %.0f (%.0f%%), "
                "attesa %.0f (%.0f%%), lock %.0f (%.0f%%)\n",
                lettura, somma ? 100.0 * lettura / somma : 0.0,
                fault, somma ? 100.0 * fault / somma : 0.0,
                commit, somma ? 100.0 * commit / somma : 0.0,
                dec, somma ? 100.0 * dec / somma : 0.0,
                att, somma ? 100.0 * att / somma : 0.0,
                lock, somma ? 100.0 * lock / somma : 0.0);

        const uint64_t n_nodi = nodi.load();
        // Il tempo dentro il nostro codice MENO quello speso davvero a leggere,
        // impegnare e aspettare slice e' tempo di barriera: thread fermi ad
        // aspettarsi fra loro. E' il numero che dice se il collo di bottiglia e'
        // l'I/O o la sincronizzazione.
        const double barriera = nostro > somma ? nostro - somma : 0.0;
        fprintf(stderr,
                "argine: dentro il nostro codice %.0f ms su %llu nodi MoE (%.2f ms per "
                "nodo); di cui I/O %.0f ms (%.0f%%) e attesa fra thread %.0f ms (%.0f%%)\n",
                nostro, (unsigned long long) n_nodi,
                n_nodi ? nostro / (double) n_nodi : 0.0,
                somma, nostro ? 100.0 * somma / nostro : 0.0,
                barriera, nostro ? 100.0 * barriera / nostro : 0.0);

        // Profondita' di coda e banda, le due cose che i contatori sommati sui
        // thread da soli non possono dire.
        //
        // `lettura` e' la somma su tutti i thread del tempo passato dentro una
        // read; `parete` e' il tempo vero trascorso. Il loro rapporto e' quanti
        // thread stanno leggendo nello stesso istante, in media. Serve perche'
        // il dispositivo rende in funzione di quante richieste ha in volo: su
        // questa macchina `diskbench` misura 1324 MB/s con una sola e 4470 con
        // trentadue, quindi sapere di essere a 1 oppure a 4 cambia dove va il
        // lavoro — e nessuno dei numeri qui sopra lo diceva.
        //
        // La banda e' quella aggregata vista dal processo: byte letti diviso
        // tempo di parete, confrontabile con i numeri di `diskbench` e con il
        // "lettura disco" del benchmark.
        const double parete = inizio.time_since_epoch().count()
            ? (double) std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - inizio).count()
            : 0.0;
        if (parete > 0.0) {
            const double mib = (double) st.bytes_read / (1024.0 * 1024.0);
            // Con le letture sovrapposte ns_read misura il parete di un blocco, non
            // la somma delle letture che conteneva: senza questo fattore la coda
            // risulta piu' piccola del vero, ed e' esattamente l'errore che ha reso
            // illeggibile la prima misura dei blocchi.
            const double per_blocco = st.blocchi
                ? (double) st.letture / (double) st.blocchi : 1.0;
            fprintf(stderr,
                    "argine: parete %.0f ms; profondita' di coda media %.2f "
                    "(letture contemporanee), banda %.0f MB/s aggregati, "
                    "%.0f MB/s per blocco di lettura\n",
                    parete, (lettura / parete) * per_blocco,
                    mib / (parete / 1000.0),
                    lettura > 0.0 ? mib / (lettura / 1000.0) : 0.0);
            if (st.blocchi) {
                fprintf(stderr,
                        "argine: %llu attese, %.2f letture in volo per attesa; ns_read "
                        "e' il parete dell'attesa, non la somma delle letture: la coda "
                        "qui sopra ne tiene conto\n",
                        (unsigned long long) st.blocchi, per_blocco);
            }
        }
        if (st.misses) {
            fprintf(stderr,
                    "argine: per miss: %.3f ms di lettura, %.3f ms di pagine nuove, "
                    "%.3f ms di commit\n",
                    lettura / (double) st.misses, fault / (double) st.misses,
                    commit / (double) st.misses);
        }
        // La copertura va DENTRO questo blocco, dopo la riga "expert cache": il
        // lettore Python del resoconto parte da quella riga, e una riga stampata
        // prima finirebbe fuori dal blocco che lo sweep mostra. Un avviso che
        // compare solo in un percorso di stampa su due non e' un avviso.
        {
            uint32_t presi = 0, attesi = 0;
            std::lock_guard<std::mutex> lk(mtx);
            copertura(&presi, &attesi);
            fprintf(stderr, "argine: copertura: %u tensori di expert su %u serviti dallo store\n",
                    presi, attesi);
            if (attesi && presi < attesi) {
                fprintf(stderr,
                        "argine: ATTENZIONE: i restanti %u stanno sulla memoria mappata, "
                        "quindi il budget NON e' un tetto sulla RAM totale e questa misura "
                        "non descrive lo store da solo.\n",
                        attesi - presi);
            }
        }
        fflush(stderr);
    }

    ~Contesto() {
        if (attivo && verbose) {
            resoconto("fine");
        }
    }

    static Contesto & instance() {
        static Contesto c;
        static std::once_flag una_volta;
        std::call_once(una_volta, [] { c.inizializza(); });
        return c;
    }
};

// -------------------------------------------------------------- buffer -----

struct Buffer {
    psx_region * region = nullptr;
    size_t       bytes  = 0;
};

ggml_backend_buffer_type_t argine_buffer_type();

const char * buffer_type_name(ggml_backend_buffer_type_t) {
    return "CPU_ARGINE_EXPERTS";
}

size_t buffer_type_alignment(ggml_backend_buffer_type_t) {
    // ogni tensore comincia a un multiplo della pagina: e' cio' che permette
    // alle singole slice di essere allineate e quindi leggibili senza cache
    return 4096;
}

bool buffer_type_is_host(ggml_backend_buffer_type_t) {
    // false come per repack: cosi' ggml considera questa memoria opaca e non
    // lascia che altri backend la leggano direttamente. Sarebbe un disastro,
    // perche' la maggior parte delle pagine non e' impegnata.
    return false;
}

void buffer_free(ggml_backend_buffer_t buffer) {
    auto * b = (Buffer *) buffer->context;
    if (b) {
        psx_region_free(b->region);
        delete b;
    }
}

void * buffer_get_base(ggml_backend_buffer_t buffer) {
    auto * b = (Buffer *) buffer->context;
    // ggml pretende un puntatore non nullo anche per un buffer vuoto
    return b && b->region ? psx_region_base(b->region) : (void *) 0x1000;
}

// Al caricamento i pesi arrivano qui. Non li scriviamo: sono gia' nello store,
// ed e' il motivo per cui il caricamento non deve leggere 16 GiB. Verifichiamo
// pero' che lo store corrisponda al modello, perche' uno store sbagliato non fa
// rumore: degrada il modello in silenzio.
void buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data,
                       size_t offset, size_t size) {

    Contesto & c  = Contesto::instance();
    const TensorId id = identifica(tensor->name);
    if (!c.attivo || !id.valido) {
        return;
    }
    if (offset != 0 || size != ggml_nbytes(tensor)) {
        // scrittura parziale: non la sappiamo gestire, e non dovrebbe avvenire
        GGML_ABORT("argine: scrittura parziale su %s (offset %zu, %zu byte): non supportata",
                   tensor->name, offset, size);
    }

    const uint32_t n_exp = psx_n_experts(c.store);
    const size_t   slab  = (size_t) tensor->nb[2];
    const psx_entry * e  = psx_get(c.store, id.layer, 0);
    if (!e) {
        GGML_ABORT("argine: lo store non contiene il layer %u (tensore %s)", id.layer,
                   tensor->name);
    }
    if ((size_t) tensor->ne[2] != (size_t) n_exp || slab != (size_t) e->size[id.kind]) {
        GGML_ABORT(
            "argine: lo store non corrisponde al modello. %s ha %" PRId64
            " expert da %zu byte, lo store ne dichiara %u da %u. Ricostruiscilo con "
            "`argine store build` sul GGUF che stai caricando.",
            tensor->name, tensor->ne[2], slab, n_exp, e->size[id.kind]);
    }

    // registriamo dove vive: servira' a caricare i tre tensori di un layer insieme
    {
        auto * b = (Buffer *) buffer->context;
        if (b && b->region) {
            auto * base = (unsigned char *) psx_region_base(b->region);
            Contesto::Posizione pos;
            pos.region = b->region;
            pos.t_off  = (size_t) ((unsigned char *) tensor->data - base);
            pos.slab   = slab;
            pos.nota   = true;
            std::lock_guard<std::mutex> lk2(c.mtx);
            c.posizioni[Contesto::chiave(id.layer, id.kind)] = pos;
        }
    }

    // confronto dei byte, una volta per tipo di tensore: basta a scoprire uno
    // store costruito da un'altra quantizzazione o da un altro modello, e costa
    // una lettura di meno di un megabyte
    static bool provato[PSX_N_KINDS] = { false, false, false };
    std::lock_guard<std::mutex> lk(c.mtx);
    if (!provato[id.kind]) {
        provato[id.kind] = true;
        // il buffer di confronto va allineato: con la cache dell'OS scavalcata
        // una destinazione non allineata fa fallire la lettura con EINVAL, ed e'
        // esattamente l'errore in cui sono caduto usando un std::vector qui
        const size_t n = psx_slice_read_bytes(c.store, id.layer, 0, id.kind);
        void *       mio = psx_alloc_aligned(c.store, n);
        if (!mio) {
            GGML_ABORT("argine: memoria insufficiente per verificare lo store");
        }
        const bool letto = psx_read_slice(c.store, id.layer, 0, id.kind, mio) != 0;
        const bool uguale = letto && memcmp(mio, data, slab) == 0;
        const std::string errore = letto ? "" : (psx_error(c.store) ? psx_error(c.store) : "?");
        psx_free_aligned(mio, n);
        if (!letto) {
            GGML_ABORT("argine: lo store non si legge: %s", errore.c_str());
        }
        if (!uguale) {
            GGML_ABORT(
                "argine: i pesi dello store non corrispondono a quelli del GGUF (%s, "
                "expert 0). Lo store e' di un altro modello o di un'altra quantizzazione: "
                "ricostruiscilo.",
                tensor->name);
        }
    }
}

// I pesi non si azzerano mai in pratica, ma `clear` non e' opzionale
// nell'interfaccia: meglio un no-op rumoroso che un puntatore nullo.
void buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
    GGML_LOG_DEBUG("argine: clear su un buffer di expert ignorato\n");
}

ggml_backend_buffer_t buffer_type_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    // Assegnata per nome, non per posizione. La lista posizionale che avevo
    // scritto prima era disallineata: ggml_backend_buffer_i ha set_tensor_2d e
    // get_tensor_2d fra get_tensor e cpy_tensor, che non avevo contato. Era
    // innocua solo perche' tutto quello che seguiva era nullptr, ma sarebbe
    // diventata un puntatore a funzione sbagliato alla prima aggiunta di un
    // campo in upstream.
    static const ggml_backend_buffer_i iface = [] {
        ggml_backend_buffer_i i = {};
        i.free_buffer = buffer_free;
        i.get_base    = buffer_get_base;
        i.set_tensor  = buffer_set_tensor;
        i.clear       = buffer_clear;
        return i;
    }();

    auto * b  = new Buffer();
    b->bytes  = size;
    if (size > 0) {
        // Riserva gli indirizzi, non la memoria: su 64 bit prendere 16 GiB di
        // spazio virtuale non costa niente, e il puntatore resta contiguo per
        // tutto il tensore, che e' l'unica cosa che ggml pretende. La RAM
        // occupata sara' solo quella delle slice in cache.
        b->region = psx_region_new(size, 0, 0);
        if (!b->region) {
            GGML_LOG_ERROR("argine: impossibile riservare %zu MiB di spazio virtuale\n",
                           size >> 20);
            delete b;
            return nullptr;
        }
    }
    // un buffer di dimensione zero e' legittimo: llama.cpp lo alloca per
    // interrogare il tipo durante la scelta del buffer per ogni peso
    return ggml_backend_buffer_init(buft, iface, b, size);
}

// ---------------------------------------------------------- compute ---------

class ExpertTraits : public ggml::cpu::tensor_traits {
    bool work_size(int, const ggml_tensor *, size_t &) override {
        return false;  // la dimensione di lavoro la decide il percorso standard
    }

    bool compute_forward(ggml_compute_params * params, ggml_tensor * op) override {
        if (op->op != GGML_OP_MUL_MAT_ID) {
            return false;
        }
        Contesto & c = Contesto::instance();
        if (!c.attivo) {
            return false;
        }
        ggml_tensor *       src0 = op->src[0];
        const ggml_tensor * ids  = op->src[2];
        const TensorId      id   = identifica(src0->name);
        if (!id.valido || !src0->buffer || src0->buffer->buft != argine_buffer_type()) {
            return false;
        }

        const auto t_nostro = std::chrono::steady_clock::now();
        auto * b     = (Buffer *) src0->buffer->context;
        auto * base  = (unsigned char *) psx_region_base(b->region);
        const size_t   t_off = (unsigned char *) src0->data - base;
        const size_t   slab  = (size_t) src0->nb[2];
        const uint32_t n_exp = (uint32_t) src0->ne[2];
        const int      ith   = params->ith;
        const int      nth   = params->nth;

        // Quali expert servono a questo nodo. Lo calcola OGNI thread invece di
        // farlo fare a uno e pubblicarlo: e' deterministico, in sola lettura, e
        // costa un paio di microsecondi contro una barriera in piu'.
        //
        // Durante la generazione sono 8, durante l'elaborazione del prompt un
        // ubatch puo' toccarne tutti: e' per questo che il budget si dimensiona
        // sul caso prompt.
        uint32_t da_leggere[PSX_MAX_EXPERTS];
        uint32_t n_serve = 0;
        {
            // bitmap su stack: n_expert e' 128 su Qwen3-30B-A3B, al massimo
            // PSX_MAX_EXPERTS. Niente allocazioni nel percorso caldo.
            unsigned char serve[PSX_MAX_EXPERTS] = { 0 };
            const uint32_t limite = n_exp < PSX_MAX_EXPERTS ? n_exp : PSX_MAX_EXPERTS;
            for (int64_t i1 = 0; i1 < ids->ne[1]; i1++) {
                for (int64_t i0 = 0; i0 < ids->ne[0]; i0++) {
                    const int32_t x = *(const int32_t *) ((const char *) ids->data +
                                                          i1 * ids->nb[1] + i0 * ids->nb[0]);
                    if (x >= 0 && (uint32_t) x < limite) {
                        serve[x] = 1;
                    }
                }
            }
            for (uint32_t x = 0; x < limite; x++) {
                if (serve[x]) {
                    da_leggere[n_serve++] = x;
                }
            }
        }

        if (ith == 0) {
            std::lock_guard<std::mutex> lk(c.mtx);
            if (!c.detto_in_uso) {
                c.detto_in_uso = true;
                // Aprire lo store e servire davvero un nodo sono due cose
                // diverse: --no-repack, per esempio, disabilita tutti gli extra
                // buffer type e lascia lo store aperto ma inutilizzato. Solo
                // questa riga prova che il percorso e' in uso.
                fprintf(stderr,
                        "argine: expert store IN USO: primo nodo MoE servito "
                        "(layer %u, %s, %u expert richiesti, %d thread)\n",
                        id.layer, src0->name, n_serve, nth);
            }
            c.prossima_slice.store(0, std::memory_order_relaxed);
        }

        // Argine2 research: router decision and forward-pass boundaries.
        // Observation only — nothing below changes which slices are loaded.
        const bool arx = arx_enabled() != 0;
        if (arx && ith == 0) {
            if ((int64_t) id.layer < c.arx_last_layer) {
                // the layer index went backwards: a new forward pass started.
                // Snapshot the cache at the boundary of the one that ended.
                psx_cache_stats st;
                psx_cache_get_stats(c.cache, &st);
                arx_snapshot(c.arx_graph, st.bytes_resident, st.budget, st.hits, st.misses,
                             st.evictions, st.bytes_read, st.violazioni);
                c.arx_graph++;
            }
            c.arx_last_layer = (int64_t) id.layer;
            if (c.arx_route_graph != c.arx_graph || c.arx_route_layer != (int64_t) id.layer) {
                c.arx_route_graph = c.arx_graph;
                c.arx_route_layer = (int64_t) id.layer;
                // ids: ne[0] = experts used per token, ne[1] = tokens in the ubatch
                static thread_local std::vector<int32_t> buf;
                const int64_t n_used = ids->ne[0];
                const int64_t n_tok  = ids->ne[1];
                buf.resize((size_t) (n_used * n_tok));
                for (int64_t i1 = 0; i1 < n_tok; i1++) {
                    for (int64_t i0 = 0; i0 < n_used; i0++) {
                        buf[(size_t) (i1 * n_used + i0)] = *(const int32_t *) (
                            (const char *) ids->data + i1 * ids->nb[1] + i0 * ids->nb[0]);
                    }
                }
                arx_route(c.arx_graph, id.layer, (uint32_t) n_tok, (uint32_t) n_used, buf.data());
            }
        }

        // I miei pin del nodo precedente: da qui possono essere sfrattati.
        // Nessun lock e nessuna barriera: tocco solo roba mia.
        {
            auto & precedenti = Contesto::miei_pin();
            for (const auto & p : precedenti) {
                psx_cache_unpin(c.cache, p.region, p.offset);
            }
            precedenti.clear();
        }

        // Una sola barriera prima delle letture, per l'azzeramento del
        // contatore condiviso della distribuzione dinamica.
        ggml_barrier(params->threadpool);

        // Tutti e tre i tensori del layer, non solo quello di questo nodo.
        //
        // gate, up e down usano gli stessi expert: il router decide una volta
        // per layer. Caricarli qui significa che i due nodi successivi trovano
        // tutto in cache, e i round sincronizzati per token scendono da 144 a
        // 48. Se il registro non conosce una posizione (tensore finito sul
        // percorso normale) quel tipo si salta e verra' caricato dal suo nodo.
        struct Dove {
            psx_region * region;
            size_t       t_off;
            size_t       slab;
            enum psx_kind kind;
        };
        Dove dove[PSX_N_KINDS];
        int  n_dove = 0;
        {
            std::lock_guard<std::mutex> lk(c.mtx);
            for (int k = 0; k < (int) PSX_N_KINDS; k++) {
                if ((enum psx_kind) k == id.kind) {
                    dove[n_dove++] = Dove{ b->region, t_off, slab, id.kind };
                    continue;
                }
                auto it = c.posizioni.find(Contesto::chiave(id.layer, (enum psx_kind) k));
                if (it != c.posizioni.end() && it->second.nota && it->second.region) {
                    dove[n_dove++] = Dove{ it->second.region, it->second.t_off,
                                           it->second.slab, (enum psx_kind) k };
                }
            }
        }

        // Le letture le fanno TUTTI i thread, una fetta per ciascuno.
        //
        // Nella prima versione le faceva il solo thread 0, in sequenza, mentre
        // gli altri aspettavano alla barriera: profondita' di coda 1. Il costo
        // di quell'errore e' misurato, e il diskbench lo aveva detto prima che
        // lo scrivessi: letture dirette da 896 KiB rendono 1324 MB/s a un
        // thread e 4518 a otto. In produzione si vedevano 432 MB/s; dopo la
        // correzione il prompt e' passato da 10.3 a 22.1 tok/s.
        // Distribuzione DINAMICA: chi finisce prima prende il prossimo, invece
        // di avere una fetta assegnata per indice.
        //
        // La misura dice che qui stava il grosso: il 65% del tempo nel nostro
        // codice erano thread fermi ad aspettarsi, contro il 35% di lavoro vero.
        // Con il 96% di hit, un hit costa quasi zero e un miss 1.47 ms: su un
        // nodo da 24 slice con un solo miss, la divisione per indice faceva
        // aspettare sette thread su otto per 1.47 ms interi. Parti uguali e
        // costo bimodale sono il caso peggiore possibile.
        const uint32_t totale = n_serve * (uint32_t) n_dove;
        bool           guaio  = false;
        auto & pin_miei = Contesto::miei_pin();
        pin_miei.reserve(PSX_N_KINDS * 16);
        // Ogni thread prende un BLOCCO di slice e le legge tutte in volo insieme,
        // invece di una alla volta aspettando ciascuna.
        //
        // La ragione e' misurata, e per un po' era stata misurata male: avevo
        // archiviato questa idea con "0,75 miss per nodo, i miss sono rari, non
        // serializzati". Quel numero veniva da un budget al 49% di residenza; ai
        // budget che contano per un modello grande i miss per nodo sono 2,30 e
        // 4,99, e la profondita' di coda effettiva 2,09 e 2,49. Il dispositivo a
        // quella coda da' 2552 MB/s e a coda 7,5 ne da' 3522: la leva c'era, ed era
        // stata chiusa sul regime sbagliato.
        //
        // Il blocco non e' piu' grande di cosi' perche' le sue slice restano tutte
        // bloccate fino alla fine: un blocco che non sta nel budget non ci starebbe
        // mai, e `psx_cache_ensure_many` lo rifiuta subito.
        // Emetti e continua: una slice alla volta — distribuzione fine come
        // sempre — con la lettura che parte e il thread che va avanti. Si aspetta
        // tutto alla fine, prima della barriera, dove i byte servono davvero.
        //
        // Il blocco di n slice non bastava, per un vincolo aritmetico: per
        // raccogliere piu' miss serve consumare piu' slice per lotto, e piu' slice
        // per lotto significa meno lotti e meno margine di bilanciamento. Su 45
        // slice e 12 thread, lotti da 4 ne lasciano 12 — margine zero — e costano
        // in attesa fra thread piu' di quanto rendano in I/O (misurato: 2,26x di
        // attesa, +8% di parete). Il 2 era il massimo sotto quel vincolo, e
        // raccoglieva 1,48 letture su 2.
        //
        // Qui le due cose sono slegate: i lotti restano 45, e la profondita' di
        // coda diventa pari ai miss che un thread incontra.
        //
        // I tre modi stanno qui accanto, selezionabili a runtime, perche' il
        // confronto fra loro e' un risultato del progetto e un risultato va
        // poter essere rifatto. Vedi `ModoLettura`.
        if (c.modo == LETTURA_ASINCRONA) {
            for (;;) {
                const uint32_t k = c.prossima_slice.fetch_add(1, std::memory_order_relaxed);
                if (k >= totale) {
                    break;
                }
                const Dove &   d   = dove[k / n_serve];
                const uint32_t x   = da_leggere[k % n_serve];
                const size_t   off = d.t_off + (size_t) x * d.slab;
                if (psx_cache_avvia(c.cache, c.store, d.region, off, id.layer, x, d.kind) == 0) {
                    guaio = true;
                    break;
                }
                // I pin si accumulano in locale e si versano una volta sola alla fine:
                // prenderli sotto mutex uno per uno metteva un punto di contesa nel
                // percorso caldo, e la misura lo mostrava (lock al 48% con 8 thread).
                pin_miei.push_back(Contesto::Pin{ d.region, off });
            }
            // Le letture emesse vanno attese QUI: da questo punto i byte si usano.
            // Uno zero e' fatale per il nodo, e lo stato dei pin non e' piu' quello
            // registrato sopra: si aborta, non si sblocca.
            if (!psx_cache_attendi(c.cache, c.store)) {
                guaio = true;
            }
        } else if (c.modo == LETTURA_SINCRONA) {
            // La prima versione: una slice, la si aspetta, la prossima. Coda 1
            // per thread, qualunque sia il numero di miss del nodo.
            for (;;) {
                const uint32_t k = c.prossima_slice.fetch_add(1, std::memory_order_relaxed);
                if (k >= totale) {
                    break;
                }
                const Dove &   d   = dove[k / n_serve];
                const uint32_t x   = da_leggere[k % n_serve];
                const size_t   off = d.t_off + (size_t) x * d.slab;
                if (!psx_cache_ensure(c.cache, c.store, d.region, off, id.layer, x, d.kind)) {
                    guaio = true;
                    break;
                }
                pin_miei.push_back(Contesto::Pin{ d.region, off });
            }
        } else {
            // La seconda: blocchi di dimensione fissa, tutto-o-niente, con la
            // fine del blocco come punto di attesa.
            psx_region *  rr[ARGINE_BLOCCO_MAX];
            size_t        oo[ARGINE_BLOCCO_MAX];
            uint32_t      ll[ARGINE_BLOCCO_MAX];
            uint32_t      ee[ARGINE_BLOCCO_MAX];
            enum psx_kind kk[ARGINE_BLOCCO_MAX];
            const uint32_t quante = c.blocco > ARGINE_BLOCCO_MAX ? ARGINE_BLOCCO_MAX : c.blocco;
            while (!guaio) {
                size_t n = 0;
                while (n < quante) {
                    const uint32_t k = c.prossima_slice.fetch_add(1, std::memory_order_relaxed);
                    if (k >= totale) {
                        break;
                    }
                    const Dove &   d = dove[k / n_serve];
                    const uint32_t x = da_leggere[k % n_serve];
                    rr[n] = d.region;
                    oo[n] = d.t_off + (size_t) x * d.slab;
                    ll[n] = id.layer;
                    ee[n] = x;
                    kk[n] = d.kind;
                    n++;
                }
                if (n == 0) {
                    break;
                }
                if (!psx_cache_ensure_many(c.cache, c.store, n, rr, oo, ll, ee, kk)) {
                    guaio = true;
                    break;
                }
                for (size_t i = 0; i < n; i++) {
                    pin_miei.push_back(Contesto::Pin{ rr[i], oo[i] });
                }
            }
        }
        if (guaio) {
            // Non si puo' proseguire: la memoria non e' impegnata, e calcolare
            // qui sarebbe un crash o, peggio, un risultato sbagliato in
            // silenzio. Meglio fermarsi dicendo perche'.
            psx_cache_stats st;
            psx_cache_get_stats(c.cache, &st);
            GGML_ABORT(
                "argine: impossibile rendere residenti gli expert del layer %u (%s). "
                "Residenti %" PRIu64 " MiB su un budget di %" PRIu64 " MiB, questo nodo "
                "ne chiede %u su %u. Alza ARGINE_EXPERT_BUDGET: il minimo e' lo spazio "
                "per tutti gli expert che un singolo ubatch puo' toccare.",
                id.layer, src0->name, st.bytes_resident >> 20, st.budget >> 20, n_serve,
                n_exp);
        }

        // tutti aspettano che le slice siano a posto, poi il kernel standard di
        // ggml fa il calcolo sulla memoria che abbiamo riempito
        ggml_barrier(params->threadpool);
        c.ns_nostro.fetch_add(
            (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - t_nostro).count(),
            std::memory_order_relaxed);
        if (ith == 0) {
            c.nodi.fetch_add(1, std::memory_order_relaxed);
            if (c.inizio.time_since_epoch().count() == 0) {
                c.inizio = std::chrono::steady_clock::now();
            }
            if (c.verbose) {
                // ogni 5 secondi: cosi' i numeri esistono anche se il processo
                // viene terminato invece di uscire
                const auto adesso = std::chrono::steady_clock::now();
                if (c.ultimo_resoconto.time_since_epoch().count() == 0) {
                    c.ultimo_resoconto = adesso;
                } else if (adesso - c.ultimo_resoconto >= std::chrono::seconds(5)) {
                    c.ultimo_resoconto = adesso;
                    c.resoconto("in corso");
                }
            }
        }
        if (arx) {
            if (ith == 0) {
                const uint64_t t_start = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                                             t_nostro.time_since_epoch()).count();
                arx_node(c.arx_graph, id.layer, (uint32_t) id.kind, t_start, arx_now_ns(), n_serve,
                         (uint32_t) nth);
            }
            arx_flush_thread();
            if (ith == 0) {
                arx_flush_file(250);
            }
        }
        return false;
    }
};

class ExpertBufferType : public ggml::cpu::extra_buffer_type {
    bool supports_op(ggml_backend_dev_t, const ggml_tensor * op) override {
        if (op->op != GGML_OP_MUL_MAT_ID) {
            return false;
        }
        const ggml_tensor * w = op->src[0];
        if (!w || !w->buffer || w->buffer->buft != argine_buffer_type()) {
            return false;
        }
        Contesto & c = Contesto::instance();
        if (!c.attivo) {
            return false;
        }
        const TensorId id = identifica(w->name);
        if (!id.valido) {
            // Un tensore `ffn_gate_up_exps` e' gate e up FUSI in uno. llama.cpp
            // lo preferisce quando il GGUF lo contiene, e per noi e' il modo
            // peggiore di fallire: non corrisponde a nessuno dei tre nomi che
            // cerchiamo, quindi lo store servirebbe solo `down` — un terzo dei
            // byte — e la RAM sforerebbe il budget di due terzi senza che una
            // sola riga lo dica. Lo store non lo gestisce: va detto.
            if (strstr(w->name, "ffn_gate_up_exps")) {
                std::lock_guard<std::mutex> lk(c.mtx);
                if (!c.detto_fuso) {
                    c.detto_fuso = true;
                    fprintf(stderr,
                            "argine: %s ha gate e up FUSI in un tensore solo: l'expert "
                            "store non lo gestisce (cerca ffn_gate_exps e ffn_up_exps "
                            "separati). Questi tensori restano sulla memoria mappata, "
                            "quindi il budget NON e' un tetto sulla RAM totale: lo "
                            "store sta servendo solo down_exps.\n",
                            w->name);
                }
            }
            return false;
        }
        const psx_entry * e = psx_get(c.store, id.layer, 0);
        if (!e) {
            return false;
        }
        // le dimensioni devono combaciare, altrimenti lo store non e' di questo
        // modello e conviene lasciare il tensore al percorso normale
        if ((uint32_t) w->ne[2] != psx_n_experts(c.store) ||
            (size_t) w->nb[2] != (size_t) e->size[id.kind]) {
            return false;
        }
        // La selezione degli expert usa una bitmap su stack: oltre il tetto
        // dovremmo troncarla, e un expert troncato non verrebbe mai caricato,
        // producendo risultati sbagliati senza dire niente. Meglio rifiutare il
        // tensore e lasciarlo al percorso normale.
        if ((uint32_t) w->ne[2] > PSX_MAX_EXPERTS) {
            std::lock_guard<std::mutex> lk(c.mtx);
            if (!c.detto_troppi_expert) {
                c.detto_troppi_expert = true;
                fprintf(stderr,
                        "argine: %s ha %" PRId64 " expert, oltre il limite di %d di questa "
                        "build: questi tensori restano sul percorso normale\n",
                        w->name, w->ne[2], PSX_MAX_EXPERTS);
            }
            return false;
        }
        // con la cache dell'OS scavalcata la destinazione deve essere allineata
        // al settore: se la slice di questo modello non lo e', non possiamo
        // servirlo e lo diciamo una volta sola
        if (c.direct && ((size_t) w->nb[2] % psx_alignment(c.store)) != 0) {
            std::lock_guard<std::mutex> lk(c.mtx);
            if (!c.detto_disallineato) {
                c.detto_disallineato = true;
                GGML_LOG_WARN(
                    "argine: le slice di %s sono %zu byte, non multipli di %u: con "
                    "ARGINE_EXPERT_DIRECT=1 non sono leggibili senza cache. Questi "
                    "tensori restano sul percorso normale; con "
                    "ARGINE_EXPERT_DIRECT=0 verrebbero serviti dallo store, ma la RAM "
                    "verrebbe contata due volte e il budget non sarebbe piu' rigido.\n",
                    w->name, (size_t) w->nb[2], psx_alignment(c.store));
            }
            return false;
        }
        return true;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(const ggml_tensor * op) override {
        if (op->op != GGML_OP_MUL_MAT_ID) {
            return nullptr;
        }
        if (!op->src[0] || !op->src[0]->buffer ||
            op->src[0]->buffer->buft != argine_buffer_type()) {
            return nullptr;
        }
        static ExpertTraits traits;
        return &traits;
    }
};

ggml_backend_buffer_type_t argine_buffer_type() {
    static ExpertBufferType          contesto;
    static ggml_backend_buffer_type  tipo = {
        /* .iface = */ {
            /* .get_name       = */ buffer_type_name,
            /* .alloc_buffer   = */ buffer_type_alloc,
            /* .get_alignment  = */ buffer_type_alignment,
            /* .get_max_size   = */ nullptr,
            /* .get_alloc_size = */ nullptr,
            /* .is_host        = */ buffer_type_is_host,
        },
        /* .device  = */ nullptr,
        /* .context = */ &contesto,
    };
    if (!tipo.device) {
        tipo.device = ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0);
    }
    return &tipo;
}

}  // namespace

// Esposto a ggml-cpu.cpp: nullptr se la variabile d'ambiente non c'e', cosi'
// senza configurazione llama.cpp si comporta esattamente come prima.
ggml_backend_buffer_type_t ggml_backend_cpu_argine_buffer_type(void) {
    return Contesto::instance().attivo ? argine_buffer_type() : nullptr;
}

void ggml_backend_cpu_argine_report(void) {
    Contesto & c = Contesto::instance();
    if (!c.attivo || !c.verbose) {
        return;
    }
    psx_cache_stats st;
    psx_cache_get_stats(c.cache, &st);
    const uint64_t tot = st.hits + st.misses;
    GGML_LOG_INFO(
        "argine: expert cache: %" PRIu64 " hit, %" PRIu64 " miss (%.1f%%), %" PRIu64
        " sfratti, %" PRIu64 " MiB letti, %" PRIu64 "/%" PRIu64 " MiB residenti\n",
        st.hits, st.misses, tot ? 100.0 * (double) st.hits / (double) tot : 0.0, st.evictions,
        st.bytes_read >> 20, st.bytes_resident >> 20, st.budget >> 20);
}
