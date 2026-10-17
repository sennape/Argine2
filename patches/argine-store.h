// Argine — lettore dell'expert store e gestore di residenza.
//
// Autonomo per scelta: non include nulla di llama.cpp o di ggml, cosi' si
// compila e si collauda da solo, contro uno store prodotto da
// `argine store build`. Il buffer type che lo usa e' un altro file.
//
// Cosa risolve
// ------------
// Su una macchina da 16 GiB un MoE da 17 GiB non ci sta, quindi gli expert
// vanno letti da disco mentre il modello gira. Farlo con la memoria mappata
// costa 4.3x in throughput (310 contro 1324 MB/s misurati a un thread sullo
// stesso disco, stessa slice, cache fredda in entrambi i casi) e, soprattutto,
// lascia decidere la residenza al memory manager: il tetto di RAM diventa una
// speranza, non un numero.
//
// Qui la residenza e' nostra: lo spazio di indirizzamento viene riservato per
// intero, ma la memoria fisica si impegna una slice per volta e si rilascia
// quando la cache e' piena. Il puntatore che ggml riceve resta valido e
// contiguo per tutto il tensore, che e' la sua unica pretesa.

#ifndef ARGINE_STORE_H
#define ARGINE_STORE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PSX_MAGIC        0x53585350u  // 'PSXS'
#define PSX_VERSION      2u
#define PSX_HEADER_BYTES 256u
#define PSX_N_KINDS      3u  // gate, up, down

enum psx_kind {
    PSX_GATE = 0,
    PSX_UP   = 1,
    PSX_DOWN = 2,
};

// Dove stanno le tre slice di un expert. Ogni offset e' multiplo
// dell'allineamento dichiarato, quindi ogni slice si legge con l'I/O non
// bufferizzato direttamente nella memoria del suo tensore.
struct psx_entry {
    uint64_t offset[PSX_N_KINDS];
    uint32_t size[PSX_N_KINDS];
};

struct psx_store;

// --- apertura --------------------------------------------------------------
//
// `direct` a 1 apre il file scavalcando la cache dell'OS: e' la condizione per
// un budget rigido, perche' altrimenti ogni byte esiste due volte (nella nostra
// cache e in quella di Windows) e il tetto che dichiariamo e' falso.
// Costo misurato di questa scelta: 9% degli expert al secondo.
struct psx_store * psx_open(const char * path, int direct);
void               psx_close(struct psx_store * s);

// Messaggio dell'ultimo errore, o NULL. La stringa vive nello store.
const char * psx_error(const struct psx_store * s);

// 1 se le letture scavalcano davvero la cache dell'OS. Puo' essere 0 anche
// avendolo chiesto: alcuni filesystem e alcuni dispositivi non accettano
// letture non bufferizzate, e in quel caso si ripiega su quelle normali. Vale
// la pena saperlo, perche' con la cache dell'OS attiva la RAM viene contata due
// volte e il budget non e' piu' rigido.
int psx_direct(const struct psx_store * s);

uint32_t psx_n_layers(const struct psx_store * s);
uint32_t psx_n_experts(const struct psx_store * s);
uint32_t psx_alignment(const struct psx_store * s);
uint64_t psx_total_bytes(const struct psx_store * s);
uint64_t psx_model_bytes(const struct psx_store * s);

// Indice di riga del layer del modello, o -1 se lo store non lo contiene.
int  psx_layer_row(const struct psx_store * s, uint32_t layer);
// L'indice per un expert. Restituisce 0 se fuori intervallo.
const struct psx_entry * psx_get(const struct psx_store * s, uint32_t layer, uint32_t expert);

// --- lettura ---------------------------------------------------------------
//
// Legge una slice in `dst`. Con `direct` attivo, `dst` deve essere allineato
// alla pagina e ci devono stare `psx_slice_read_bytes` byte, non `size`: l'I/O
// non bufferizzato legge a multipli del settore, quindi la coda di
// allineamento viene scritta anche lei.
//
// Thread safe: ogni thread usa il proprio descrittore, preso da un pool.
// Restituisce i byte utili letti, oppure 0 in caso di errore.
size_t psx_read_slice(struct psx_store * s, uint32_t layer, uint32_t expert, enum psx_kind kind,
                      void * dst);

// Buffer allineato alla pagina, da usare come destinazione delle letture.
//
// Non e' una comodita': con l'I/O non bufferizzato la destinazione deve essere
// allineata al settore, e un buffer preso da malloc o da std::vector non lo e'.
// Il primo utente di questa API (il codice di verifica del buffer type) ci e'
// cascato subito, leggendo dentro un std::vector e ottenendo EINVAL.
void * psx_alloc_aligned(const struct psx_store * s, size_t bytes);
void   psx_free_aligned(void * p, size_t bytes);

// Byte che una lettura tocca davvero (size arrotondato all'allineamento).
size_t psx_slice_read_bytes(const struct psx_store * s, uint32_t layer, uint32_t expert,
                            enum psx_kind kind);

// --- residenza -------------------------------------------------------------
//
// Una regione e' lo spazio di un tensore: riservata per intero (indirizzi, non
// memoria), impegnata a pezzi. Su 64 bit riservare 20 GiB non costa nulla, la
// RAM occupata e' solo quella dei pezzi impegnati.
struct psx_region;

struct psx_region * psx_region_new(size_t bytes, size_t slot_bytes, size_t n_slots);
void                psx_region_free(struct psx_region * r);
void *              psx_region_base(const struct psx_region * r);
size_t              psx_region_committed(const struct psx_region * r);

// Impegna memoria fisica per [offset, offset+bytes) dentro la regione.
// `offset` e `bytes` vengono allargati ai confini di pagina.
int psx_region_commit(struct psx_region * r, size_t offset, size_t bytes);
// La rilascia, tenendo gli indirizzi riservati: il puntatore di ggml resta valido.
int psx_region_decommit(struct psx_region * r, size_t offset, size_t bytes);

// --- cache -----------------------------------------------------------------
//
// LRU con budget rigido in byte. Le chiavi sono (regione, offset): una slice.
struct psx_cache;

struct psx_cache_stats {
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;
    uint64_t bytes_read;
    uint64_t bytes_resident;
    uint64_t budget;

    // Blocchi di lettura emessi e letture che contenevano. Servono a interpretare
    // ns_read, che con le letture sovrapposte e' il parete del blocco e non la
    // somma delle singole: la profondita' di coda effettiva e'
    // (ns_read/parete) x (letture/blocchi).
    uint64_t blocchi;
    uint64_t letture;

    // Dove se ne va il tempo. Sono qui perche' due modelli di fila su cosa
    // limitasse la generazione sono stati refutati da una misura: prima la
    // banda del disco (tagliato il traffico del 32%, velocita' invariata), poi
    // le sincronizzazioni in catena (tolte due terzi, guadagnato il 2%). La CPU
    // sta al 35% e il disco a un decimo di quanto sa fare: qualcosa fa aspettare
    // entrambi, e tirare a indovinare e' costato piu' tempo di quanto misurare
    // ne richieda.
    //
    // Nanosecondi sommati su tutti i thread, quindi il totale puo' superare il
    // tempo reale: e' il costo aggregato, non la durata.
    uint64_t ns_commit;    // VirtualAlloc(MEM_COMMIT): Windows azzera le pagine
    uint64_t ns_read;      // la lettura vera dallo store
    uint64_t ns_decommit;  // rilascio della memoria sfrattata
    uint64_t ns_attesa;    // fermi ad aspettare una slice che carica un altro
    uint64_t ns_lock;      // fermi sul mutex della cache
    uint64_t ns_fault;     // pagine nuove toccate prima della lettura (solo col pre-tocco)

    // Argine2: admissions after which a shard exceeded its budget. The hard
    // memory invariant (resident <= budget) says this must be 0; it is counted
    // so that each measurement can demonstrate it instead of assuming it.
    uint64_t violazioni;
};

// `shards` a 0 = scelta automatica. Il numero ottimale non e' deducibile: piu'
// settori tolgono contesa sul mutex della cache, ma espongono la contesa sul
// lock dello spazio di indirizzamento del processo, perche' commit e decommit
// smettono di essere serializzati. Misurato su 16 settori: mutex dal 16% al 6%,
// commit+decommit dal 5% al 16%. Va trovato per misura, non per ragionamento.
struct psx_cache * psx_cache_new_n(uint64_t budget_bytes, unsigned shards);
struct psx_cache * psx_cache_new(uint64_t budget_bytes);
// Quanti settori indipendenti ha la cache. Serve a poter verificare che la
// divisione sia davvero attiva invece di darla per scontata.
unsigned           psx_cache_shards(const struct psx_cache * c);
// Byte residenti in ciascun settore: serve a vedere se la divisione e'
// equilibrata. Con un hash sbagliato tutte le chiavi finiscono in un settore
// solo e la cache vale un sedicesimo di quella dichiarata, senza che niente lo
// segnali.
void               psx_cache_shard_load(const struct psx_cache * c, uint64_t * out, unsigned n);
// Settore a cui finirebbe questa chiave (per i test sulla distribuzione).
unsigned           psx_debug_shard_index(const struct psx_cache * c, struct psx_region * region,
                                         size_t offset);
void               psx_cache_free(struct psx_cache * c);
void               psx_cache_get_stats(const struct psx_cache * c, struct psx_cache_stats * out);

// Garantisce che la slice sia in memoria e leggibile all'indirizzo
// `psx_region_base(region) + offset`, e la **blocca** finche' non si chiama
// `psx_cache_unpin`. Se manca, sfratta quanto serve per rispettare il budget,
// impegna la memoria e legge dallo store.
//
// Il blocco non e' un dettaglio: senza di esso un altro thread puo' sfrattare
// la slice nell'istante fra il ritorno di questa funzione e il suo uso, e
// l'accesso finisce su memoria decommitata, cioe' in un crash. E' il difetto
// che il test da stress ha trovato con un segfault alla prima esecuzione.
//
// Restituisce 1 se la slice e' residente e bloccata, 0 in caso di errore
// (compreso il caso in cui tutto lo spazio del budget e' bloccato da altri).
int psx_cache_ensure(struct psx_cache * c, struct psx_store * s, struct psx_region * region,
                     size_t offset, uint32_t layer, uint32_t expert, enum psx_kind kind);

// Come sopra per n slice insieme, con **le letture tutte in volo** dove il
// sistema lo permette (su Windows: handle con FILE_FLAG_OVERLAPPED).
//
// Esiste perche' il guadagno misurabile non sta nel leggere piu' byte per volta
// ma nell'avere piu' letture contemporanee: a byte totali fissi per nodo MoE le
// due cose si scambiano, e il dispositivo premia la seconda. Alla profondita' di
// coda 2,49 misurata in generazione rende 2552 MB/s; a coda 7,5 ne rende 3522.
//
// **Tutto o niente.** Se ritorna 1 tutte le n slice sono residenti e bloccate, e
// chi chiama deve sbloccarle tutte. Se ritorna 0 non ne resta bloccata nessuna,
// nemmeno quelle lette con successo prima del guasto: chi chiama non ha modo di
// sapere quali sarebbero, e un pin che nessuno sblocca occupa il budget per
// sempre.
//
// Rifiuta subito, senza tentare, un blocco la cui somma supera il budget: le sue
// slice devono stare residenti tutte insieme, quindi l'impossibilita' e'
// calcolabile in partenza. Senza quel controllo ogni tentativo spendeva due
// secondi di attese prima di arrendersi, e dal vivo sembrava uno stallo.
//
// Su POSIX le letture sono sequenziali: il risultato e' identico, manca il
// guadagno. Il bersaglio del progetto e' Windows.
int psx_cache_ensure_many(struct psx_cache * c, struct psx_store * s, size_t n,
                          struct psx_region * const * regions, const size_t * offsets,
                          const uint32_t * layers, const uint32_t * experts,
                          const enum psx_kind * kinds);

// Prenota una slice e **avvia** la sua lettura senza aspettarla, oppure la serve
// subito se e' gia' residente.
//
// Ritorna 1 se era un hit (pronta, bloccata), 2 se la lettura e' partita (slice
// bloccata, byte non ancora arrivati), 0 in caso di errore (niente bloccato).
//
// Esiste perche' il blocco di n slice non slega le due cose che servono: per
// raccogliere piu' miss bisogna consumare piu' slice per lotto, e piu' slice per
// lotto significa meno lotti e meno margine per il bilanciamento dinamico fra i
// thread. Misurato: su 45 slice e 12 thread, lotti da 4 costano in attesa fra
// thread piu' di quanto rendano in I/O.
//
// Qui le due cose si slegano, perche' i byte non servono al ritorno di questa
// funzione ma alla barriera prima del calcolo: il thread prende una slice alla
// volta — distribuzione fine — e la profondita' di coda diventa pari ai miss che
// incontra, non a una costante scelta a priori.
//
// **Chi chiama deve invocare `psx_cache_attendi` prima di leggere i byte.**
int psx_cache_avvia(struct psx_cache * c, struct psx_store * s, struct psx_region * region,
                    size_t offset, uint32_t layer, uint32_t expert, enum psx_kind kind);

// Attende tutte le letture avviate da QUESTO thread e le annuncia. Ritorna 1 se
// sono arrivate tutte, 0 se almeno una no.
//
// Uno zero e' **fatale per il nodo**: i byte di una slice non ci sono, e calcolare
// darebbe un risultato sbagliato in silenzio. Dopo uno zero lo stato dei pin non e'
// quello che il chiamante ha registrato — le slice fallite non sono piu' bloccate —
// quindi non va sbloccato niente: va abortito.
int psx_cache_attendi(struct psx_cache * c, struct psx_store * s);

// Sblocca la slice: da qui puo' essere sfrattata. Va chiamata una volta per
// ogni `psx_cache_ensure` andato a buon fine.
void psx_cache_unpin(struct psx_cache * c, struct psx_region * region, size_t offset);

// Budget minimo utilizzabile, in byte: lo spazio che serve per tenere bloccate
// insieme tutte le slice di una singola operazione.
//
// Con un budget piu' piccolo il motore non riesce a completare un layer e
// `psx_cache_ensure` fallisce. Attenzione: durante l'elaborazione del prompt un
// ubatch tocca molti piu' expert che durante la generazione (nel caso peggiore
// tutti), quindi il minimo va calcolato sul caso prompt, non su
// `expert_used_count`.
uint64_t psx_cache_min_budget(const struct psx_store * s, uint32_t n_expert_per_op);

// Marca la slice come usata di recente senza leggerla (per il prefetch).
void psx_cache_touch(struct psx_cache * c, struct psx_region * region, size_t offset);

// --- solo per i test -------------------------------------------------------
//
// Allarga a comando la finestra fra "la slice e' stata annunciata nell'indice"
// e "i byte sono arrivati". Serve a rendere riproducibile una corsa che
// altrimenti si vede una volta su diecimila: leggere 8 KiB dalla page cache
// dura un microsecondo, avviare un thread ne dura cinquanta, quindi per caso
// non ci si inciampa mai. Un bug che non si riesce a far scattare e' il tipo
// peggiore di bug. In produzione il ritardo e' zero.
void psx_debug_set_read_delay_us(unsigned us);

// Fa fallire le prossime n letture (l'ultima richiesta di ogni blocco). Solo per
// i test: il percorso "una slice del blocco non arriva" non si raggiunge
// altrimenti, e il contratto tutto-o-niente resterebbe affermato invece che
// verificato.
void psx_debug_fail_next_reads(unsigned n);

// Tocca le pagine appena impegnate PRIMA di leggerci dentro, cronometrandolo a
// parte.
//
// Serve a separare due costi che altrimenti finiscono nello stesso secchio.
// `VirtualAlloc(MEM_COMMIT)` costa 0.005 ms per slice, ma non e' li' che si paga
// la memoria nuova: Windows azzera le pagine al primo accesso, e quel primo
// accesso avviene dentro il DMA della lettura. Misurando solo il commit si
// conclude che le pagine sono gratis, il che e' falso: su 55.715 miss da 227
// pagine l'una i page fault misurati erano 14 milioni.
//
// Col pre-tocco attivo il costo dei fault si legge in `ns_fault` e la lettura
// resta pulita. Se la somma non cambia, i fault sono inevitabili; se cala, la
// strada e' smettere di impegnare e rilasciare pagine a ogni slice.
void psx_debug_set_pretouch(int on);

#ifdef __cplusplus
}
#endif

#endif  // ARGINE_STORE_H
