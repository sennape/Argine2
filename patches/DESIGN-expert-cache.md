# Buffer type per expert su disco — progetto

Note tecniche ricavate leggendo llama.cpp (build b11214 circa, `ggml/src/ggml-cpu/`).
Servono a scrivere la patch senza riscoprire ogni volta le stesse cose.

## Il punto di innesto

llama.cpp permette a un terzo di registrare un proprio **buffer type** che prende
possesso di certi tensori e fornisce la propria implementazione delle operazioni
su di essi. Lo usano gia' `repack.cpp`, `amx/amx.cpp` e `kleidiai/`.

Due interfacce, in `ggml/src/ggml-cpu/traits.h`:

```cpp
namespace ggml::cpu {
class tensor_traits {
    virtual bool work_size(int n_threads, const ggml_tensor * op, size_t & size) = 0;
    virtual bool compute_forward(ggml_compute_params * params, ggml_tensor * op) = 0;
};

class extra_buffer_type {
    virtual bool            supports_op(ggml_backend_dev_t, const ggml_tensor * op) = 0;
    virtual tensor_traits * get_tensor_traits(const ggml_tensor * op) = 0;
};
}
```

La registrazione avviene in `ggml_backend_cpu_get_extra_buffer_types()`
(`ggml-cpu.cpp`), esposta anche come `ggml_backend_dev_get_extra_bufts`.

`amx/amx.cpp` e' il modello piu' compatto da imitare: ~200 righe per un buffer
type completo con `get_name`, `alloc_buffer`, `get_alignment`, `set_tensor`.

## L'operazione che ci interessa

`GGML_OP_MUL_MAT_ID` e' l'operazione MoE. In `ggml_compute_forward_mul_mat_id`
(`ggml-cpu.c`):

```cpp
const ggml_tensor * src0 = dst->src[0];  // pesi degli expert, 3D: ne02 = n_expert
const ggml_tensor * src1 = dst->src[1];  // attivazioni
const ggml_tensor * ids  = dst->src[2];  // int32: ne[0] = n_expert_used, ne[1] = n_token
```

Gli id sono disponibili **prima** di eseguire la moltiplicazione: e' esattamente
il punto in cui sappiamo quali expert servono e possiamo procurarceli.

## Il problema della residenza parziale

ggml alloca i buffer in anticipo e si aspetta che `tensor->data` sia un puntatore
valido per l'intera dimensione del tensore. Non esiste un meccanismo di "fault"
sull'accesso.

La via d'uscita: **riservare lo spazio di indirizzamento virtuale senza
impegnare memoria fisica**, e impegnare/rilasciare le singole slice.

- Windows: `VirtualAlloc(MEM_RESERVE)` per l'intero tensore, poi `MEM_COMMIT`
  per le slice in cache e `VirtualFree(MEM_DECOMMIT)` all'eviction.
- POSIX: `mmap(PROT_NONE)` per riservare, `mprotect` + `MADV_DONTNEED` per
  rilasciare.

Lo spazio di indirizzamento a 64 bit e' abbondante (128 TB), quindi riservare
20 GiB non costa nulla. Il puntatore resta contiguo e valido, che e' la
condizione che ggml pretende, ma la RAM fisica occupata e' solo quella delle
slice impegnate: **budget rigido e nostro**.

Da verificare prima di impegnarsi: che nessun percorso di ggml tocchi l'intero
tensore (per esempio in fase di validazione o di copia), perche' toccherebbe
pagine non impegnate.

## Da dove arrivano i byte

`set_tensor` riceve i pesi durante il caricamento, ma non dice da quale offset
del GGUF provengano. Due opzioni:

1. **Sidecar preparato offline.** `argine reorder` produce gia' un file
   riordinato; puo' produrre anche un "expert store" con le slice allineate e
   un indice. Il buffer type legge da li', senza costo al caricamento. Preferita.
2. Scrittura del sidecar al primo caricamento, riuso nei successivi.

L'opzione 1 si sposa con il riordino: le slice sono gia' disposte per localita'.

## Piano a stadi

Ogni stadio e' utile da solo e non rompe il precedente.

**Stadio 1 — sola telemetria, rischio nullo.**
Il buffer type si registra, `compute_forward` legge gli id, aggiorna una cache
LRU simulata e le statistiche, poi **ritorna false** lasciando eseguire il
percorso CPU standard. Otteniamo hit rate e pattern di accesso misurati dentro
il motore, con la certezza che i risultati numerici non cambiano.

**Stadio 2 — residenza reale.**
Riserva virtuale + commit/decommit, lettura delle slice dal sidecar, budget
rigido. `compute_forward` garantisce le slice necessarie e poi delega.

**Stadio 3 — prefetch profondo.**
Thread che precarica gli expert dei layer successivi. L'obiettivo non e'
indovinare meglio ma **tenere piena la coda dell'SSD**: su NVMe la profondita' di
coda conta piu' dell'accuratezza della predizione.

**Stadio 4 — doppia precisione.**
Copia a 2 bit dell'intero set di expert residente in permanenza, piu' un overlay
a precisione piena degli expert caldi. In caso di miss si calcola subito a bassa
precisione invece di aspettare il disco: la velocita' resta costante e a degradare
e' la qualita', non la latenza. Da validare con la perplessita'.

## Rischi noti

- **Threading**: `compute_forward` e' chiamata da piu' thread con `params->ith`
  diverso. Il caricamento deve avvenire una volta sola, con barriera prima
  dell'uso.
- **Backend Vulkan**: se il grafo viene offloadato, il nostro percorso non viene
  eseguito. Il buffer type vale per i tensori tenuti su CPU.
- **Rebase**: llama.cpp cambia in fretta; la patch va mantenuta.
- **Interazione col repack**: entrambi sono extra buffer type e competono per gli
  stessi tensori. Con `--no-repack` (gia' default in Argine) il conflitto non
  si pone.
