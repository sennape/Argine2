// Argine — registrazione del buffer type per gli expert serviti da disco.
#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Il buffer type per gli expert MoE, oppure NULL se ARGINE_EXPERT_STORE non
// e' impostata: in quel caso llama.cpp si comporta esattamente come prima.
ggml_backend_buffer_type_t ggml_backend_cpu_argine_buffer_type(void);

// Stampa le statistiche della expert cache, se ARGINE_EXPERT_VERBOSE=1.
void ggml_backend_cpu_argine_report(void);

#ifdef __cplusplus
}
#endif
