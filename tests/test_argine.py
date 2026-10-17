"""Test di Argine. Esecuzione:  python -m unittest discover -s tests -v

Non richiedono rete ne' modelli veri: i GGUF sono fixture sparse generate al volo.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from argine import cli                          # noqa: E402
from argine import bench as benchmod           # noqa: E402
from argine import diskbench as db              # noqa: E402
from argine import expert_cache as ec           # noqa: E402
from argine import expert_store as es           # noqa: E402
from argine import hardware as hwmod            # noqa: E402
from argine import reorder as rd                # noqa: E402
from argine import planner as planmod           # noqa: E402
from argine import quality as qmod              # noqa: E402
from argine import quantize as qzmod            # noqa: E402
from argine import anthropic_api as aa          # noqa: E402
from argine import runtime as rtmod             # noqa: E402
from argine import server as srv                # noqa: E402
from argine import telemetry            # noqa: E402
from argine.gguf import GGML_TYPES, GGUFError, find_gguf, read_gguf  # noqa: E402
from argine.util import GB, MB, fmt_bytes, parse_size  # noqa: E402

import make_fixtures  # noqa: E402

_TMP: tempfile.TemporaryDirectory | None = None
DENSE: Path
MOE: Path
TINY_MOE: Path


def setUpModule() -> None:
    global _TMP, DENSE, MOE, TINY_MOE
    _TMP = tempfile.TemporaryDirectory(prefix="argine-test-")
    base = Path(_TMP.name)
    DENSE = base / "dense-8B-Q4_K_M.gguf"
    MOE = base / "moe-30B-A3B-Q4_K_M.gguf"
    TINY_MOE = base / "tiny-moe.gguf"
    make_fixtures.make_dense_8b(DENSE)
    make_fixtures.make_moe_30b(MOE)
    from make_tiny_model import make_tiny_moe
    make_tiny_moe(TINY_MOE)


def tearDownModule() -> None:
    if _TMP is not None:
        _TMP.cleanup()


def hw_16gb() -> hwmod.HardwareReport:
    hw = hwmod.HardwareReport()
    hw.ram_total = 16 * GB
    hw.ram_available = 12 * GB
    hw.cpu_cores_physical = 12
    hw.cpu_cores_logical = 16
    hw.cpu_brand = "Intel Core i7 (test)"
    hw.gpus = [hwmod.GPUInfo(name="Intel(R) Iris(R) Xe Graphics", vendor="intel",
                             integrated=True, source="test")]
    hw.vulkan_runtime = True
    hw.backends = ["cpu", "vulkan"]
    return hw


# ------------------------------------------------------------------- util ---

class TestUtil(unittest.TestCase):
    def test_parse_size(self):
        self.assertEqual(parse_size("11G"), 11 * GB)
        self.assertEqual(parse_size("11GB"), 11 * GB)
        self.assertEqual(parse_size("11GiB"), 11 * GB)
        self.assertEqual(parse_size("512M"), 512 * MB)
        self.assertEqual(parse_size("2048"), 2048)
        self.assertEqual(parse_size("1.5G"), int(1.5 * GB))
        self.assertIsNone(parse_size(None))
        with self.assertRaises(ValueError):
            parse_size("tanta")

    def test_fmt_bytes(self):
        self.assertEqual(fmt_bytes(11 * GB), "11.00 GiB")
        self.assertEqual(fmt_bytes(None), "n/d")


# ------------------------------------------------------------------- gguf ---

class TestGGUF(unittest.TestCase):
    def test_lelenco_dei_file_vicini_mette_per_primo_il_piu_simile(self):
        """Caso reale, con nove GGUF dai nomi quasi uguali: l'elenco era
        alfabetico e tagliato agli otto *senza dirlo*, e il file giusto e' finito
        nono. Su Windows l'ordinamento ignora le maiuscole e
        '...-Q4_K_M-reordered.gguf' precede '...-Q4_K_M.gguf' perche' il trattino
        viene prima del punto. Risultato: il messaggio nato per aiutare elencava
        otto file sbagliati e nascondeva l'unico utile.

        Il test gira anche su Linux, dove l'ordinamento alfabetico e' diverso:
        per questo fissa la *pertinenza*, che non dipende dal sistema."""
        d = Path(_TMP.name) / "simili"  # type: ignore[union-attr]
        d.mkdir(exist_ok=True)
        nomi = ["Qwen3-14B-Q4_K_M", "Qwen3-8B-Q4_K_M", "qwen30b-IQ2-reale",
                "qwen30b-IQ2S", "qwen30b-IQ2XS", "Qwen_Qwen3-30B-A3B-Q3_K_M",
                "Qwen_Qwen3-30B-A3B-Q4_K_M-experts-IQ2_XXS",
                "Qwen_Qwen3-30B-A3B-Q4_K_M-reordered",
                "Qwen_Qwen3-30B-A3B-Q4_K_M"]
        for n in nomi:
            (d / f"{n}.gguf").touch()

        with self.assertRaises(GGUFError) as cm:
            find_gguf(d / "Qwen3-30B-A3B-Q4_K_M.gguf")  # manca il prefisso "Qwen_"
        testo = str(cm.exception)
        # solo le voci dell'elenco: sono indentate di quattro spazi. La prima
        # riga del messaggio finisce anch'essa in .gguf (e' il file chiesto).
        righe = [r.strip() for r in testo.splitlines()
                 if r.startswith("    ") and r.strip().endswith(".gguf")]
        self.assertEqual(righe[0], "Qwen_Qwen3-30B-A3B-Q4_K_M.gguf",
                         "il file che l'utente intendeva deve stare in cima")
        self.assertIn("e altri 1", testo,
                      "se l'elenco viene troncato bisogna dirlo: il bug era "
                      "esattamente un troncamento silenzioso")
        # e il -reordered, che prima arrivava prima, ora sta sotto l'originale
        self.assertGreater(righe.index("Qwen_Qwen3-30B-A3B-Q4_K_M-reordered.gguf"),
                           righe.index("Qwen_Qwen3-30B-A3B-Q4_K_M.gguf"))

    def test_dense(self):
        m = read_gguf(DENSE)
        self.assertEqual(m.architecture, "qwen3")
        self.assertFalse(m.is_moe)
        self.assertEqual(m.n_layers, 36)
        self.assertEqual(m.n_head_kv, 8)
        self.assertEqual(m.head_dim_k, 128)
        self.assertEqual(m.n_params_active, m.n_params_total)
        self.assertGreater(m.n_params_total, 7e9)
        self.assertLess(m.n_params_total, 9e9)
        self.assertEqual(m.quantization, "Q4_K")
        self.assertAlmostEqual(m.bits_per_weight, 5.15, delta=0.3)
        self.assertEqual(m.expert_bytes, 0)

    def test_moe(self):
        m = read_gguf(MOE)
        self.assertTrue(m.is_moe)
        self.assertEqual(m.expert_count, 128)
        self.assertEqual(m.expert_used_count, 8)
        self.assertEqual(m.n_moe_layers, 48)
        # ~30B totali, ~3B attivi
        self.assertGreater(m.n_params_total, 25e9)
        self.assertLess(m.n_params_total, 35e9)
        self.assertGreater(m.n_params_active, 2e9)
        self.assertLess(m.n_params_active, 5e9)
        # gli expert dominano il file
        self.assertGreater(m.expert_bytes, m.shared_bytes * 8)
        self.assertGreater(m.expert_bytes_per_expert_per_layer, MB)

    def test_gguf_splittato_viene_ricomposto(self):
        """Ogni shard ha un header proprio: leggere solo il primo significherebbe
        contare una frazione degli expert."""
        base = Path(_TMP.name) / "split"
        base.mkdir(exist_ok=True)
        primo = base / "moe-00001-of-00002.gguf"
        secondo = base / "moe-00002-of-00002.gguf"
        kv = make_fixtures._moe_metadata(split_count=2)
        make_fixtures.build_gguf(primo, b"".join(kv), len(kv),
                                 make_fixtures._moe_layer_tensors(range(0, 24)))
        make_fixtures.build_gguf(secondo, b"".join(kv), len(kv),
                                 make_fixtures._moe_layer_tensors(range(24, 48)))
        m = read_gguf(primo)
        self.assertEqual(len(m.shards), 2)
        self.assertEqual(len(m.tensors), 48 * 3)
        self.assertEqual(m.n_moe_layers, 48)
        # ogni tensore sa da quale shard viene e dove inizia l'area dati
        self.assertEqual(len({t.file for t in m.tensors}), 2)
        idx = ec.build_expert_index(m)
        tardo = idx.get(47, 0)
        self.assertTrue(tardo)
        self.assertIn("00002", tardo[0].file)
        self.assertGreater(tardo[0].offset, 0)

    def test_shard_mancante_viene_segnalato(self):
        base = Path(_TMP.name) / "split-rotto"
        base.mkdir(exist_ok=True)
        solo = base / "moe-00001-of-00003.gguf"
        kv = make_fixtures._moe_metadata(split_count=3)
        make_fixtures.build_gguf(solo, b"".join(kv), len(kv),
                                 make_fixtures._moe_layer_tensors(range(0, 16)))
        with self.assertRaises(GGUFError) as ctx:
            read_gguf(solo)
        self.assertIn("scaricali tutti", str(ctx.exception))

    def test_file_non_gguf(self):
        with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as fh:
            fh.write(b"NOTGGUF" + b"\x00" * 64)
            path = fh.name
        try:
            with self.assertRaises(GGUFError):
                read_gguf(path)
        finally:
            os.unlink(path)

    def test_file_mancante(self):
        with self.assertRaises(GGUFError):
            read_gguf("/percorso/che/non/esiste.gguf")


# ---------------------------------------------------------------- planner ---

class TestPlanner(unittest.TestCase):
    def test_budget_automatico(self):
        hw = hw_16gb()
        budget, _ = planmod.safe_budget(hw, None)
        self.assertGreaterEqual(budget, 10 * GB)
        self.assertLessEqual(budget, 12.5 * GB)

    def test_budget_richiesto_viene_limitato(self):
        hw = hw_16gb()
        budget, notes = planmod.safe_budget(hw, 15 * GB)
        self.assertLessEqual(budget, int(16 * GB * 0.90))
        self.assertTrue(any("90%" in n for n in notes))

    def test_kv_cache_scala_con_ctx_e_tipo(self):
        m = read_gguf(DENSE)
        f16_4k = planmod.kv_cache_bytes(m, 4096, "f16", "f16")
        f16_8k = planmod.kv_cache_bytes(m, 8192, "f16", "f16")
        q4_4k = planmod.kv_cache_bytes(m, 4096, "q4_0", "q4_0")
        self.assertAlmostEqual(f16_8k / f16_4k, 2.0, places=5)
        self.assertLess(q4_4k, f16_4k / 3)
        # verifica analitica: 36 layer * 4096 tok * 8 kv-head * 128 dim * 2 (K,V) * 2 byte
        self.assertEqual(f16_4k, 36 * 4096 * 8 * 128 * 2 * 2)

    def test_dense_8b_sta_nel_budget(self):
        m = read_gguf(DENSE)
        hw = hw_16gb()
        p = planmod.plan(m, hw, planmod.PlanRequest(backend="cpu"))
        self.assertTrue(p.feasible)
        self.assertEqual(p.ctx, 4096)
        self.assertEqual(p.kv_type_k, "f16")
        self.assertLessEqual(p.estimate.total, p.budget)

    def test_moe_attiva_streaming_e_mantiene_context(self):
        m = read_gguf(MOE)
        hw = hw_16gb()
        p = planmod.plan(m, hw, planmod.PlanRequest(backend="cpu"))
        self.assertTrue(p.feasible)
        self.assertTrue(p.moe_streaming, "il modello da 16 GiB deve attivare lo streaming")
        self.assertGreaterEqual(p.ctx, 2048, "dopo il relax il context non deve restare minimo")
        self.assertGreater(p.expert_cache_budget, 0)
        self.assertLessEqual(p.estimate.total, p.budget)
        # i pesi residenti sono solo i tensori shared
        self.assertEqual(p.estimate.weights_resident, m.shared_bytes)

    def test_ram_limit_stretto_degrada(self):
        m = read_gguf(DENSE)
        hw = hw_16gb()
        # 5.5 GiB: i pesi (4.91) ci stanno, ma KV+compute a 4096 token no
        p = planmod.plan(m, hw, planmod.PlanRequest(backend="cpu", ram_limit=int(5.5 * GB)))
        self.assertTrue(p.feasible)
        self.assertLessEqual(p.estimate.total, 5.5 * GB)
        self.assertTrue(any("safety" in d for d in p.decisions))
        self.assertLess(p.ctx, 4096)

    def test_riordino_permuta_router_e_expert_insieme(self):
        """La correttezza sta qui: se si permutano gli expert senza permutare le
        righe del router, il modello instrada verso pesi sbagliati."""
        m = read_gguf(MOE)
        tr = rd.Trace()
        lt = rd.LayerTrace(layer=0, n_expert=m.expert_count)
        rng = random.Random(3)
        for _ in range(200):
            lt.tokens.append(tuple(sorted(rng.sample(range(16), 8))))
        for layer in range(m.n_layers or 48):
            copia = rd.LayerTrace(layer=layer, n_expert=m.expert_count)
            copia.tokens = list(lt.tokens)
            tr.layers[layer] = copia
        plan = rd.plan_reorder(m, tr)
        self.assertEqual(sorted(plan.perms[0]), list(range(m.expert_count)))
        # gli expert caldi (0-15) finiscono all'inizio del tensore
        self.assertTrue(all(e < 16 for e in plan.perms[0][:16]))
        # meno blocchi contigui per token = meno letture da disco
        self.assertLess(plan.media_dopo, plan.media_prima)

    def test_riordino_rifiuta_se_manca_il_router(self):
        """Fixture senza ffn_gate_inp: deve fermarsi, non riscrivere a meta'."""
        base = Path(_TMP.name) / "senza-router"
        base.mkdir(exist_ok=True)
        f = base / "moe.gguf"
        kvs = make_fixtures._moe_metadata()
        make_fixtures.build_gguf(f, b"".join(kvs), len(kvs),
                                 make_fixtures._moe_layer_tensors(range(0, 4)))
        m = read_gguf(f)
        plan = rd.ReorderPlan(n_expert=m.expert_count)
        plan.perms[0] = list(range(m.expert_count))
        problemi = rd.verify_plan(m, plan)
        self.assertTrue(any("gate_inp" in p for p in problemi))
        with self.assertRaises(ValueError):
            rd.rewrite(m, plan, base / "out.gguf")

    def test_parser_traccia(self):
        base = Path(_TMP.name) / "traccia"
        base.mkdir(exist_ok=True)
        t = base / "route.log"
        t.write_text(
            "blk.0.ffn_gate_exps.weight 2 2 1 3 5 7\n"
            "blk.0.ffn_up_exps.weight 2 2 1 3 5 7\n"      # stessi id: da ignorare
            "blk.1.ffn_gate_exps.weight 1 2 2 4\n"
            "riga malformata\n",
            encoding="utf-8")
        tr = rd.parse_trace(t)
        self.assertEqual(len(tr.layers), 2)
        self.assertEqual(tr.layers[0].tokens, [(1, 3), (5, 7)])
        self.assertEqual(tr.layers[1].tokens, [(2, 4)])

    def test_metrica_blocchi_contigui(self):
        lt = rd.LayerTrace(layer=0, n_expert=8)
        lt.tokens = [(0, 4), (1, 5)]
        # ordine naturale: expert distanti -> 2 blocchi per token
        self.assertAlmostEqual(rd.runs_per_token(lt, list(range(8))), 2.0)
        # ordine che li avvicina -> 1 blocco
        self.assertAlmostEqual(rd.runs_per_token(lt, [0, 4, 1, 5, 2, 3, 6, 7]), 1.0)

    def test_speculativa_ngram_non_costa_ram(self):
        m = read_gguf(MOE)
        hw = hw_16gb()
        base = planmod.plan(m, hw, planmod.PlanRequest(backend="cpu"))
        ngram = planmod.plan(m, hw, planmod.PlanRequest(backend="cpu", spec="ngram"))
        self.assertEqual(ngram.spec_type, "ngram-cache")
        self.assertEqual(ngram.estimate.draft, 0)
        self.assertEqual(base.expert_cache_budget, ngram.expert_cache_budget)

    def test_draft_model_toglie_spazio_alla_expert_cache(self):
        m = read_gguf(MOE)
        hw = hw_16gb()
        base = planmod.plan(m, hw, planmod.PlanRequest(backend="cpu"))
        spec = planmod.plan(m, hw, planmod.PlanRequest(
            backend="cpu", spec="draft", draft_model=str(DENSE)))
        self.assertEqual(spec.spec_type, "draft-simple")
        self.assertGreater(spec.estimate.draft, GB)
        self.assertLess(spec.expert_cache_budget, base.expert_cache_budget)
        self.assertLessEqual(spec.estimate.total, spec.budget)

    def test_draft_senza_modello_e_errore_chiaro(self):
        m = read_gguf(MOE)
        with self.assertRaises(ValueError) as ctx:
            planmod.plan(m, hw_16gb(), planmod.PlanRequest(backend="cpu", spec="draft"))
        self.assertIn("ngram", str(ctx.exception))

    def test_spec_sconosciuto(self):
        m = read_gguf(MOE)
        with self.assertRaises(ValueError):
            planmod.plan(m, hw_16gb(), planmod.PlanRequest(backend="cpu", spec="magia"))

    def test_repack_spento_di_default(self):
        """Misurato piu' lento E piu' pesante su CPU senza AVX-512/AMX."""
        m = read_gguf(DENSE)
        hw = hw_16gb()
        p = planmod.plan(m, hw, planmod.PlanRequest(backend="cpu"))
        self.assertFalse(p.repack)
        self.assertEqual(p.estimate.repack, 0)

    def test_repack_esplicito_viene_contabilizzato(self):
        m = read_gguf(DENSE)
        hw = hw_16gb()
        p = planmod.plan(m, hw, planmod.PlanRequest(backend="cpu", repack=True))
        self.assertTrue(p.repack)
        self.assertGreater(p.estimate.repack, 2 * GB)
        # solo i tipi ripacchettabili, senza embedding ne' tensori 1D
        self.assertLess(p.estimate.repack, m.tensor_bytes)

    def test_repack_esclude_embedding_e_tipi_non_ripacchettabili(self):
        m = read_gguf(DENSE)
        q4k = sum(t.n_bytes for t in m.tensors if t.type_name == "Q4_K")
        embd = sum(t.n_bytes for t in m.tensors if t.name.startswith("token_embd"))
        self.assertAlmostEqual(m.repackable_bytes, q4k - embd, delta=1024)

    def test_budget_impossibile_viene_rifiutato(self):
        m = read_gguf(DENSE)
        hw = hw_16gb()
        p = planmod.plan(m, hw, planmod.PlanRequest(backend="cpu", ram_limit=1 * GB))
        self.assertFalse(p.feasible)
        self.assertIsNotNone(p.failure_reason)

    def test_parametri_utente_non_vengono_sovrascritti(self):
        m = read_gguf(DENSE)
        hw = hw_16gb()
        p = planmod.plan(m, hw, planmod.PlanRequest(backend="cpu", ctx=8192, kv_type="q8_0"))
        self.assertEqual(p.ctx, 8192)
        self.assertEqual(p.kv_type_k, "q8_0")

    def test_kv_type_non_valido(self):
        m = read_gguf(DENSE)
        with self.assertRaises(ValueError):
            planmod.plan(m, hw_16gb(), planmod.PlanRequest(backend="cpu", kv_type="q3_k"))

    def test_igpu_non_riduce_la_ram(self):
        m = read_gguf(DENSE)
        hw = hw_16gb()
        cpu = planmod.plan(m, hw, planmod.PlanRequest(backend="cpu"))
        vk = planmod.plan(m, hw, planmod.PlanRequest(backend="vulkan"))
        self.assertEqual(cpu.estimate.weights_resident, vk.estimate.weights_resident)
        self.assertGreater(vk.estimate.gpu_shared, 0)


# ----------------------------------------------------------- expert cache ---

class TestExpertCache(unittest.TestCase):
    def test_indice_expert(self):
        m = read_gguf(MOE)
        idx = ec.build_expert_index(m)
        self.assertEqual(idx.n_experts, 128)
        self.assertEqual(len(idx.layers), 48)
        self.assertEqual(len(idx.slices), 48 * 128)
        slices = idx.get(0, 0)
        self.assertEqual(len(slices), 3)  # gate, up, down
        self.assertTrue(all(s.nbytes > 0 for s in slices))

    def test_lru_rispetta_il_budget(self):
        m = read_gguf(MOE)
        idx = ec.build_expert_index(m)
        reader = ec.SliceReader(real=False)
        budget = 64 * MB
        cache = ec.LRUExpertCache(budget, idx, reader)
        for layer in range(4):
            for e in range(40):
                cache.get(layer, e)
        self.assertLessEqual(cache.resident, budget)
        self.assertGreater(cache.stats.evictions, 0)
        self.assertLessEqual(cache.stats.peak_resident, budget)
        reader.close()

    def test_lru_riusa_le_slice_calde(self):
        m = read_gguf(MOE)
        idx = ec.build_expert_index(m)
        reader = ec.SliceReader(real=False)
        cache = ec.LRUExpertCache(512 * MB, idx, reader)
        for _ in range(10):
            for e in range(4):
                cache.get(0, e)
        self.assertEqual(cache.stats.misses, 4)
        self.assertEqual(cache.stats.hits, 36)
        self.assertAlmostEqual(cache.stats.hit_rate, 0.9, places=5)
        reader.close()

    def test_budget_nullo_rifiutato(self):
        m = read_gguf(MOE)
        idx = ec.build_expert_index(m)
        with self.assertRaises(ValueError):
            ec.LRUExpertCache(0, idx, ec.SliceReader(real=False))

    def test_simulazione_io_reale(self):
        m = read_gguf(MOE)
        res = ec.simulate(m, cache_budget=512 * MB, tokens=3, skew=1.0, real_io=True)
        self.assertGreater(res.cache["lookups"], 0)
        self.assertGreater(res.disk_bytes, 0)
        self.assertEqual(res.layers, 48)
        self.assertEqual(res.topk, 8)

    def test_cache_grande_batte_cache_piccola(self):
        m = read_gguf(MOE)
        piccola = ec.simulate(m, cache_budget=256 * MB, tokens=6, skew=1.0, real_io=False,
                              prefetch=False, seed=1)
        grande = ec.simulate(m, cache_budget=4 * GB, tokens=6, skew=1.0, real_io=False,
                             prefetch=False, seed=1)
        self.assertGreater(grande.cache["hit_rate"], piccola.cache["hit_rate"])
        self.assertLess(grande.disk_bytes, piccola.disk_bytes)

    def test_residency_plan(self):
        m = read_gguf(MOE)
        r = ec.residency_plan(m, 5 * GB)
        self.assertEqual(r["experts"], 128)
        self.assertGreater(r["cache_slots"], 0)
        self.assertGreater(r["cold_bytes"], 0)
        self.assertLessEqual(r["coverage"], 1.0)

    def test_modello_dense_non_simula(self):
        m = read_gguf(DENSE)
        res = ec.simulate(m, cache_budget=1 * GB, tokens=2)
        self.assertEqual(res.cache, {})
        self.assertTrue(res.notes)


# ---------------------------------------------------------------- runtime ---

class TestRuntimeArgs(unittest.TestCase):
    class FakeRT:
        build = "b0"
        backend = "cpu"

        def binary(self, kind="server"):
            return Path("/fake/llama-server")

        def _help_text(self):
            return "--flash-attn [on|off|auto] --n-cpu-moe N --jinja --override-tensor"

        def supports_flag(self, flag):
            return flag in self._help_text()

    class RTConCacheRam(FakeRT):
        def _help_text(self):
            return (FakeRT._help_text(self) if False else
                    "--flash-attn [on|off|auto] --n-cpu-moe N --jinja "
                    "--override-tensor --cache-ram N --cache-reuse N")

    def test_la_cache_dei_prompt_ha_un_tetto_nostro(self):
        """llama-server tiene in RAM gli stati dei prompt gia' elaborati per non
        rifare il prefill, e il suo default e' 8192 MiB. Argine non passava
        --cache-ram, quindi su una macchina da 16 GB meta' del budget poteva
        sparire fuori da ogni contabilita' del planner. Il tetto di RAM rigido e'
        il requisito centrale del progetto: non puo' avere un buco deciso da
        qualcun altro."""
        m = read_gguf(DENSE)
        p = planmod.plan(m, hw_16gb(), planmod.PlanRequest(backend="cpu", ctx=2048))
        self.assertGreater(p.estimate.prompt_cache, 0,
                           "con spazio a sufficienza la cache dei prompt c'e'")
        args = srv.build_llama_args(self.RTConCacheRam(), m.path, p,
                                    "127.0.0.1", 9999)
        self.assertIn("--cache-ram", args)
        passato = int(args[args.index("--cache-ram") + 1])
        self.assertEqual(passato, p.estimate.prompt_cache >> 20)
        self.assertLess(passato, 8192, "piu' del default di llama.cpp: inutile")

    def test_il_tetto_si_passa_anche_quando_e_zero(self):
        """Non passare il flag e passarlo a zero non sono la stessa cosa: senza,
        vale il default di 8192 MiB."""
        m = read_gguf(MOE)
        p = planmod.plan(m, hw_16gb(), planmod.PlanRequest(
            backend="cpu", ctx=4096, prompt_cache_states=0))
        self.assertEqual(p.estimate.prompt_cache, 0)
        args = srv.build_llama_args(self.RTConCacheRam(), m.path, p,
                                    "127.0.0.1", 9999)
        self.assertEqual(args[args.index("--cache-ram") + 1], "0")

    def test_la_cache_dei_prompt_e_il_primo_sacrificio(self):
        """Rinunciarci costa secondi di prefill, non capacita' del modello:
        va sacrificata prima del contesto, del batch e della precisione KV."""
        m = read_gguf(MOE)
        p = planmod.plan(m, hw_16gb(), planmod.PlanRequest(backend="cpu", ctx=4096))
        self.assertEqual(p.estimate.prompt_cache, 0,
                         "su un 30B in 16 GB deve essere la prima a cadere")
        self.assertEqual(p.ctx, 4096, "e il contesto non deve essere stato toccato")
        self.assertTrue(any("cache dei prompt" in d for d in p.decisions),
                        "la rinuncia va dichiarata, non fatta in silenzio")

    def test_la_cache_dei_prompt_entra_nel_totale(self):
        """Contarla nel piano ma non nel totale sarebbe peggio che non contarla:
        darebbe l'impressione di averla considerata."""
        m = read_gguf(DENSE)
        p = planmod.plan(m, hw_16gb(), planmod.PlanRequest(backend="cpu", ctx=2048))
        somma = (p.estimate.weights_resident + p.estimate.kv_cache
                 + p.estimate.activations + p.estimate.runtime
                 + p.estimate.expert_cache + p.estimate.repack
                 + p.estimate.draft + p.estimate.prompt_cache)
        self.assertEqual(p.estimate.total, somma)

    def test_uno_stato_costa_quanto_la_kv_cache(self):
        """E' lo stesso tensore copiato altrove: se le due stime divergono, una
        delle due e' sbagliata."""
        m = read_gguf(MOE)
        kv = planmod.kv_cache_bytes(m, 4096, "f16", "f16")
        self.assertEqual(planmod.prompt_cache_bytes(m, 4096, "f16", "f16", 1), kv)
        self.assertEqual(planmod.prompt_cache_bytes(m, 4096, "f16", "f16", 3), kv * 3)

    def test_args_base(self):
        m = read_gguf(DENSE)
        p = planmod.plan(m, hw_16gb(), planmod.PlanRequest(backend="cpu"))
        args = srv.build_llama_args(self.FakeRT(), m.path, p, "127.0.0.1", 9999)
        self.assertIn("--model", args)
        self.assertIn("--ctx-size", args)
        self.assertEqual(args[args.index("--port") + 1], "9999")
        self.assertEqual(args[args.index("--cache-type-k") + 1], "f16")
        self.assertIn("--flash-attn", args)
        self.assertEqual(args[args.index("--flash-attn") + 1], "on")

    def test_args_moe_su_cpu(self):
        m = read_gguf(MOE)
        p = planmod.plan(m, hw_16gb(), planmod.PlanRequest(backend="vulkan"))
        args = srv.build_llama_args(self.FakeRT(), m.path, p, "127.0.0.1", 1)
        if p.gpu_layers > 0:
            self.assertIn("--n-cpu-moe", args)

    def test_args_no_mmap_e_mlock(self):
        m = read_gguf(DENSE)
        p = planmod.plan(m, hw_16gb(), planmod.PlanRequest(backend="cpu", mmap=False, mlock=True))
        args = srv.build_llama_args(self.FakeRT(), m.path, p, "127.0.0.1", 1)
        self.assertIn("--no-mmap", args)
        self.assertIn("--mlock", args)

    def test_no_jinja(self):
        m = read_gguf(DENSE)
        p = planmod.plan(m, hw_16gb(), planmod.PlanRequest(backend="cpu", jinja=False))
        args = srv.build_llama_args(self.FakeRT(), m.path, p, "127.0.0.1", 1)
        self.assertNotIn("--jinja", args)

    def test_nomi_asset_attesi(self):
        for (plat, backend), patterns in rtmod.ASSET_PATTERNS.items():
            self.assertTrue(patterns, f"{plat}/{backend} senza pattern")
        name = rtmod._guess_asset_name("b10456", "windows", "vulkan")
        self.assertEqual(name, "llama-b10456-bin-win-vulkan-x64.zip")
        import re
        rx = re.compile(rtmod.ASSET_PATTERNS[("windows", "vulkan")][0])
        self.assertTrue(rx.match(name))
        cpu = rtmod._guess_asset_name("b10456", "windows", "cpu")
        self.assertTrue(re.compile(rtmod.ASSET_PATTERNS[("windows", "cpu")][0]).match(cpu))


# ------------------------------------------------------------------ probe ---

class TestHardware(unittest.TestCase):
    def test_detect_non_esplode(self):
        hw = hwmod.detect()
        self.assertIn("cpu", hw.backends)
        self.assertIsInstance(hw.to_dict(), dict)

    def test_resolve_backend(self):
        hw = hw_16gb()
        self.assertEqual(hwmod.resolve_backend("cpu", hw)[0], "cpu")
        self.assertEqual(hwmod.resolve_backend("vulkan", hw)[0], "vulkan")
        self.assertEqual(hwmod.resolve_backend("auto", hw)[0], "vulkan")
        hw.vulkan_runtime = False
        self.assertEqual(hwmod.resolve_backend("auto", hw)[0], "cpu")

    def test_shared_memory_igpu(self):
        hw = hw_16gb()
        self.assertEqual(hw.gpu_shared_memory_budget, 8 * GB)

    def test_cpu_ibrida(self):
        hw = hw_16gb()                      # 12 fisici / 16 logici, come i5-1340P
        self.assertTrue(hw.is_hybrid_cpu)
        hw2 = hwmod.HardwareReport()
        hw2.cpu_cores_physical, hw2.cpu_cores_logical = 8, 16   # SMT su tutti
        self.assertFalse(hw2.is_hybrid_cpu)
        hw3 = hwmod.HardwareReport()
        hw3.cpu_cores_physical, hw3.cpu_cores_logical = 8, 8    # niente SMT
        self.assertFalse(hw3.is_hybrid_cpu)

    def test_threads_su_cpu_ibrida(self):
        """Un thread per P-core, non due.

        Questo test chiedeva 8 (4 P-core x 2 thread) e passava, mentre la
        macchina vera andava il 40-46% piu' piano di quanto poteva. Misurato su
        Qwen3-30B-A3B in due quantizzazioni diverse: 4 thread danno 5.93 e 7.08
        tok/s, 8 ne danno 4.23 e 4.84, e la curva scende in modo monotono fino a
        12. ggml aspetta il thread piu' lento a ogni nodo del grafo: un E-core,
        o il fratello SMT di un P-core gia' occupato, aggiunge un partecipante
        alla barriera senza aggiungere lavoro fatto.
        """
        m = read_gguf(DENSE)
        hw = hw_16gb()
        hw.cpu_p_cores, hw.cpu_e_cores = 4, 8
        p = planmod.plan(m, hw, planmod.PlanRequest(backend="cpu"))
        self.assertEqual(p.threads, 4, "un thread per P-core")
        self.assertLessEqual(p.threads, hw.cpu_cores_physical)

    def test_il_prompt_puo_usare_piu_thread_della_generazione(self):
        """La barriera pesa sulla generazione, dove ogni token attraversa tutto
        il grafo. Il prompt lavora a batch e regge piu' thread: misurato 22.3
        contro 22.5 tok/s fra 4 e 8 thread di generazione, cioe' indifferente,
        perche' li' comanda threads_batch."""
        m = read_gguf(DENSE)
        hw = hw_16gb()
        hw.cpu_p_cores, hw.cpu_e_cores = 4, 8
        p = planmod.plan(m, hw, planmod.PlanRequest(backend="cpu"))
        self.assertGreater(p.threads_batch, p.threads)

    def test_threads_espliciti_vincono_sul_planner(self):
        m = read_gguf(DENSE)
        hw = hw_16gb()
        hw.cpu_p_cores, hw.cpu_e_cores = 4, 8
        p = planmod.plan(m, hw, planmod.PlanRequest(backend="cpu", threads=9))
        self.assertEqual(p.threads, 9)

    def test_budget_segue_la_ram_realmente_libera(self):
        """Pianificare su RAM occupata da altri processi = pianificare di paginare."""
        hw = hw_16gb()
        hw.ram_available = 5 * GB
        budget, notes = planmod.safe_budget(hw, None)
        self.assertLess(budget, 5 * GB)
        self.assertGreater(budget, 4 * GB)
        self.assertTrue(any("realmente" in n for n in notes))

    def test_budget_pieno_se_la_macchina_e_libera(self):
        hw = hw_16gb()
        hw.ram_available = 14 * GB
        budget, _notes = planmod.safe_budget(hw, None)
        self.assertGreaterEqual(budget, 10 * GB)

    def test_ram_limit_esplicito_ha_la_precedenza(self):
        hw = hw_16gb()
        hw.ram_available = 5 * GB
        budget, _notes = planmod.safe_budget(hw, 10 * GB)
        self.assertEqual(budget, 10 * GB)


class TestExpertStore(unittest.TestCase):
    """L'expert store diventa la sorgente dei pesi: un byte sbagliato non fa
    rumore, degrada il modello in silenzio. Questi test confrontano i byte veri
    contro il GGUF, non dimensioni o checksum del solo store."""

    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from make_tiny_model import make_tiny_moe

        cls.dir = Path(_TMP.name) / "store"  # type: ignore[union-attr]
        cls.dir.mkdir(exist_ok=True)
        cls.gguf = cls.dir / "tiny-moe.gguf"
        if not cls.gguf.exists():
            make_tiny_moe(cls.gguf)
        cls.model = read_gguf(str(cls.gguf))
        cls.store = cls.dir / "tiny-moe.psxs"
        if not cls.store.exists():
            es.write_store(cls.model, es.build_plan(cls.model), cls.store)

    def test_modello_moe_accettato(self):
        self.assertEqual(es.check_model(self.model), [])

    def test_modello_denso_rifiutato(self):
        dense = read_gguf(str(DENSE))
        problemi = es.check_model(dense)
        self.assertTrue(problemi)
        self.assertIn("MoE", problemi[0])

    def test_tre_tensori_per_layer(self):
        per_layer = es.expert_tensors(self.model)
        self.assertEqual(len(per_layer), self.model.n_layers)
        for per_kind in per_layer.values():
            self.assertEqual(set(per_kind), set(es.KINDS))

    def test_ogni_gruppo_e_allineato(self):
        idx = es.read_index(self.store)
        for e in idx.entries:
            self.assertEqual(e.offset % idx.alignment, 0,
                             "un offset non allineato rende impossibile "
                             "FILE_FLAG_NO_BUFFERING")
            self.assertEqual(e.size % idx.alignment, 0)
            self.assertGreaterEqual(e.size, e.raw_bytes)

    def test_ogni_slice_e_allineata_da_sola(self):
        """La ragione d'essere del formato v2: ogni lettura atterra direttamente
        nella memoria del suo tensore, senza staging ne' memcpy. Serve che
        l'offset di OGNI slice sia allineato, non solo quello del gruppo."""
        idx = es.read_index(self.store)
        for layer in idx.layers:
            for expert in (0, idx.n_experts - 1):
                e = idx.entry(layer, expert)
                for kind in es.KINDS:
                    self.assertEqual(e.kind_offset(kind) % idx.alignment, 0,
                                     f"{kind} non allineato: NO_BUFFERING lo "
                                     "rifiuterebbe")
                # le slice non si sovrappongono
                ordinate = sorted(zip(e.offsets, e.kind_bytes))
                for (o1, s1), (o2, _s2) in zip(ordinate, ordinate[1:]):
                    self.assertGreaterEqual(o2, o1 + s1)

    def test_verifica_scopre_un_offset_disallineato(self):
        storto = self.dir / "storto.psxs"
        dati = bytearray(self.store.read_bytes())
        import struct as _s
        base = es.HEADER_BYTES + len(es.read_index(self.store).layers) * 4
        off, = _s.unpack_from("<Q", dati, base)
        _s.pack_into("<Q", dati, base, off + 1)      # non piu' multiplo di 4096
        storto.write_bytes(bytes(dati))
        problemi = es.verify(self.model, storto, samples=4)
        self.assertTrue(any("non allineato" in p for p in problemi))

    def test_stima_coincide_col_piano(self):
        self.assertEqual(es.estimate_bytes(self.model),
                         es.build_plan(self.model).total_bytes)

    def test_verifica_byte_per_byte(self):
        self.assertEqual(es.verify(self.model, self.store, samples=32), [])

    def test_ordine_fisico_libero_ma_pesi_identici(self):
        """La proprieta' che rende il riordino gratuito: l'indice fa da
        indirezione, quindi cambiare il layout non cambia i pesi letti."""
        rovesciato = {l: list(reversed(range(self.model.expert_count)))
                      for l in range(self.model.n_layers)}
        alt = self.dir / "rovesciato.psxs"
        es.write_store(self.model, es.build_plan(self.model, order=rovesciato), alt)
        self.assertEqual(es.verify(self.model, alt, samples=32), [])
        a = es.read_index(self.store)
        b = es.read_index(alt)
        self.assertNotEqual(a.entry(0, 0).offset, b.entry(0, 0).offset)
        self.assertGreater(b.entry(0, 0).offset, b.entry(0, self.model.expert_count - 1).offset)
        for layer in (0, self.model.n_layers - 1):
            for expert in (0, 3, self.model.expert_count - 1):
                for kind in es.KINDS:
                    self.assertEqual(
                        es.read_slice(self.store, a, layer, expert, kind),
                        es.read_slice(alt, b, layer, expert, kind),
                        "layout diverso, pesi diversi: l'indirezione e' rotta")

    def test_store_corrotto_viene_scoperto(self):
        rotto = self.dir / "rotto.psxs"
        rotto.write_bytes(bytearray(self.store.read_bytes()))
        idx = es.read_index(rotto)
        off = idx.entry(1, 2).kind_offset("up")
        with open(rotto, "r+b") as fh:
            fh.seek(off + 64)
            originale = fh.read(1)
            fh.seek(off + 64)
            fh.write(bytes([originale[0] ^ 0xFF]))
        problemi = es.verify(self.model, rotto, samples=200)
        self.assertTrue(problemi, "un byte girato non e' stato rilevato")
        self.assertTrue(any("differiscono" in p for p in problemi))

    def test_store_troncato_rifiutato(self):
        corto = self.dir / "corto.psxs"
        dati = self.store.read_bytes()
        corto.write_bytes(dati[: len(dati) - 8192])
        with self.assertRaises(es.StoreError):
            es.read_index(corto)

    def test_file_estraneo_rifiutato(self):
        finto = self.dir / "finto.psxs"
        finto.write_bytes(b"non uno store" + b"\0" * 400)
        with self.assertRaises(es.StoreError):
            es.read_index(finto)

    def test_store_di_un_altro_modello_rifiutato(self):
        # stesso formato, modello diverso: deve accorgersene dalla dimensione
        idx = es.read_index(self.store)
        finto = self.dir / "altro.psxs"
        dati = bytearray(self.store.read_bytes())
        import struct as _s
        _s.pack_into("<Q", dati, 4 * 8 + 8 * 4, 12345)  # campo model_bytes
        finto.write_bytes(bytes(dati))
        problemi = es.verify(self.model, finto, samples=4)
        self.assertTrue(any("non sono lo stesso modello" in p for p in problemi))
        self.assertEqual(idx.model_bytes, self.gguf.stat().st_size)

    def test_indice_rifiuta_expert_inesistente(self):
        idx = es.read_index(self.store)
        with self.assertRaises(es.StoreError):
            idx.entry(0, 999)
        with self.assertRaises(es.StoreError):
            idx.entry(999, 0)

    def test_allineamento_non_potenza_di_due_rifiutato(self):
        with self.assertRaises(es.StoreError):
            es.build_plan(self.model, alignment=3000)

    def test_destinazione_uguale_al_modello_rifiutata(self):
        with self.assertRaises(es.StoreError):
            es.write_store(self.model, es.build_plan(self.model), self.gguf)

    def test_confronto_delle_quattro_strade(self):
        res = es.bench_paths(self.model, self.store, threads=2, seconds=0.3)
        etichette = {r.label for r in res}
        self.assertEqual(etichette, {"mmap", "gguf", "storeb", "store"})
        for r in res:
            if not r.error:
                self.assertGreater(r.experts, 0)
                self.assertGreater(r.reads, 0)
        # tre letture per expert su entrambi i lati: cosi' `gguf` e `store`
        # differiscono solo per allineamento e cache, e il confronto misura
        # una cosa sola
        per = {r.label: r for r in res if not r.error}
        for etichetta in ("gguf", "storeb", "store"):
            if etichetta in per:
                self.assertAlmostEqual(
                    per[etichetta].reads / max(1, per[etichetta].experts), 3.0,
                    delta=0.01,
                    msg=f"{etichetta}: il numero di letture per expert deve essere "
                        "lo stesso su tutte le strade, altrimenti si confrontano "
                        "due variabili insieme")
        # l'esposizione alla cache va marcata, o il verdetto confronta mele e pere
        self.assertTrue(per["gguf"].cached)
        self.assertTrue(per["storeb"].cached)
        self.assertFalse(per["store"].cached)

    def _res(self, label, mb_s, experts_s, cached, phys=None):
        sec = 1.0
        return es.PathResult(
            label, int(mb_s * 1e6 * sec), sec, int(experts_s * sec), 8,
            int(experts_s * sec), None if phys is None else int(phys * 1e6 * sec),
            cached=cached)

    def test_verdetto_rifiuta_il_confronto_che_mi_ha_ingannato(self):
        """gguf bufferizzato contro store diretto: due variabili insieme.
        Il tool deve dire che non misura niente, non dare un numero."""
        res = [
            self._res("mmap", 2149, 753, True, 1432),
            self._res("gguf", 6670, 2334, True, 2027),
            self._res("store", 4055, 1420, False, 4056),
        ]
        testo = " ".join(es.paths_verdict(res, group_bytes=2 << 20))
        self.assertIn("non e' disponibile a parita' di cache", testo)
        self.assertNotIn("0.61x", testo)

    def test_verdetto_confronta_il_layout_a_parita_di_cache(self):
        res = [
            self._res("gguf", 6670, 2000, True, 2027),
            self._res("storeb", 8000, 3000, True, 2500),
            self._res("store", 4055, 2800, False, 4056),
        ]
        righe = es.paths_verdict(res, group_bytes=2 << 20)
        testo = " ".join(righe)
        self.assertIn("1.50x", testo)          # 3000 / 2000, effetto del layout
        self.assertIn("scavalcare la cache", testo)  # 2800 / 3000, effetto della cache

    def test_verdetto_riporta_solo_il_disco_dei_percorsi_non_bufferizzati(self):
        res = [
            self._res("gguf", 9999, 5000, True, 2027),   # logico gonfiato dalla cache
            self._res("store", 4055, 1420, False, 4056),
        ]
        testo = " ".join(es.paths_verdict(res, group_bytes=2 << 20))
        self.assertIn("4056 MB/s", testo)
        self.assertNotIn("9999", testo)

    def test_proiezione_include_il_pavimento_di_calcolo(self):
        # una proiezione di I/O senza il calcolo e' una promessa che non si mantiene
        res = [self._res("store", 4055, 1420, False, 4056)]
        testo = " ".join(es.paths_verdict(res, group_bytes=int(2.92 * (1 << 20))))
        self.assertIn("resta il calcolo", testo)
        self.assertIn("49 ms", testo)

    def test_metriche_di_cache_sul_risultato(self):
        r = self._res("gguf", 1000, 100, True, 300)
        self.assertAlmostEqual(r.cache_ratio, 0.7, places=2)
        self.assertAlmostEqual(r.phys_mb_s, 300, places=0)
        senza = es.PathResult("x", 100, 1.0, 1, 1, 1, None)
        self.assertIsNone(senza.cache_ratio)
        self.assertIsNone(senza.phys_mb_s)


class TestExpertStoreStatus(unittest.TestCase):
    """La rilevazione di "lo store e' davvero in uso?".

    Esiste perche' lo strumento ha affermato due volte il falso, in direzioni
    opposte. Prima ha detto "expert serviti dallo store" solo per aver impostato
    una variabile d'ambiente. Poi, corretto, ha detto "NON attivo" quando lo era:
    cercava le righe dei buffer di llama.cpp, che senza -v non vengono stampate.
    Da qui la distinzione fra tre stati e la scelta di provarli con righe che il
    nostro codice scrive direttamente su stderr.
    """

    APERTO = ("argine: expert store aperto: C:/m/x.psxs, 48 layer x 128 expert, "
              "budget 6144 MiB, cache dell'OS scavalcata")
    IN_USO = ("argine: expert store IN USO: primo nodo MoE servito "
              "(layer 0, blk.0.ffn_gate_exps.weight)")
    STATS = ("argine: expert cache: 1200 hit, 300 miss (80.0% hit), 250 sfratti, "
             "800 MiB letti dallo store, 6000/6144 MiB residenti alla fine")

    def _proc(self, buffer_lines=(), psx_lines=()):
        p = srv.LlamaServerProcess(["finto"])
        p._load_lines = list(buffer_lines)
        p._psx_lines = list(psx_lines)
        return p

    def test_tre_stati_distinti(self):
        niente = self._proc().expert_store_status()
        self.assertFalse(niente["aperto"])
        self.assertFalse(niente["in_uso"])

        aperto = self._proc(psx_lines=[self.APERTO]).expert_store_status()
        self.assertTrue(aperto["aperto"])
        self.assertFalse(aperto["in_uso"], "aprire lo store non significa usarlo: "
                                           "--no-repack lo apre e non lo usa")

        in_uso = self._proc(psx_lines=[self.APERTO, self.IN_USO]).expert_store_status()
        self.assertTrue(in_uso["in_uso"])

    def test_in_uso_implica_aperto_anche_senza_la_riga(self):
        st = self._proc(psx_lines=[self.IN_USO]).expert_store_status()
        self.assertTrue(st["aperto"])
        self.assertTrue(st["in_uso"])

    def test_assenza_delle_righe_dei_buffer_non_prova_niente(self):
        # e' l'errore che ha prodotto il falso negativo: senza -v llama-server
        # non stampa "model buffer size", quindi la sua assenza non e' un dato
        st = self._proc(buffer_lines=[], psx_lines=[self.APERTO, self.IN_USO])
        st = st.expert_store_status()
        self.assertTrue(st["in_uso"])
        self.assertEqual(st["bytes"], 0)

    def test_buffer_pieno_da_la_dimensione(self):
        st = self._proc(
            buffer_lines=["load_tensors: CPU_ARGINE_EXPERTS model buffer size = 16740.00 MiB"],
            psx_lines=[self.APERTO, self.IN_USO],
        ).expert_store_status()
        self.assertGreater(st["bytes"], 16 * GB)

    def test_buffer_a_zero_e_la_passata_di_stima(self):
        st = self._proc(
            buffer_lines=["load_tensors: CPU_ARGINE_EXPERTS model buffer size = 0.00 MiB"],
            psx_lines=[self.APERTO],
        ).expert_store_status()
        self.assertEqual(st["bytes"], 0)
        self.assertFalse(st["in_uso"])

    def test_statistiche_della_cache_estratte(self):
        st = self._proc(psx_lines=[self.APERTO, self.IN_USO, self.STATS])
        st = st.expert_store_status()
        self.assertIsNotNone(st["stats"])
        self.assertIn("1200 hit", st["stats"])
        self.assertNotIn("argine:", st["stats"])

    def test_si_tiene_l_ultimo_resoconto_non_il_primo(self):
        """Il resoconto si ripete ogni 5 secondi: contano le ultime righe.
        La prima versione teneva le prime 40 e buttava via proprio quelle."""
        primo = "argine: expert cache (in corso): 10 hit, 5 miss (66.7% hit), 0 sfratti"
        ultimo = "argine: expert cache (in corso): 9000 hit, 500 miss (94.7% hit), 80 sfratti"
        tempo = "argine: tempo (ms, somma sui thread): lettura 900 (70%), commit 300 (23%)"
        st = self._proc(psx_lines=[self.APERTO, self.IN_USO, primo, "x", ultimo, tempo])
        st = st.expert_store_status()
        self.assertIn("9000 hit", st["stats"])
        self.assertEqual(st["resoconto"], [ultimo, tempo])

    def test_senza_resoconto_la_lista_e_vuota(self):
        st = self._proc(psx_lines=[self.APERTO, self.IN_USO]).expert_store_status()
        self.assertEqual(st["resoconto"], [])

    def test_errore_di_apertura_conservato(self):
        st = self._proc(psx_lines=[
            "argine: expert store non utilizzabile: store in versione 1, questa build "
            "legge la 2"]).expert_store_status()
        self.assertFalse(st["aperto"])
        self.assertTrue(any("non utilizzabile" in r for r in st["righe"]))


class _FakeRuntime:
    """Runtime finto che dichiara di conoscere --no-repack, cosi' il test puo'
    verificare se l'argomento arriva davvero a llama-server."""

    build = "b0"
    backend = "cpu"

    def binary(self, kind="server"):
        return Path("/fake/llama-server")

    def _help_text(self):
        return "--flash-attn [on|off|auto] --no-repack --jinja --n-cpu-moe N"

    def supports_flag(self, flag):
        return flag in self._help_text()


class TestRepackNonSpegnibileConStore(unittest.TestCase):
    """La scala di sicurezza non deve poter spegnere il repack quando l'expert
    store e' attivo.

    E' il bug che ha reso inutili due misure. `--no-repack` in llama.cpp non
    spegne il repack: imposta `no_extra_bufts`, che disabilita TUTTI gli extra
    buffer type, incluso quello dello store. La scala lo usava come primo
    gradino di risparmio, perche' libera due terzi dei byte di un modello Q4 —
    una scelta giusta in generale e distruttiva con lo store, perche' il
    risultato e' uno store aperto che non serve un solo tensore, senza che
    niente lo dica.
    """

    def _hw(self, libera_gib):
        hw = hw_16gb()
        hw.ram_available = int(libera_gib * GB)
        return hw

    def test_sotto_pressione_il_repack_resta_acceso(self):
        m = read_gguf(str(MOE))
        # budget molto sotto il necessario: la scala scende tutti i gradini
        req = planmod.PlanRequest(ram_limit=6 * GB, backend="cpu",
                                 repack=True, expert_store=True)
        p = planmod.plan(m, self._hw(7), req)
        self.assertTrue(p.repack,
                        "la scala ha spento il repack: con lo store equivale a "
                        "spegnere lo store")
        self.assertTrue(any("repack tenuto acceso" in d for d in p.decisions))

    def test_senza_store_la_scala_lo_spegne_come_prima(self):
        m = read_gguf(str(MOE))
        req = planmod.PlanRequest(ram_limit=6 * GB, backend="cpu", repack=True)
        p = planmod.plan(m, self._hw(7), req)
        self.assertFalse(p.repack)
        self.assertTrue(any("repack SIMD disattivato" in d for d in p.decisions))

    def test_con_lo_store_il_repack_si_accende_anche_se_non_richiesto(self):
        m = read_gguf(str(MOE))
        req = planmod.PlanRequest(backend="cpu", expert_store=True)  # repack=None
        p = planmod.plan(m, self._hw(10), req)
        self.assertTrue(p.repack)

    def test_il_piano_porta_il_flag_dello_store(self):
        m = read_gguf(str(MOE))
        p = planmod.plan(m, self._hw(10),
                         planmod.PlanRequest(backend="cpu", expert_store=True))
        self.assertTrue(p.expert_store)
        p2 = planmod.plan(m, self._hw(10), planmod.PlanRequest(backend="cpu"))
        self.assertFalse(p2.expert_store)

    def test_gli_argomenti_di_llama_non_contengono_no_repack(self):
        """La prova che conta: l'argomento non deve arrivare a llama-server."""
        m = read_gguf(str(MOE))
        p = planmod.plan(m, self._hw(7),
                         planmod.PlanRequest(ram_limit=6 * GB, backend="cpu",
                                             expert_store=True))
        rt = _FakeRuntime()
        args = srv.build_llama_args(rt, str(MOE), p, "127.0.0.1", 8080)
        self.assertNotIn("--no-repack", args)

    def test_senza_store_no_repack_arriva_eccome(self):
        m = read_gguf(str(MOE))
        p = planmod.plan(m, self._hw(7),
                         planmod.PlanRequest(ram_limit=6 * GB, backend="cpu"))
        rt = _FakeRuntime()
        args = srv.build_llama_args(rt, str(MOE), p, "127.0.0.1", 8080)
        self.assertIn("--no-repack", args)


class TestBinarioSenzaStore(unittest.TestCase):
    """Un llama-server ufficiale ignora le variabili dell'expert store in
    silenzio: il benchmark parte, gira e restituisce un numero che descrive il
    percorso normale mentre si crede di misurare lo store. E' costato due
    misure, quindi va rilevato prima di partire e non dopo."""

    def setUp(self):
        self.dir = Path(_TMP.name) / "rt"  # type: ignore[union-attr]
        self.dir.mkdir(exist_ok=True)

    def _runtime(self, contenuto: bytes, nome: str) -> rtmod.LlamaRuntime:
        d = self.dir / nome
        d.mkdir(exist_ok=True)
        for cand in ("llama-server", "llama-server.exe"):
            (d / cand).write_bytes(contenuto)
        return rtmod.LlamaRuntime(root=d, backend="cpu", build=nome)

    def test_binario_patchato_riconosciuto(self):
        rt = self._runtime(b"x" * 1000 + b"ARGINE_EXPERT_STORE" + b"y" * 1000, "patch")
        self.assertTrue(rt.supports_expert_store())

    def test_binario_ufficiale_riconosciuto(self):
        rt = self._runtime(b"llama.cpp ufficiale" * 5000, "ufficiale")
        self.assertFalse(rt.supports_expert_store())

    def test_marcatore_a_cavallo_di_due_blocchi(self):
        # il file si legge a blocchi di 4 MiB: il marcatore puo' cadere in mezzo,
        # e una lettura ingenua non lo troverebbe
        blocco = 4 << 20
        marc = b"ARGINE_EXPERT_STORE"
        dati = b"z" * (blocco - 5) + marc + b"z" * 100
        rt = self._runtime(dati, "cavallo")
        self.assertTrue(rt.supports_expert_store())

    def test_risultato_memorizzato(self):
        rt = self._runtime(b"niente" * 1000, "cache")
        self.assertFalse(rt.supports_expert_store())
        # il file cambia, ma la risposta resta quella gia' calcolata
        for cand in ("llama-server", "llama-server.exe"):
            (self.dir / "cache" / cand).write_bytes(b"ARGINE_EXPERT_STORE")
        self.assertFalse(rt.supports_expert_store())

    def test_marcatore_in_una_libreria_condivisa(self):
        """Su Linux ggml e' in libggml-cpu.so e l'eseguibile non contiene il
        marcatore: guardare solo il binario dava un falso "non supportato"
        proprio sulla build con cui stavo provando."""
        d = self.dir / "condivisa"
        d.mkdir(exist_ok=True)
        for cand in ("llama-server", "llama-server.exe"):
            (d / cand).write_bytes(b"eseguibile senza marcatore" * 100)
        (d / "libggml-cpu.so.0.25.3").write_bytes(b"a" * 500 + b"ARGINE_EXPERT_STORE")
        rt = rtmod.LlamaRuntime(root=d, backend="cpu", build="condivisa")
        self.assertTrue(rt.supports_expert_store())

    def test_marcatore_in_una_dll(self):
        d = self.dir / "dll"
        d.mkdir(exist_ok=True)
        for cand in ("llama-server", "llama-server.exe"):
            (d / cand).write_bytes(b"niente" * 100)
        (d / "ggml-cpu.dll").write_bytes(b"ARGINE_EXPERT_STORE")
        rt = rtmod.LlamaRuntime(root=d, backend="cpu", build="dll")
        self.assertTrue(rt.supports_expert_store())

    def test_binario_mancante_non_esplode(self):
        d = self.dir / "vuota"
        d.mkdir(exist_ok=True)
        rt = rtmod.LlamaRuntime(root=d, backend="cpu", build="vuota")
        self.assertFalse(rt.supports_expert_store())


class TestBudgetUnico(unittest.TestCase):
    """Il piano e lo store devono usare lo STESSO numero per gli expert.

    Quando non lo facevano, il piano dichiarava 6.37 GiB di expert e lo store ne
    prendeva 8: picco al 106% del budget, 206 MiB di RAM libera, e un benchmark
    che misurava una macchina che paginava. Il calo del 13% fra prima e ultima
    ripetizione l'avevo attribuito al calore, senza poterlo distinguere dalla
    paginazione.
    """

    def setUp(self):
        self.m = read_gguf(str(MOE))
        self.hw = hw_16gb()
        self.hw.ram_available = 10 * GB

    def test_budget_esplicito_entra_nel_piano(self):
        p = planmod.plan(self.m, self.hw, planmod.PlanRequest(
            backend="cpu", expert_store=True, expert_store_budget=4 * GB))
        self.assertEqual(p.expert_cache_budget, 4 * GB)
        self.assertEqual(p.estimate.expert_cache, 4 * GB,
                         "il totale del piano deve contenere il budget vero")

    def test_budget_eccessivo_avvisa(self):
        p = planmod.plan(self.m, self.hw, planmod.PlanRequest(
            backend="cpu", expert_store=True, expert_store_budget=30 * GB))
        self.assertTrue(any("paginare" in w for w in p.warnings))

    def test_senza_budget_esplicito_decide_il_planner(self):
        p = planmod.plan(self.m, self.hw,
                         planmod.PlanRequest(backend="cpu", expert_store=True))
        self.assertGreater(p.expert_cache_budget, 0)
        self.assertTrue(any("decisa dal" in d for d in p.decisions))

    def test_il_budget_non_supera_gli_expert_del_modello(self):
        p = planmod.plan(self.m, self.hw, planmod.PlanRequest(
            backend="cpu", expert_store=True, expert_store_budget=999 * GB))
        self.assertLessEqual(p.expert_cache_budget, self.m.expert_bytes)

    def test_il_cli_passa_allo_store_il_numero_del_piano(self):
        import argparse as _a
        p = planmod.plan(self.m, self.hw,
                         planmod.PlanRequest(backend="cpu", expert_store=True))
        store = Path(_TMP.name) / "finto.psxs"  # type: ignore[union-attr]
        store.write_bytes(b"x")
        args = _a.Namespace(expert_store=str(store), expert_budget=None,
                            expert_direct=True, expert_stats=False, verbose=False)
        env = cli._expert_store_env(args, self.m, None, p)
        self.assertEqual(env["ARGINE_EXPERT_BUDGET"], str(p.expert_cache_budget))

    def test_il_budget_esplicito_ha_la_precedenza_nel_cli(self):
        import argparse as _a
        p = planmod.plan(self.m, self.hw,
                         planmod.PlanRequest(backend="cpu", expert_store=True))
        store = Path(_TMP.name) / "finto2.psxs"  # type: ignore[union-attr]
        store.write_bytes(b"x")
        args = _a.Namespace(expert_store=str(store), expert_budget="3G",
                            expert_direct=True, expert_stats=False, verbose=False)
        env = cli._expert_store_env(args, self.m, None, p)
        self.assertEqual(env["ARGINE_EXPERT_BUDGET"], str(3 * GB))


class TestRepackConflitto(unittest.TestCase):
    """--no-repack in llama.cpp disabilita TUTTI gli extra buffer type, non solo
    il repack: con lo store attivo va tenuto acceso, o lo store non riceve
    nessun tensore. E' il bug che ha fatto sembrare inutile una funzione che
    funzionava."""

    def _args(self, **kw):
        import argparse as _a
        base = dict(no_repack=False, repack=False, expert_store=None)
        base.update(kw)
        return _a.Namespace(**base)

    def test_default_senza_store_resta_al_planner(self):
        self.assertIsNone(cli._decidi_repack(self._args()))

    def test_no_repack_esplicito_senza_store(self):
        self.assertIs(cli._decidi_repack(self._args(no_repack=True)), False)

    def test_con_lo_store_il_repack_resta_acceso(self):
        self.assertIs(cli._decidi_repack(self._args(expert_store="auto")), True)

    def test_no_repack_con_lo_store_viene_ignorato(self):
        # preferisco 230 MiB di repack a una funzione silenziosamente spenta
        self.assertIs(
            cli._decidi_repack(self._args(no_repack=True, expert_store="auto")), True)

    def test_repack_esplicito_vince_sempre(self):
        self.assertIs(cli._decidi_repack(self._args(repack=True)), True)


class TestPerplessita(unittest.TestCase):
    """La perplessita' e' il numero con cui giustificheremo di scambiare qualita'
    per velocita'. Se il confronto e' sbagliato, lo scambio e' cieco."""

    def test_l_avanzamento_arriva_prima_della_fine(self):
        """llama-perplexity stampa `[1]7.12,[2]6.98,...` senza andare a capo
        fino al termine. Leggendo per righe l'avanzamento non arrivava mai, e
        per quaranta minuti il programma sembrava bloccato — due volte, e la
        seconda dopo che avevo promesso che si sarebbe visto."""
        class Flusso:
            def __init__(self, blocchi):
                self.b, self.i = list(blocchi), 0

            def read(self, _n):
                if self.i >= len(self.b):
                    return ""
                d = self.b[self.i]
                self.i += 1
                return d

        f = Flusso(["system_info: n_threads = 4\n", "[1]7.12", "34,",
                    "[2]6.9", "876,[3]7.0001,",
                    "\nFinal estimate: PPL = 6.9877 +/- 0.04321\n"])
        visti = []
        for p in qmod._pezzi(f, dimensione=4096):
            m = qmod._PROGRESS_RE.search(p)
            if m:
                visti.append((int(m.group(1)), float(m.group(2))))
        self.assertEqual(visti, [(1, 7.1234), (2, 6.9876), (3, 7.0001)])

    def test_il_pezzo_a_meta_non_viene_consegnato_spezzato(self):
        """Un numero tagliato a meta' fra due letture deve ricomporsi, non
        diventare due avanzamenti sbagliati."""
        class Flusso:
            def __init__(self, blocchi):
                self.b, self.i = list(blocchi), 0

            def read(self, _n):
                if self.i >= len(self.b):
                    return ""
                d = self.b[self.i]
                self.i += 1
                return d

        pezzi = list(qmod._pezzi(Flusso(["[1]7.1", "234,[2]", "6.98,"])))
        self.assertIn("[1]7.1234,", pezzi)
        self.assertIn("[2]6.98,", pezzi)

    def test_il_riferimento_e_il_migliore_non_il_piu_recente(self):
        """Successo davvero: tre modelli in gioco — Q4, un IQ2 mal calibrato e un
        IQ2 ricalibrato. Misurando il terzo, la vecchia regola ('l'ultima misura
        su un modello diverso') ha scelto il secondo e annunciato che il
        candidato era migliore, mentre il confronto che contava era contro il
        Q4. Il riferimento e' cio' che si cerca di non peggiorare."""
        storico = [
            self._res("q4.gguf", 12.06, 0.291),
            self._res("iq2-scarso.gguf", 16.44, 0.406),
        ]
        rif = qmod.trova_riferimento(storico, storico[0].firma,
                                     escludi="iq2-buono.gguf")
        self.assertEqual(Path(rif.model).name, "q4.gguf")

    def test_si_puo_scegliere_il_riferimento_a_mano(self):
        storico = [self._res("q4.gguf", 12.06), self._res("iq2.gguf", 16.44)]
        rif = qmod.trova_riferimento(storico, storico[0].firma, nome="iq2.gguf")
        self.assertEqual(Path(rif.model).name, "iq2.gguf")

    def test_un_riferimento_inesistente_non_inventa_un_altro(self):
        storico = [self._res("q4.gguf", 12.06)]
        self.assertIsNone(qmod.trova_riferimento(storico, storico[0].firma,
                                                 nome="mai-visto.gguf"))

    def test_confrontabili_tiene_una_misura_per_modello(self):
        """Rimisurando lo stesso modello vale l'ultima, non due righe."""
        storico = [
            self._res("q4.gguf", 12.50),
            self._res("iq2.gguf", 16.44),
            self._res("q4.gguf", 12.06),      # rimisura
        ]
        c = qmod.confrontabili(storico, storico[0].firma)
        self.assertEqual(len(c), 2)
        q4 = [r for r in c if Path(r.model).name == "q4.gguf"][0]
        self.assertAlmostEqual(q4.ppl, 12.06)
        self.assertLess(c[0].ppl, c[1].ppl, "ordinati dal migliore")

    def test_firme_diverse_restano_fuori(self):
        storico = [self._res("a.gguf", 12.0, chunk=60),
                   self._res("b.gguf", 9.0, chunk=18)]
        c = qmod.confrontabili(storico, storico[0].firma)
        self.assertEqual([Path(r.model).name for r in c], ["a.gguf"])

    def test_battere_una_quantizzazione_peggiore_non_e_sospetto(self):
        """Il messaggio diceva 'risultato sospetto' ogni volta che il candidato
        vinceva. Fra due quantizzazioni e' il risultato normale, e quella riga
        ha gia' fatto scambiare un miglioramento vero per un errore."""
        c = qmod.Confronto(riferimento=self._res("iq2-scarso.gguf", 16.44, 0.406),
                           candidato=self._res("iq2-buono.gguf", 15.19, 0.371))
        testo = " ".join(c.giudizio())
        self.assertIn("riuscita meglio", testo)

    def _res(self, model, ppl, err=0.05, ctx=512, chunk=100, h="abc"):
        return qmod.PPLResult(model=model, ppl=ppl, errore=err, ctx=ctx,
                              chunk=chunk, testo_hash=h)

    def test_firma_include_tutto_cio_che_cambia_il_numero(self):
        a = self._res("m", 6.0)
        self.assertNotEqual(a.firma, self._res("m", 6.0, ctx=1024).firma)
        self.assertNotEqual(a.firma, self._res("m", 6.0, chunk=50).firma)
        self.assertNotEqual(a.firma, self._res("m", 6.0, h="xyz").firma)
        self.assertEqual(a.firma, self._res("altro", 9.9).firma,
                         "il modello non entra nella firma: e' cio' che si confronta")

    def test_confronto_rifiutato_se_i_parametri_differiscono(self):
        c = qmod.Confronto(self._res("q4", 6.0), self._res("iq2", 7.0, ctx=1024))
        self.assertFalse(c.confrontabile)
        self.assertIn("NON sono confrontabili", " ".join(c.giudizio()))

    def test_differenza_dentro_il_rumore_non_e_un_risultato(self):
        # 6.00 +/- 0.05 contro 6.03 +/- 0.05: le barre si sovrappongono
        c = qmod.Confronto(self._res("q4", 6.00), self._res("iq2", 6.03))
        self.assertTrue(c.confrontabile)
        self.assertFalse(c.significativo)
        self.assertIn("dentro l'incertezza", " ".join(c.giudizio()))

    def test_degrado_grande_e_significativo(self):
        c = qmod.Confronto(self._res("q4", 6.00), self._res("iq2", 7.50))
        self.assertTrue(c.significativo)
        self.assertAlmostEqual(c.delta_pct, 25.0, places=1)
        self.assertIn("degrado", " ".join(c.giudizio()))

    def test_candidato_migliore_del_riferimento(self):
        """Va segnalato, ma senza decidere che sia un errore: dipende da cosa e'
        il riferimento, e il codice non lo sa. La versione precedente diceva
        sempre "sospetto" e ha fatto scambiare un IQ2 meglio calibrato, che
        batteva legittimamente il precedente, per un file da ricontrollare."""
        c = qmod.Confronto(self._res("q4", 7.00), self._res("iq2", 6.00))
        testo = " ".join(c.giudizio())
        self.assertIn("migliore del riferimento", testo)
        self.assertIn("precisione piu' alta", testo)

    def test_soglie_del_giudizio(self):
        # barre d'errore strette, altrimenti la differenza sparisce nel rumore:
        # con +/-0.05 un +1.67% NON e' significativo, e il codice ha ragione a
        # dirlo. E' il test che era sciatto, non la soglia.
        for ppl, atteso in ((6.10, "piccolo"), (6.40, "percepibile"),
                            (7.00, "forte"), (9.00, "grave")):
            c = qmod.Confronto(self._res("q4", 6.00, err=0.005),
                               self._res("x", ppl, err=0.005))
            self.assertIn(atteso, " ".join(c.giudizio()), f"ppl {ppl}")

    def test_un_punto_e_mezzo_percento_con_barre_larghe_non_conta(self):
        """La stessa differenza e' un risultato o rumore a seconda della
        precisione della misura: e' il motivo per cui il giudizio guarda le
        barre d'errore e non solo la percentuale."""
        stretto = qmod.Confronto(self._res("q4", 6.00, err=0.005),
                                 self._res("x", 6.10, err=0.005))
        largo = qmod.Confronto(self._res("q4", 6.00, err=0.05),
                               self._res("x", 6.10, err=0.05))
        self.assertTrue(stretto.significativo)
        self.assertFalse(largo.significativo)

    def test_storico_scritto_e_riletto(self):
        p = Path(_TMP.name) / "ppl.jsonl"  # type: ignore[union-attr]
        qmod.salva(self._res("uno", 6.0), p)
        qmod.salva(self._res("due", 6.5), p)
        letti = qmod.carica(p)
        self.assertEqual(len(letti), 2)
        self.assertEqual(letti[1].ppl, 6.5)
        self.assertEqual(letti[0].firma, letti[1].firma)

    def test_riferimento_e_l_ultimo_di_un_altro_modello(self):
        storico = [self._res("a", 6.0), self._res("b", 6.2), self._res("a", 6.1)]
        r = qmod.trova_riferimento(storico, storico[0].firma, escludi="a")
        self.assertEqual(r.model, "b")
        # nessun riferimento se l'unico presente e' il modello stesso
        self.assertIsNone(qmod.trova_riferimento([self._res("a", 6.0)],
                                                 storico[0].firma, escludi="a"))

    def test_estrazione_dal_log_di_llama(self):
        righe = ["[1]6.9,", "[2]6.85,", "Final estimate: PPL = 6.9877 +/- 0.04321"]
        m = qmod._PPL_RE.search("\n".join(righe))
        self.assertIsNotNone(m)
        self.assertAlmostEqual(float(m.group(1)), 6.9877)
        self.assertAlmostEqual(float(m.group(2)), 0.04321)

    def test_impronta_del_testo_distingue_contenuti(self):
        a = Path(_TMP.name) / "t1.txt"  # type: ignore[union-attr]
        b = Path(_TMP.name) / "t2.txt"  # type: ignore[union-attr]
        a.write_text("uno" * 500)
        b.write_text("due" * 500)
        self.assertNotEqual(qmod.hash_testo(a), qmod.hash_testo(b))
        self.assertEqual(qmod.hash_testo(a), qmod.hash_testo(a))

    def test_testo_di_ripiego_generato_una_volta_sola(self):
        p = Path(_TMP.name) / "ripiego.txt"  # type: ignore[union-attr]
        qmod.testo_di_prova(p, parole=500)
        primo = p.read_text()
        qmod.testo_di_prova(p, parole=500)
        self.assertEqual(primo, p.read_text())
        self.assertGreater(len(primo.split()), 400)

    def test_il_testo_di_ripiego_non_si_ripete(self):
        """Il primo che avevo scritto ciclava dieci frasi: dopo un giro anche un
        modello rotto le predice, la perplessita' esce bassa per tutti e il
        confronto non distingue niente. Era inutile per l'unica cosa che deve
        fare."""
        p = Path(_TMP.name) / "vario.txt"  # type: ignore[union-attr]
        qmod.testo_di_prova(p, parole=2000)
        self.assertGreater(qmod.diversita(p), 0.9)

    def test_diversita_riconosce_un_testo_ciclico(self):
        p = Path(_TMP.name) / "ciclico.txt"  # type: ignore[union-attr]
        p.write_text(". ".join(["frase uno", "frase due"] * 200) + ".")
        self.assertLess(qmod.diversita(p), 0.1)

    def test_un_testo_ciclico_gia_presente_viene_rigenerato(self):
        p = Path(_TMP.name) / "daRifare.txt"  # type: ignore[union-attr]
        p.write_text(". ".join(["sempre la stessa frase"] * 300) + ".")
        qmod.testo_di_prova(p, parole=1000)
        self.assertGreater(qmod.diversita(p), 0.9)

    def test_i_campi_hardware_usati_dal_cli_esistono(self):
        """Un nome di campo inventato passa i test finche' nessuno percorre
        quel ramo: `hw.cpu_physical` non e' mai esistito (si chiama
        cpu_cores_physical) ed e' arrivato fino alla macchina dell'utente."""
        hw = hw_16gb()
        for campo in ("cpu_cores_physical", "cpu_cores_logical", "ram_total",
                      "ram_available", "cpu_brand"):
            self.assertTrue(hasattr(hw, campo), f"manca {campo}")
        self.assertFalse(hasattr(hw, "cpu_physical"),
                         "se questo campo esistesse, il test sopra non direbbe nulla")

    def test_i_thread_per_la_perplessita_vengono_dal_planner(self):
        m = read_gguf(str(MOE))
        p = planmod.plan(m, hw_16gb(), planmod.PlanRequest(backend="cpu", ctx=512))
        self.assertGreater(p.threads, 0)
        self.assertLessEqual(p.threads, 64)

    def test_argomenti_di_llama_perplexity(self):
        args = qmod.build_args(Path("/x/llama-perplexity"), "m.gguf", "t.txt",
                               ctx=512, threads=8, chunks=50)
        self.assertIn("--ctx-size", args)
        self.assertEqual(args[args.index("--chunks") + 1], "50")
        self.assertEqual(args[args.index("--n-gpu-layers") + 1], "0",
                         "senza offload: la misura deve dipendere dal modello")


class TestLeveDiskBench(unittest.TestCase):
    """La conclusione di `diskbench` su *come* leggere.

    Diceva "conviene emettere piu' letture insieme piuttosto che letture piu'
    grandi" quando il dispositivo scalava coi thread. La misura era giusta e il
    consiglio inservibile: descriveva il dispositivo a 32 thread, mentre lo store
    lavora a 0,75 miss per nodo, cioe' a profondita' di coda ~1, dove letture da
    sovrapporre non ce ne sono — e dove le letture grandi rendono il 59% in piu'.
    Il verdetto consigliava esattamente la leva non disponibile e sconsigliava
    quella che paga."""

    PICCOLA = 917504      # un tensore expert
    GRANDE = 2883584      # la tripletta

    def _leve(self, dati, scelto, coda=1.0):
        res = [db.DiskResult(kind="moe", slice_bytes=s, threads=t,
                             bytes_logical=int(mb * 1e6), seconds=1.0)
               for s, t, mb in dati]
        suite = db.Suite.__new__(db.Suite)
        suite.results = res
        deciso = next(r for r in res if (r.slice_bytes, r.threads) == scelto)
        return "\n".join(suite._leve(deciso, coda))

    def test_a_coda_uno_consiglia_le_letture_grandi(self):
        """Il caso della macchina di riferimento: +59% passando da 896 KiB a
        2,75 MiB con un thread solo. Prima qui diceva il contrario."""
        out = self._leve([(self.PICCOLA, 1, 300), (self.GRANDE, 1, 477),
                          (self.PICCOLA, 32, 1800), (self.GRANDE, 32, 2000)],
                         (self.GRANDE, 32))
        self.assertIn("LEVA DA TIRARE: letture piu' grandi", out)
        self.assertIn("1.59x", out)
        self.assertIn("niente da sovrapporre", out,
                      "va detto perche' il parallelismo non si puo' usare")

    def test_a_coda_due_e_mezzo_entrambe_le_leve_sono_in_gioco(self):
        """La coda misurata davvero ai budget che contano per un modello piu'
        grande: 2,09 a 4 GiB e 2,49 a 1,5 GiB, non ~1. A quel regime il
        parallelismo e' in parte disponibile, e cresce da se' al calare della
        residenza perche' i miss aumentano."""
        out = self._leve([(self.PICCOLA, 1, 300), (self.GRANDE, 1, 477),
                          (self.PICCOLA, 32, 1800), (self.GRANDE, 32, 2000)],
                         (self.GRANDE, 32), coda=2.49)
        self.assertIn("entrambe le leve", out)
        self.assertIn("si tira da se'", out)
        self.assertNotIn("LEVA DA TIRARE", out,
                         "a questa coda non si sceglie una leva sola")

    def test_dice_che_le_due_leve_si_scambiano(self):
        """Terzo difetto dello stesso verdetto: presentava le leve come
        indipendenti. A byte totali fissi per nodo non lo sono — letture piu'
        grandi sono anche meno numerose, quindi alzare la dimensione abbassa la
        coda. Misurato: unire le tre matrici in una lettura da 2,75 MiB porta la
        coda da 2,49 a ~0,83, e a quella coda il dispositivo da' 2351 MB/s contro
        i 3522 che darebbe con le stesse letture piccole mandate in volo insieme."""
        out = self._leve([(self.PICCOLA, 1, 1338), (self.PICCOLA, 8, 3563),
                          (self.GRANDE, 1, 2351), (self.GRANDE, 8, 4070)],
                         (self.GRANDE, 8), coda=2.49)
        self.assertIn("si scambiano", out)
        self.assertIn("4070", out, "va indicata la cella migliore della tabella")

    def test_senza_la_coda_osservata_il_verdetto_tace(self):
        """Il regime non e' una costante del programma, e' un dato della misura.
        Prima era inchiodato a 0,75 miss per nodo — un valore preso al 49% di
        residenza — e il verdetto che ne derivava era tarato su un regime
        sbagliato, esattamente come quello che aveva sostituito."""
        out = self._leve([(self.PICCOLA, 1, 300), (self.GRANDE, 1, 477),
                          (self.PICCOLA, 32, 1800), (self.GRANDE, 32, 2000)],
                         (self.GRANDE, 32), coda=None)
        self.assertNotIn("LEVA DA TIRARE", out)
        self.assertNotIn("entrambe le leve", out)
        self.assertIn("--coda", out, "va detto come ottenere il verdetto")
        self.assertIn("--expert-stats", out, "e da dove viene il numero")
        self.assertNotIn("0.75", out)

    def test_non_consiglia_le_letture_grandi_se_non_rendono(self):
        """Il verdetto deve dipendere dalla misura, non dall'aver imparato la
        lezione precedente: se a coda 1 le letture grandi non rendono, dirlo."""
        out = self._leve([(self.PICCOLA, 1, 300), (self.GRANDE, 1, 305),
                          (self.PICCOLA, 32, 1800), (self.GRANDE, 32, 2000)],
                         (self.GRANDE, 32))
        self.assertNotIn("LEVA DA TIRARE", out)
        self.assertIn("ridurre i miss", out)

    def test_dispositivo_saturo_a_un_thread(self):
        out = self._leve([(self.PICCOLA, 1, 1000), (self.GRANDE, 1, 1010),
                          (self.PICCOLA, 4, 1050), (self.GRANDE, 4, 1060)],
                         (self.GRANDE, 4))
        self.assertIn("NESSUNA DELLE DUE LEVE", out)

    def test_il_vecchio_consiglio_non_ricompare(self):
        """La frase esatta che era sbagliata, in tutti e tre i regimi."""
        casi = (
            [(self.PICCOLA, 1, 300), (self.GRANDE, 1, 477),
             (self.PICCOLA, 32, 1800), (self.GRANDE, 32, 2000)],
            [(self.PICCOLA, 1, 300), (self.GRANDE, 1, 305),
             (self.PICCOLA, 32, 1800), (self.GRANDE, 32, 2000)],
            [(self.PICCOLA, 1, 1000), (self.GRANDE, 1, 1010),
             (self.PICCOLA, 4, 1050), (self.GRANDE, 4, 1060)],
        )
        for dati in casi:
            scelto = max(((r[0], r[1]) for r in dati), key=lambda x: x[1])
            out = self._leve(dati, scelto)
            self.assertNotIn("piuttosto che letture piu' grandi", out)

    def test_con_una_sola_dimensione_di_slice_non_inventa_la_seconda_leva(self):
        out = self._leve([(self.GRANDE, 1, 477), (self.GRANDE, 32, 2000)],
                         (self.GRANDE, 32))
        self.assertNotIn("letture piu' grandi':", out)


class TestDiskBench(unittest.TestCase):
    """La misura dello storage. Verifica la meccanica e la logica del verdetto:
    un verdetto sbagliato manderebbe settimane di lavoro nella direzione
    sbagliata, quindi le sue soglie sono coperte da test."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.file = Path(_TMP.name) / "diskbench.bin"  # type: ignore[union-attr]
        with open(cls.file, "wb") as fh:
            fh.write(os.urandom(1 << 20) * 24)  # 24 MiB, abbastanza per le prove

    def test_offset_allineati_al_settore(self):
        nxt = db._offsets(100 << 20, 917504, seed=7)
        for _ in range(200):
            off = nxt()
            self.assertEqual(off % db.SECTOR, 0)
            self.assertLessEqual(off + 917504, 100 << 20)

    def test_semi_diversi_danno_sequenze_diverse(self):
        # se i thread condividessero la sequenza si sincronizzerebbero sulle
        # stesse pagine e la prova misurerebbe la cache invece del disco
        a = [db._offsets(1 << 30, 4096, 1)() for _ in range(20)]
        b = [db._offsets(1 << 30, 4096, 2)() for _ in range(20)]
        self.assertNotEqual(a, b)

    def test_mmap_legge_davvero(self):
        r = db.bench_mmap(self.file, 262144, threads=2, seconds=0.3)
        self.assertEqual(r.kind, "mmap")
        self.assertGreater(r.ops, 0)
        self.assertEqual(r.bytes_logical, r.ops * 262144)
        self.assertGreater(r.mb_s, 0)

    def test_pread_legge_davvero(self):
        r = db.bench_positioned(self.file, 262144, threads=2, seconds=0.3, direct=False)
        self.assertEqual(r.kind, "pread")
        self.assertGreater(r.ops, 0)
        self.assertGreater(r.bytes_logical, 0)

    def test_moe_ha_la_barriera_e_termina(self):
        # il rischio del pattern con barriera e' il deadlock a fine misura
        r = db.bench_moe(self.file, 262144, threads=4, seconds=0.3)
        self.assertEqual(r.kind, "moe")
        self.assertGreater(r.ops, 0)
        self.assertLess(r.seconds, 5.0)

    def test_metriche_derivate(self):
        r = db.DiskResult("direct", 1 << 20, 4, bytes_logical=400 << 20, seconds=2.0,
                          bytes_physical=200 << 20, ops=400)
        self.assertAlmostEqual(r.mb_s, (400 << 20) / 2.0 / 1e6, places=3)
        self.assertAlmostEqual(r.phys_mb_s, (200 << 20) / 2.0 / 1e6, places=3)
        self.assertAlmostEqual(r.iops, 200.0, places=3)
        self.assertAlmostEqual(r.cache_ratio, 0.5, places=3)
        # latenza per singolo thread: 2s * 4 thread / 400 op
        self.assertAlmostEqual(r.latency_ms, 20.0, places=3)

    def test_cache_ratio_assente_senza_contatore(self):
        r = db.DiskResult("mmap", 4096, 1, 4096, 1.0, None, 1)
        self.assertIsNone(r.cache_ratio)
        self.assertIsNone(r.phys_mb_s)

    def _suite_finta(self, moe_mb_s: float) -> db.Suite:
        s = db.Suite(self.file)
        secondi = 1.0
        byte = int(moe_mb_s * 1e6 * secondi)
        s.results = [
            db.DiskResult("seq", 8 << 20, 1, int(3000e6), secondi, int(3000e6), 400),
            db.DiskResult("mmap", 917504, 8, int(900e6), secondi, int(900e6), 900),
            db.DiskResult("direct", 917504, 1, byte // 2, secondi, byte // 2, 500),
            db.DiskResult("direct", 917504, 8, byte, secondi, byte, 1000),
            db.DiskResult("moe", 917504, 8, byte, secondi, byte, 1000),
        ]
        return s

    def test_verdetto_favorevole_se_il_disco_ha_margine(self):
        s = self._suite_finta(moe_mb_s=2000.0)  # 3.6x i 562 di produzione
        testo = " ".join(s.verdict())
        self.assertIn("vale il lavoro", testo)

    def test_verdetto_contrario_se_il_disco_e_saturo(self):
        s = self._suite_finta(moe_mb_s=600.0)  # 1.07x: niente da guadagnare
        testo = " ".join(s.verdict())
        self.assertIn("va cancellato", testo)
        self.assertIn("ridurre i byte per token", testo)

    def test_verdetto_prudente_nella_zona_grigia(self):
        s = self._suite_finta(moe_mb_s=562.0 * 1.5)
        testo = " ".join(s.verdict())
        self.assertIn("margine stretto", testo)

    def test_verdetto_usa_il_pattern_realistico_non_il_migliore(self):
        # direct senza barriera e' generoso: il verdetto deve basarsi su moe
        s = db.Suite(self.file)
        s.results = [
            db.DiskResult("direct", 917504, 8, int(3000e6), 1.0, int(3000e6), 3000),
            db.DiskResult("moe", 917504, 8, int(600e6), 1.0, int(600e6), 600),
        ]
        testo = " ".join(s.verdict())
        self.assertIn("va cancellato", testo)

    def test_verdetto_segnala_la_cache_nel_controllo_mmap(self):
        s = db.Suite(self.file)
        s.results = [
            db.DiskResult("mmap", 917504, 8, int(20000e6), 1.0, 0, 20000),
            db.DiskResult("moe", 917504, 8, int(2000e6), 1.0, int(2000e6), 2000),
        ]
        testo = " ".join(s.verdict())
        self.assertIn("page cache", testo)

    def test_verdetto_si_rifiuta_senza_prove_dirette(self):
        s = db.Suite(self.file)
        s.results = [db.DiskResult("mmap", 917504, 8, int(900e6), 1.0, int(900e6), 900)]
        testo = " ".join(s.verdict())
        self.assertIn("non c'e' verdetto", testo)

    def test_baseline_sovrascrivibile(self):
        s = self._suite_finta(moe_mb_s=2000.0)
        s.PRODUZIONE_MB_S = 1900.0
        self.assertIn("va cancellato", " ".join(s.verdict()))

    def test_formattazione_include_gli_errori(self):
        righe = db.format_results([
            db.DiskResult("direct", 4096, 1, 0, 0.0, None, 0, error="O_DIRECT non supportato"),
        ])
        self.assertTrue(any("O_DIRECT" in r for r in righe))

    def test_suite_salta_le_prove_fallite_senza_fermarsi(self):
        s = db.Suite(self.file, slices=(262144,), threads=(1,), kinds=("mmap",),
                     seconds=0.2, sequential=False)
        s.run()
        self.assertEqual(len(s.results), 1)

    def test_purge_zero_non_fa_niente(self):
        self.assertEqual(db.purge_cache(0), 0)
        self.assertEqual(db.purge_cache(-5), 0)


class TestConversazione(unittest.TestCase):
    """Il banco che mancava: ogni altra misura manda prompt indipendenti, mentre
    una chat e' un prefisso che cresce. Progettare una cache del prefisso contro
    un banco che non riproduce il caso d'uso e' l'errore che questa classe serve
    a non far ripetere."""

    def test_i_turni_sono_diversi_fra_loro(self):
        """Turni identici renderebbero il prefisso innaturalmente comprimibile:
        misureremmo la ripetizione invece della conversazione."""
        testi = [benchmod._testo_turno(i) for i in range(1, 9)]
        self.assertEqual(len(set(testi)), len(testi))

    def test_il_prefisso_cresce_davvero(self):
        pezzi = [benchmod._testo_turno(i, parole=60) for i in range(1, 5)]
        for p in pezzi:
            self.assertGreaterEqual(len(p.split()), 60)

    def test_il_resoconto_mostra_il_rapporto_col_baseline(self):
        con = [benchmod.TurnoResult(turno=1, prompt_tokens=100, nuovi_tokens=100,
                                    ttft_s=1.0, cached_tokens=0),
               benchmod.TurnoResult(turno=2, prompt_tokens=200, nuovi_tokens=100,
                                    ttft_s=1.1, cached_tokens=100)]
        senza = [benchmod.TurnoResult(turno=1, prompt_tokens=100, ttft_s=1.0),
                 benchmod.TurnoResult(turno=2, prompt_tokens=200, ttft_s=4.4)]
        righe = " ".join(benchmod.format_conversazione(con, senza))
        self.assertIn("4.0x", righe, "il guadagno del riuso va mostrato come rapporto")

    def test_senza_baseline_non_inventa_rapporti(self):
        con = [benchmod.TurnoResult(turno=1, prompt_tokens=100, ttft_s=1.0)]
        righe = " ".join(benchmod.format_conversazione(con))
        self.assertNotIn("rapporto", righe)


class TestCambioBuild(unittest.TestCase):
    """Due velocita' prese su binari diversi non si possono confrontare, e
    niente lo segnalava. Su qwen30b-IQ2S, stesso file e stessi flag, la build
    ufficiale b11214 ha dato 13.53 tok/s e la nostra patchata 7.45: 1.82x
    attribuito per mezz'ora alla quantizzazione, perche' ARGINE_LLAMA_DIR si
    era persa aprendo un terminale nuovo."""

    def _log(self, righe):
        import json as _j
        d = Path(_TMP.name) / "build"        # type: ignore[union-attr]
        d.mkdir(exist_ok=True)
        p = d / f"runs-{len(righe)}-{abs(hash(str(righe)))}.jsonl"
        p.write_text("\n".join(_j.dumps(r) for r in righe) + "\n", encoding="utf-8")
        return p

    def _run(self, path, build, tok):
        return {"event": "benchmark", "ts_iso": "2026-10-01T19:00",
                "model": {"path": path, "name": "Qwen3"},
                "runtime": {"build": build}, "metrics": {"gen_tok_s": tok}}

    def test_rileva_il_cambio_di_build(self):
        p = self._log([self._run("C:/m/a.gguf", "b11214", 13.53)])
        prec = telemetry.ultima_build("C:/m/a.gguf", "ARGINE_LLAMA_DIR", p)
        self.assertEqual(prec["build"], "b11214")
        self.assertAlmostEqual(prec["tok_s"], 13.53)

    def test_tace_se_la_build_e_la_stessa(self):
        p = self._log([self._run("C:/m/a.gguf", "b11214", 13.53)])
        self.assertIsNone(telemetry.ultima_build("C:/m/a.gguf", "b11214", p))

    def test_non_confonde_due_quantizzazioni_dello_stesso_modello(self):
        """Quattro file derivati dallo stesso GGUF condividono `model.name` dai
        metadati: confrontare per nome li mescolerebbe tutti."""
        p = self._log([self._run("C:/m/q4.gguf", "ARGINE_LLAMA_DIR", 5.93),
                       self._run("C:/m/iq2s.gguf", "ARGINE_LLAMA_DIR", 7.45)])
        prec = telemetry.ultima_build("C:/m/iq2s.gguf", "b11214", p)
        self.assertAlmostEqual(prec["tok_s"], 7.45)

    def test_un_modello_mai_misurato_non_avvisa(self):
        p = self._log([self._run("C:/m/a.gguf", "b11214", 13.53)])
        self.assertIsNone(telemetry.ultima_build("C:/m/nuovo.gguf", "x", p))

    def test_due_cartelle_locali_hanno_etichette_diverse(self):
        """L'etichetta era la stringa fissa "ARGINE_LLAMA_DIR" per qualunque
        cartella: tre build compilate in modo diverso finivano nello storico
        indistinguibili, e il controllo sul cambio di binario era cieco proprio
        nel caso in cui serve."""
        base = Path(_TMP.name) / "build-locali"   # type: ignore[union-attr]
        for nome in ("llama-omp", "llama-noomp"):
            (base / nome).mkdir(parents=True, exist_ok=True)
            (base / nome / "llama-server").write_bytes(nome.encode())
        a = rtmod._etichetta_locale(base / "llama-omp")
        b = rtmod._etichetta_locale(base / "llama-noomp")
        self.assertNotEqual(a, b)
        self.assertTrue(a.startswith("llama-omp@"), a)

    def test_ricompilare_nella_stessa_cartella_cambia_l_etichetta(self):
        """Il modo piu' facile di cambiare binario senza accorgersene."""
        d = Path(_TMP.name) / "ricompilata"       # type: ignore[union-attr]
        d.mkdir(exist_ok=True)
        (d / "llama-server").write_bytes(b"prima")
        a = rtmod._etichetta_locale(d)
        (d / "llama-server").write_bytes(b"dopo")
        self.assertNotEqual(a, rtmod._etichetta_locale(d))

    def test_ignora_gli_eventi_che_non_sono_benchmark(self):
        r = self._run("C:/m/a.gguf", "b11214", 13.53)
        r["event"] = "perplexity"
        self.assertIsNone(telemetry.ultima_build("C:/m/a.gguf", "altro",
                                                 self._log([r])))


class TestQuantizzazione(unittest.TestCase):
    """La quantizzazione e' un'operazione che dura un'ora e fallisce in fondo.

    Tutti questi test esistono per spostare un fallimento dalla fine all'inizio:
    o il file non sarebbe entrato in RAM, o llama-quantize si sarebbe rifiutato,
    e in entrambi i casi e' meglio saperlo prima di leggere diciotto gigabyte.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.moe = read_gguf(str(MOE))
        cls.dense = read_gguf(str(DENSE))

    # --- le tabelle ---------------------------------------------------------

    def test_i_byte_per_blocco_coincidono_con_quelli_di_ggml(self):
        """La tabella dei tipi e' scritta due volte: qui e in gguf.py, che legge
        i file veri. Se divergono, la stima e' credibile e sbagliata — il modo
        peggiore di essere sbagliata. Questo test le tiene insieme."""
        da_gguf = {nome: (blk, size) for nome, blk, size in GGML_TYPES.values()}
        confrontati = 0
        for t in qzmod.tipi_disponibili():
            if t.solo_ftype or t.nome not in da_gguf:
                continue
            self.assertEqual((t.blocco, t.byte), da_gguf[t.nome],
                             f"{t.nome}: byte per blocco diversi fra "
                             f"quantize.py e gguf.py")
            confrontati += 1
        self.assertGreater(confrontati, 12, "il confronto non ha coperto nulla")

    def test_bit_per_peso_noti(self):
        """Valori pubblicati, ricalcolati dai byte per blocco: se una delle due
        cose e' sbagliata non coincidono."""
        attesi = {"IQ1_S": 1.5625, "IQ2_XXS": 2.0625, "IQ2_XS": 2.3125,
                  "IQ2_S": 2.5625, "Q2_K": 2.625, "IQ3_XXS": 3.0625,
                  "Q4_K": 4.5, "Q6_K": 6.5625, "Q8_0": 8.5}
        for nome, bpw in attesi.items():
            self.assertAlmostEqual(qzmod.tipo(nome).bpw, bpw, places=4, msg=nome)

    def test_il_nome_ggml_dei_k_quant_ha_la_K_maiuscola(self):
        """--tensor-type confronta la stringa con ggml_type_name() byte per
        byte. 'q4_k' non e' 'q4_K', e l'errore che ne esce parla di tipo
        sconosciuto senza dire quale grafia si aspettava."""
        self.assertEqual(qzmod.tipo("Q4_K").ggml_name, "q4_K")
        self.assertEqual(qzmod.tipo("Q2_K").ggml_name, "q2_K")
        self.assertEqual(qzmod.tipo("Q6_K").ggml_name, "q6_K")
        # gli IQ invece sono tutti minuscoli
        self.assertEqual(qzmod.tipo("IQ2_XXS").ggml_name, "iq2_xxs")

    # --- la stima -----------------------------------------------------------

    def test_la_stima_tocca_solo_gli_expert(self):
        st = qzmod.stima(self.moe, "IQ2_XXS")
        expert = st.voce("expert")
        altro = [v for v in st.voci if v.classe != "expert"]
        self.assertLess(expert.byte_dopo, expert.byte_prima * 0.55)
        for v in altro:
            self.assertEqual(v.byte_prima, v.byte_dopo,
                             f"{v.classe} non doveva cambiare")

    def test_la_stima_segue_i_bit_per_peso(self):
        """Il file finale deve stare entro l'1% di quello che dicono i byte per
        blocco: se non ci sta, il conto sta sbagliando qualcosa di strutturale
        (elementi, blocchi o classificazione dei tensori)."""
        prima = self.moe.expert_bytes
        st = qzmod.stima(self.moe, "IQ2_XXS")
        atteso = prima * (qzmod.tipo("IQ2_XXS").bpw / qzmod.tipo("Q4_K").bpw)
        self.assertLess(abs(st.byte_expert_dopo - atteso) / atteso, 0.01)

    def test_un_tipo_piu_piccolo_da_un_file_piu_piccolo(self):
        ordinati = ["IQ2_XXS", "IQ2_XS", "IQ2_S", "Q2_K", "IQ3_XXS", "Q4_K"]
        taglie = [qzmod.stima(self.moe, t).byte_dopo for t in ordinati]
        self.assertEqual(taglie, sorted(taglie), "l'ordine delle taglie non "
                                                 "segue l'ordine dei bit")

    def test_norm_e_router_non_vengono_mai_toccati(self):
        """Il router sceglie quali expert usare: un errore li' non calcola male,
        calcola l'expert sbagliato. llama.cpp lo lascia in F32 e la stima deve
        saperlo, altrimenti promette byte che non risparmiera'."""
        st = qzmod.stima(self.moe, "IQ2_XXS", resto_type="IQ2_XXS")
        v = st.voce("norm/router")
        self.assertIsNotNone(v)
        self.assertEqual(v.byte_prima, v.byte_dopo)

    def test_righe_non_multiple_del_blocco_vengono_segnalate(self):
        """Un tensore con righe non multiple di 256 non puo' usare un K-quant:
        llama.cpp ricade su un altro tipo e il file esce piu' grande del
        previsto. Meglio un avviso che una sorpresa."""
        piccolo = read_gguf(str(TINY_MOE))
        st = qzmod.stima(piccolo, "IQ2_XXS")
        self.assertTrue(any("multiple del blocco" in a for a in st.avvisi))

    # --- le precondizioni ---------------------------------------------------

    def test_iq2_senza_imatrix_viene_rifiutato_prima_di_partire(self):
        """Il controllo che vale piu' di tutti gli altri: llama.cpp non degrada
        con un avviso, carica il modello e poi muore."""
        p = qzmod.controlla(self.moe, "IQ2_XXS", imatrix=None,
                            consenti_riquantizzazione=True)
        self.assertFalse(p.ok)
        self.assertTrue(any("importanza" in e for e in p.errori))

    def test_q2_k_non_richiede_imatrix(self):
        p = qzmod.controlla(self.moe, "Q2_K", imatrix=None,
                            consenti_riquantizzazione=True)
        self.assertTrue(p.ok, p.errori)

    def test_riquantizzare_va_chiesto_esplicitamente(self):
        p = qzmod.controlla(self.moe, "Q2_K", imatrix=None,
                            consenti_riquantizzazione=False)
        self.assertFalse(p.ok)
        self.assertTrue(any("allow-requantize" in e for e in p.errori))

    def test_riquantizzare_avvisa_del_doppio_arrotondamento(self):
        p = qzmod.controlla(self.moe, "Q2_K", imatrix=None,
                            consenti_riquantizzazione=True)
        self.assertTrue(any("seconda volta" in a for a in p.avvisi),
                        "partire da Q4_K peggiora il risultato e va detto")

    def test_una_mescola_non_si_puo_assegnare_ai_soli_expert(self):
        """IQ2_M non e' un tipo di tensore ma una scelta che llama.cpp fa
        tensore per tensore: --tensor-type la rifiuta."""
        p = qzmod.controlla(self.moe, "IQ2_M", imatrix=Path(__file__),
                            consenti_riquantizzazione=True, usa_tensor_type=True)
        self.assertFalse(p.ok)
        self.assertTrue(any("mescola" in e for e in p.errori))

    def test_la_stessa_mescola_va_bene_su_tutto_il_modello(self):
        p = qzmod.controlla(self.moe, "IQ2_M", imatrix=Path(__file__),
                            consenti_riquantizzazione=True, usa_tensor_type=False)
        self.assertTrue(p.ok, p.errori)

    def test_matrice_indicata_ma_assente(self):
        p = qzmod.controlla(self.moe, "IQ2_XXS", imatrix=Path("/non/esiste.dat"),
                            consenti_riquantizzazione=True)
        self.assertFalse(p.ok)
        self.assertTrue(any("non trovata" in e for e in p.errori))

    # --- la riga di comando -------------------------------------------------

    def test_gli_expert_finiscono_tutti_e_tre_in_tensor_type(self):
        args = qzmod.build_args(Path("q"), "in.gguf", "out.gguf", "IQ2_XXS",
                                imatrix="i.dat", consenti_riquantizzazione=True)
        coppie = [args[i + 1] for i, a in enumerate(args) if a == "--tensor-type"]
        self.assertEqual(sorted(coppie), sorted([
            "ffn_down_exps=iq2_xxs", "ffn_gate_exps=iq2_xxs",
            "ffn_up_exps=iq2_xxs"]))

    def test_senza_resto_la_ftype_generale_non_rimpicciolisce_niente(self):
        """Gli ultimi tre argomenti sono posizionali e l'ultimo e' la ftype che
        governa i tensori non nominati. Se li' finisse IQ2, l'attenzione
        verrebbe quantizzata a 2 bit di nascosto."""
        args = qzmod.build_args(Path("q"), "in.gguf", "out.gguf", "IQ2_XXS",
                                imatrix="i.dat", consenti_riquantizzazione=True)
        self.assertEqual(args[-4:-1], ["in.gguf", "out.gguf", "Q4_K_M"])

    def test_senza_tensor_type_la_ftype_e_quella_degli_expert(self):
        args = qzmod.build_args(Path("q"), "in.gguf", "out.gguf", "IQ2_XXS",
                                imatrix="i.dat", usa_tensor_type=False)
        self.assertEqual(args[-2], "IQ2_XXS")
        self.assertNotIn("--tensor-type", args)

    def test_i_thread_sono_un_posizionale_non_un_opzione(self):
        """Tre strumenti della stessa famiglia, tre convenzioni: llama-server e
        llama-perplexity vogliono `--threads N`, llama-quantize vuole `nthreads`
        come quarto argomento posizionale. Passandogli `--threads` stampa
        l'elenco dei tipi ed esce con 1, senza dire quale argomento non gli
        piaceva."""
        args = qzmod.build_args(Path("q"), "in.gguf", "out.gguf", "Q2_K",
                                threads=12, usa_tensor_type=False)
        self.assertNotIn("--threads", args)
        self.assertEqual(args[-4:], ["in.gguf", "out.gguf", "Q2_K", "12"])

    def test_perplexity_e_imatrix_invece_i_thread_li_vogliono_come_opzione(self):
        """Il rovescio della medaglia: se un giorno si uniformasse quantize
        anche qui, questi due smetterebbero di funzionare in silenzio."""
        ppl = qmod.build_args(Path("p"), "m.gguf", "t.txt", 512, 8, None)
        self.assertIn("--threads", ppl)
        imat = qzmod.build_imatrix_args(Path("i"), "m.gguf", "t.txt", "o.dat",
                                        threads=8)
        self.assertIn("--threads", imat)

    def test_la_schermata_d_uso_viene_riconosciuta(self):
        """llama-quantize rifiuta un argomento stampando l'uso e uscendo con 1.
        Riconoscerlo e' la differenza fra 'codice 1' e una diagnosi."""
        finto = ["usage: llama-quantize [--help] ... type [nthreads]", "",
                 "  10  or  Q2_K : ..."]
        self.assertTrue(qzmod._e_uso(finto))
        self.assertFalse(qzmod._e_uso(["load: caricamento", "quantizzo..."]))

    def test_del_log_si_tengono_testa_e_coda(self):
        """Tenere solo le ultime righe conserva le pagine di aiuto e butta via
        la riga che dice cosa e' andato storto — che sta in cima. E' lo stesso
        errore, capovolto, gia' fatto con il log di llama-server."""
        righe = [f"riga {i}" for i in range(200)]
        tenute = qzmod._estremi(righe)
        self.assertIn("riga 0", tenute)
        self.assertIn("riga 199", tenute)
        self.assertLess(len(tenute), 60)
        self.assertTrue(any("omesse" in r for r in tenute))

    def test_allow_requantize_solo_se_chiesto(self):
        senza = qzmod.build_args(Path("q"), "a", "b", "Q2_K")
        self.assertNotIn("--allow-requantize", senza)
        con = qzmod.build_args(Path("q"), "a", "b", "Q2_K",
                               consenti_riquantizzazione=True)
        self.assertIn("--allow-requantize", con)

    def test_le_opzioni_precedono_i_posizionali(self):
        """llama-quantize legge le opzioni finche' non incontra il primo
        argomento che non comincia per '-': una opzione dopo il nome del file
        finisce interpretata come ftype."""
        args = qzmod.build_args(Path("q"), "in.gguf", "out.gguf", "Q2_K",
                                imatrix="i.dat", consenti_riquantizzazione=True)
        primo_posizionale = args.index("in.gguf")
        for a in args[primo_posizionale:]:
            self.assertFalse(a.startswith("--"),
                             f"{a} sta dopo i posizionali")

    # --- la scelta del tipo -------------------------------------------------

    def test_sceglie_il_piu_grande_che_ci_sta_non_il_piu_piccolo(self):
        """Ogni bit in meno costa qualita': scendere sotto il necessario la
        regala senza comprare niente."""
        scelto, _note = qzmod.scegli_tipo_per_budget(self.moe, 11 * GB)
        self.assertIsNotNone(scelto)
        taglia = qzmod.stima(self.moe, scelto).byte_dopo
        self.assertLessEqual(taglia, 11 * GB * 0.90)
        piu_grandi = [t for t in qzmod.tipi_disponibili()
                      if t.bpw > qzmod.tipo(scelto).bpw and not t.solo_ftype]
        for t in piu_grandi:
            if t.bpw > 5.0:
                continue
            self.assertGreater(qzmod.stima(self.moe, t.nome).byte_dopo,
                               11 * GB * 0.90,
                               f"{t.nome} ci stava ed e' stato scartato")

    def test_budget_impossibile_lo_dice(self):
        scelto, note = qzmod.scegli_tipo_per_budget(self.moe, 1 * GB)
        self.assertIsNone(scelto)
        self.assertTrue(any("modello piu' piccolo" in n for n in note))

    def test_il_testo_di_riferimento_non_viene_rigenerato_di_nascosto(self):
        """Il testo entra nella firma della perplessita': se una chiamata
        successiva lo riscrivesse, la misura di riferimento gia' presa
        diventerebbe inconfrontabile con tutte le altre, e nulla lo direbbe."""
        d = Path(_TMP.name) / "testo-stabile"       # type: ignore[union-attr]
        d.mkdir(exist_ok=True)
        f = d / "testo-ppl.txt"
        qmod.testo_di_prova(f)
        prima = qmod.hash_testo(f)
        for _ in range(3):
            qmod.testo_di_prova(f)
            self.assertEqual(qmod.hash_testo(f), prima)

    def test_un_testo_troppo_corto_per_i_blocchi_viene_allungato(self):
        """La matrice di importanza si calcola un blocco alla volta: se il testo
        finisce prima, i blocchi richiesti finiscono li' e gli expert mai
        attivati restano senza dati."""
        d = Path(_TMP.name) / "testo-lungo"         # type: ignore[union-attr]
        d.mkdir(exist_ok=True)
        f = d / "testo-imatrix.txt"
        qmod.testo_di_prova(f, parole=2000, seed=23)
        corto = f.stat().st_size
        qmod.testo_di_prova(f, parole=40000, seed=23)
        self.assertGreater(f.stat().st_size, corto * 10)

    def test_i_comandi_suggeriti_sono_eseguibili(self):
        """`imatrix` e `quantize` finiscono stampando il comando successivo. Li
        avevo scritti con `--model NOME`, ma il modello e' un argomento
        posizionale: chi copiava la riga suggerita si prendeva un errore di
        parsing. Un comando stampato e' un comando promesso, quindi il parser
        deve accettarlo davvero."""
        import ast

        albero = ast.parse(Path(cli.__file__).read_text(encoding="utf-8"))
        comandi = []
        for nodo in ast.walk(albero):
            # Le f-string spezzate su piu' righe arrivano gia' unite: e' il
            # motivo per cui si guarda l'albero e non il testo.
            if isinstance(nodo, ast.JoinedStr):
                pezzi = ["FINTO.gguf" if isinstance(p, ast.FormattedValue)
                         else p.value for p in nodo.values]
            elif isinstance(nodo, ast.Constant) and isinstance(nodo.value, str):
                pezzi = [nodo.value]
            else:
                continue
            testo = "".join(pezzi).strip()
            if testo.startswith("argine ") and len(testo.split()) > 2:
                comandi.append(testo)

        parser = cli.build_parser()
        noti = {s.strip() for s in
                ("quantize", "ppl", "imatrix", "store", "benchmark", "run",
                 "inspect", "experts", "diskbench", "reorder")}
        visti = 0
        for testo in comandi:
            argomenti = testo.split()[1:]
            if argomenti[0] not in noti:
                continue
            with self.subTest(comando=testo):
                parser.parse_args(argomenti)
            visti += 1
        self.assertGreaterEqual(visti, 2,
                                f"quasi nessun comando verificato: {comandi}")

    def test_lo_store_senza_budget_avvisa_che_non_c_e_tetto(self):
        """Sul lato C++ budget zero significa *nessuno sfratto*: la cache cresce
        fino a contenere tutti gli expert. Lo store continua a funzionare, ma il
        tetto rigido — la ragione per cui esiste — sparisce. Se succede va
        detto, non lasciato dedurre dalla RAM che si riempie."""
        import io
        import contextlib

        store = Path(_TMP.name) / "finto-budget.psxs"  # type: ignore[union-attr]
        store.write_bytes(b"x")
        args = argparse.Namespace(expert_store=str(store), expert_budget=None,
                                  expert_direct=True, expert_stats=False)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            env = cli._expert_store_env(args, self.moe, None, None)
        uscita = buf.getvalue()
        self.assertNotIn("ARGINE_EXPERT_BUDGET", env)
        self.assertIn("nessun tetto", uscita)

    def test_col_piano_il_budget_arriva_al_lato_c(self):
        import io
        import contextlib

        store = Path(_TMP.name) / "finto-budget2.psxs"  # type: ignore[union-attr]
        store.write_bytes(b"x")
        args = argparse.Namespace(expert_store=str(store), expert_budget=None,
                                  expert_direct=True, expert_stats=False)
        piano = planmod.RuntimePlan(expert_cache_budget=5 * GB, expert_store=True)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            env = cli._expert_store_env(args, self.moe, None, piano)
        self.assertEqual(env["ARGINE_EXPERT_BUDGET"], str(5 * GB))
        self.assertNotIn("nessun tetto", buf.getvalue())

    def test_ppl_e_imatrix_costruiscono_un_piano_con_lo_store(self):
        """Il piano non serve a lanciare llama-server: serve a far uscire un
        budget. `ppl` e `imatrix` lo passavano a None e il tetto spariva."""
        args = argparse.Namespace(ctx=512, ram_limit=None, expert_store="auto",
                                  expert_budget=None)
        piano = cli._piano_per_misura(self.moe, hw_16gb(), args)
        self.assertIsNotNone(piano)
        self.assertTrue(piano.expert_store)
        self.assertGreater(piano.expert_cache_budget, 0,
                           "un piano con lo store deve produrre un budget")

    def test_il_dense_avvisa_che_non_ha_expert(self):
        p = qzmod.controlla(self.dense, "Q2_K", imatrix=None,
                            consenti_riquantizzazione=True)
        self.assertTrue(any("non e' MoE" in a for a in p.avvisi))


class TestCurvaResidenza(unittest.TestCase):
    """`--sweep-expert-budget` esiste per rispondere a una domanda da decine di
    GiB di banda senza spenderli: cosa succede quando il set di expert e' molto
    piu' grande del budget. Un modello con tre volte gli expert e lo stesso
    budget si comporta come questo modello con un terzo del budget.

    Il README afferma che "con lo store la dimensione non ha un tetto". E' una
    previsione, e questo e' lo strumento che la mette alla prova."""

    def setUp(self):
        self.tmp = Path(_TMP.name) / "curva"  # type: ignore[union-attr]
        self.tmp.mkdir(exist_ok=True)

    def test_esponente_zero_se_la_residenza_non_conta(self):
        r = cli.esponente_residenza([{"frazione": 1.0, "gen_tok_s": 10.0},
                                     {"frazione": 0.25, "gen_tok_s": 10.0}])
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r[0], 0.0, places=6)
        self.assertEqual(len(r), 4, "serve anche l'incertezza")

    def test_esponente_uno_se_la_velocita_e_proporzionale(self):
        """Il caso in cui l'affermazione del README e' falsa: un modello tre
        volte piu' grande andrebbe tre volte piu' piano."""
        r = cli.esponente_residenza([{"frazione": 1.0, "gen_tok_s": 10.0},
                                     {"frazione": 0.25, "gen_tok_s": 2.5}])
        self.assertAlmostEqual(r[0], 1.0, places=6)
        self.assertAlmostEqual(r[1], 4.0, places=6)
        self.assertAlmostEqual(r[2], 4.0, places=6)

    def test_lesponente_non_dipende_dall_ordine_delle_righe(self):
        """Lo sweep gira dal budget piu' grande al piu' piccolo, ma una riga che
        non parte viene saltata: l'ordine delle righe sopravvissute non e'
        garantito, e una pendenza che cambia segno con l'ordine sarebbe un
        risultato inventato."""
        su = [{"frazione": 1.0, "gen_tok_s": 10.0}, {"frazione": 0.25, "gen_tok_s": 5.0}]
        giu = list(reversed(su))
        self.assertAlmostEqual(cli.esponente_residenza(su)[0],
                               cli.esponente_residenza(giu)[0], places=9)

    def test_lincertezza_cresce_quando_lescursione_si_restringe(self):
        """Il denominatore e' log(escursione): un intervallo corto amplifica il
        rumore. Misurato sui dati veri: 1,33x di escursione da 0,47 +/- 0,15,
        mentre 2,72x con la stessa rumorosita' da' +/- 0,04."""
        stretto = cli.esponente_residenza(
            [{"frazione": 0.2446, "gen_tok_s": 5.34},
             {"frazione": 0.1835, "gen_tok_s": 4.67}])
        largo = cli.esponente_residenza(
            [{"frazione": 0.49, "gen_tok_s": 9.0},
             {"frazione": 0.18, "gen_tok_s": 4.67}])
        self.assertGreater(stretto[3], largo[3] * 2,
                           "un'escursione tre volte piu' piccola deve dichiarare "
                           "un'incertezza molto maggiore")
        self.assertLess(stretto[1], cli.SPAN_MINIMO_PER_VERDETTO)
        self.assertGreater(largo[1], cli.SPAN_MINIMO_PER_VERDETTO)

    def test_nessun_verdetto_se_lintervallo_scavalca_una_soglia(self):
        """La prima versione stampava "la velocita' scende in modo sensibile" da
        un esponente 0,47 che valeva 0,47 +/- 0,15, cioe' [0,32, 0,61]: un
        intervallo che contiene la soglia 0,35 e quindi non distingue "regge" da
        "scende". Un verdetto che i dati non sostengono e' peggio di nessun
        verdetto, perche' viene creduto."""
        rows = [{"budget": 4 * GB, "frazione": 0.2446, "gen_tok_s": 5.34,
                 "ttft_s": 3.95, "ram_peak": None, "err_rel": None},
                {"budget": 3 * GB, "frazione": 0.1835, "gen_tok_s": 4.67,
                 "ttft_s": 4.02, "ram_peak": None, "err_rel": None}]
        esp, rr, _rv, inc = cli.esponente_residenza(rows)
        self.assertTrue(rr < cli.SPAN_MINIMO_PER_VERDETTO
                        or any(esp - inc < s < esp + inc for s in (0.35, 0.8)),
                        "su questi dati il verdetto va sospeso")

    def test_la_dispersione_si_misura_dalle_ripetizioni(self):
        """Meglio misurare la rumorosita' che assumerla; ma con meno di due
        ripetizioni non c'e' niente da stimare, e allora si usa il valore assunto
        *dichiarandolo*."""
        class FintaRun:
            def __init__(self, v):
                self.gen_tok_s = v

        class FintoRes:
            def __init__(self, vals):
                self.runs = [FintaRun(v) for v in vals]

        self.assertIsNone(cli._dispersione_relativa(FintoRes([])))
        self.assertIsNone(cli._dispersione_relativa(FintoRes([5.0])))
        d = cli._dispersione_relativa(FintoRes([5.0, 5.0, 5.0]))
        self.assertAlmostEqual(d, 0.0, places=9)
        d = cli._dispersione_relativa(FintoRes([4.8, 5.0, 5.2]))
        self.assertGreater(d, 0.0)
        self.assertLess(d, 0.1)
        self.assertIsNone(cli._dispersione_relativa(object()))

    def test_i_tempi_per_miss_si_leggono_dalla_riga_vera_del_cpp(self):
        """Il lato C++ stampa i millisecondi per miss da sempre, e nessuno li
        leggeva in modo strutturato. Sono il dato che spiega *perche'* la curva ha
        quella pendenza, invece di constatarla.

        Il test non si limita a una riga inventata: ricostruisce la riga dalla
        stringa di formato dentro argine-experts.cpp, perche' due formati che
        divergono in silenzio sono una classe di bug che questo progetto ha gia'
        pagato (i byte per blocco in due tabelle diverse)."""
        sorgente = (Path(__file__).resolve().parents[1]
                    / "patches" / "argine-experts.cpp").read_text(encoding="utf-8")
        self.assertIn("per miss: %.3f ms di lettura", sorgente,
                      "se il C++ cambia formato, il parser va aggiornato con lui")
        riga = ("argine: per miss: %.3f ms di lettura, %.3f ms di pagine nuove, "
                "%.3f ms di commit" % (1.337, 0.212, 0.370))
        t = cli._tempi_per_miss({"resoconto": ["altro rumore", riga]})
        self.assertAlmostEqual(t["ms_lettura_miss"], 1.337)
        self.assertAlmostEqual(t["ms_pagine_miss"], 0.212)
        self.assertAlmostEqual(t["ms_commit_miss"], 0.370)

        for vuoto in (None, {}, {"resoconto": []}, {"resoconto": ["niente"]}):
            self.assertIsNone(cli._tempi_per_miss(vuoto)["ms_lettura_miss"])

    def test_niente_pendenza_da_un_punto_solo(self):
        self.assertIsNone(cli.esponente_residenza([{"frazione": 1.0, "gen_tok_s": 10.0}]))
        self.assertIsNone(cli.esponente_residenza([]))
        self.assertIsNone(cli.esponente_residenza(
            [{"frazione": 1.0, "gen_tok_s": None}, {"frazione": 0.5, "gen_tok_s": None}]))

    def test_due_budget_identici_non_producono_una_pendenza(self):
        """Divisione per log(1) = 0. Capita se si scrive --sweep-expert-budget
        8G,8192M: due modi di dire lo stesso numero."""
        self.assertIsNone(cli.esponente_residenza(
            [{"frazione": 0.5, "gen_tok_s": 10.0}, {"frazione": 0.5, "gen_tok_s": 9.0}]))

    def test_lo_sweep_rifiuta_di_girare_senza_store(self):
        """Senza store gli expert stanno in memoria mappata e il budget non e' un
        tetto: la curva misurerebbe niente, e il numero sembrerebbe valido."""
        args = argparse.Namespace(
            sweep_expert_budget="8G,4G", expert_store=None, repeat=1,
            cooldown=0, prompt=None, prompt_file=None, prompt_tokens=None)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rc = cli._cmd_benchmark_sweep_budget(args)
        self.assertNotEqual(rc, 0)
        self.assertIn("--expert-store", buf.getvalue())

    def test_pretouch_lasciato_in_ambiente_non_falsa_la_misura(self):
        """Capitato davvero: PICOSTAR_EXPERT_PRETOUCH era rimasta in ambiente da
        un esperimento di giorni prima, e la prima curva velocita'/residenza e'
        stata misurata con ogni slice azzerata (memset) prima di essere letta.

        Non e' un rallentamento costante che si potrebbe sottrarre: il memset si
        paga **per ogni miss**, cioe' esattamente sulla variabile che lo sweep fa
        variare, quindi la curva risultava piu' ripida del vero. Da qui in poi la
        variabile si impone a 0, come --cache-ram: cio' che non si dichiara lo
        decide qualcun altro."""
        m = read_gguf(MOE)
        store = self.tmp / "finto.psxs"
        store.touch()
        base = dict(expert_store=str(store), expert_direct=True, verbose=False,
                    expert_stats=False, expert_budget="4G")
        salvati = {n: os.environ.pop(n, None)
                   for n in ("ARGINE_EXPERT_PRETOUCH", "PICOSTAR_EXPERT_PRETOUCH")}
        try:
            for nome in ("ARGINE_EXPERT_PRETOUCH", "PICOSTAR_EXPERT_PRETOUCH"):
                os.environ[nome] = "1"
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    env = cli._expert_store_env(
                        argparse.Namespace(**base, pretouch=False), m)
                self.assertEqual(env.get("ARGINE_EXPERT_PRETOUCH"), "0",
                                 f"{nome} residua deve essere neutralizzata")
                self.assertIn(nome, buf.getvalue(),
                              "e va detto: averla ignorata in silenzio sarebbe "
                              "un altro modo di non far sapere com'e' stata "
                              "presa la misura")
                del os.environ[nome]

            # senza niente in ambiente: comunque esplicito, non assente
            env = cli._expert_store_env(
                argparse.Namespace(**base, pretouch=False), m)
            self.assertEqual(env.get("ARGINE_EXPERT_PRETOUCH"), "0")

            # chi la vuole per diagnosi la chiede, e la ottiene
            env = cli._expert_store_env(
                argparse.Namespace(**base, pretouch=True), m)
            self.assertEqual(env.get("ARGINE_EXPERT_PRETOUCH"), "1")
        finally:
            for n, v in salvati.items():
                os.environ.pop(n, None)
                if v is not None:
                    os.environ[n] = v

    def test_il_piano_non_dice_mmap_quando_il_tetto_e_nostro(self):
        """La riga sotto "expert residenti" nasce prima dell'expert store, quando
        la residenza era davvero della page cache dell'OS. Con lo store attivo
        diceva l'esatto contrario del vero, proprio nel punto in cui si legge se
        il tetto esiste: "expert residenti 8.00 GiB" seguito da "residenza
        governata da mmap"."""
        m = read_gguf(MOE)
        hw = hwmod.detect()
        for store, atteso, vietato in ((True, "tetto rigido", "governata da mmap"),
                                       (False, "governata da mmap", "tetto rigido")):
            p = planmod.plan(m, hw, planmod.PlanRequest(
                backend="cpu", expert_store=store,
                expert_store_budget=4 * GB if store else None,
                ram_limit=14 * GB))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                cli._print_plan(p, m)
            out = buf.getvalue()
            if not p.expert_cache_budget:
                continue  # niente budget, niente riga: nulla da verificare
            self.assertIn(atteso, out, f"expert_store={store}")
            self.assertNotIn(vietato, out, f"expert_store={store}")

    def test_un_modello_inesistente_ferma_lo_sweep_subito(self):
        """Capitato: percorso sbagliato, e lo sweep ha tentato la prima
        configurazione, fallito, dormito 20 secondi di raffreddamento, e si
        preparava a ripetere lo stesso fallimento altre tre volte — spingendo
        fuori schermo il messaggio utile, cioe' l'elenco dei .gguf che ci sono
        davvero.

        Un file che non esiste non esistera' nemmeno al giro dopo. Una
        configurazione che non ci sta in RAM, invece, va saltata e si prosegue:
        sono due guasti diversi e vanno trattati in modo diverso."""
        args = argparse.Namespace(
            model=str(self.tmp / "non-c-e.gguf"), expert_store="auto",
            sweep_expert_budget="8G,6G,4G,3G", repeat=1, cooldown=20,
            prompt=None, prompt_file=None, prompt_tokens=None)
        buf = io.StringIO()
        t0 = time.monotonic()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rc = cli._cmd_benchmark_sweep_budget(args)
        trascorso = time.monotonic() - t0
        self.assertEqual(rc, cli.EXIT_USAGE)
        self.assertLess(trascorso, 5.0,
                        "non deve aspettare nemmeno una pausa di raffreddamento")
        self.assertIn("non trovato", buf.getvalue())
        self.assertNotIn("raffreddamento", buf.getvalue())

    def test_lo_sweep_rifiuta_una_dimensione_non_valida(self):
        args = argparse.Namespace(
            sweep_expert_budget="8G,pippo", expert_store="auto", repeat=1,
            cooldown=0, prompt=None, prompt_file=None, prompt_tokens=None)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            rc = cli._cmd_benchmark_sweep_budget(args)
        self.assertNotEqual(rc, 0)


class TestApiAnthropic(unittest.TestCase):
    """Il dialetto Anthropic davanti a un backend che parla OpenAI.

    La traduzione e' fatta di funzioni pure per poter essere verificata senza
    avviare niente, ed e' una scelta presa dopo un bug nello store: la', il
    collaudo girava su una piattaforma diversa da quella di destinazione e un
    puntatore pendente ci e' passato in mezzo. Qui la piattaforma non esiste:
    esiste il JSON, e i test lo coprono tutto."""

    def test_system_passa_da_campo_a_messaggio(self):
        """La differenza strutturale fra i due formati: in Anthropic `system` e' un
        campo in cima, in OpenAI un messaggio nella lista."""
        oai = aa.richiesta_in_openai({
            "max_tokens": 10, "system": "Sei conciso.",
            "messages": [{"role": "user", "content": "ciao"}]})
        self.assertEqual(oai["messages"][0], {"role": "system", "content": "Sei conciso."})
        self.assertEqual(oai["messages"][1]["content"], "ciao")

    def test_i_blocchi_di_testo_si_appiattiscono(self):
        oai = aa.richiesta_in_openai({
            "max_tokens": 10,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "prima"}, {"type": "text", "text": "seconda"}]}]})
        self.assertEqual(oai["messages"][0]["content"], "prima\nseconda")

    def test_max_tokens_mancante_e_un_errore_non_un_default(self):
        """Inventare un default qui darebbe al client una risposta troncata senza
        che sappia perche'. In Anthropic il campo e' obbligatorio: lo si dice."""
        with self.assertRaises(aa.ApiError) as cm:
            aa.richiesta_in_openai({"messages": [{"role": "user", "content": "x"}]})
        self.assertIn("max_tokens", str(cm.exception))
        self.assertEqual(cm.exception.to_dict()["error"]["type"], "invalid_request_error")

    def test_un_immagine_viene_rifiutata_invece_che_ignorata(self):
        """Questo runtime serve modelli testuali su CPU. Accettare un blocco
        immagine e scartarlo in silenzio produrrebbe una risposta plausibile a
        meta' della domanda, che e' peggio di un errore."""
        with self.assertRaises(aa.ApiError) as cm:
            aa.richiesta_in_openai({"max_tokens": 10, "messages": [
                {"role": "user", "content": [{"type": "image", "source": {}}]}]})
        self.assertIn("image", str(cm.exception))

    def test_ruolo_system_dentro_messages_e_un_errore(self):
        """In Anthropic i ruoli ammessi sono solo user e assistant: accettare un
        system qui lo farebbe finire due volte nel prompt."""
        with self.assertRaises(aa.ApiError):
            aa.richiesta_in_openai({"max_tokens": 10, "messages": [
                {"role": "system", "content": "x"}]})

    def test_stop_sequence_si_ricostruisce_dal_testo(self):
        """OpenAI dice `stop` sia quando il modello si ferma da solo sia quando
        incontra una sequenza richiesta; Anthropic distingue i due casi e nel
        secondo dice quale sequenza. La differenza si ricava dal testo."""
        base = {"choices": [{"message": {"content": "bla FINE"}, "finish_reason": "stop"}]}
        a = aa.risposta_in_anthropic(base, "m", ["FINE"])
        self.assertEqual(a["stop_reason"], "stop_sequence")
        self.assertEqual(a["stop_sequence"], "FINE")
        b = aa.risposta_in_anthropic(base, "m", ["ALTRO"])
        self.assertEqual(b["stop_reason"], "end_turn")
        self.assertIsNone(b["stop_sequence"])

    def test_i_motivi_di_fine_e_i_conteggi_cambiano_nome(self):
        a = aa.risposta_in_anthropic({
            "choices": [{"message": {"content": "x"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3}}, "m")
        self.assertEqual(a["stop_reason"], "max_tokens")
        self.assertEqual(a["usage"], {"input_tokens": 7, "output_tokens": 3})

    def test_gli_argomenti_di_uno_strumento_diventano_oggetto(self):
        """OpenAI consegna gli argomenti come stringa JSON, Anthropic come
        oggetto: lasciarli stringa darebbe al client un `input` che non sa
        leggere."""
        a = aa.risposta_in_anthropic({"choices": [{"message": {"content": None,
            "tool_calls": [{"id": "t1", "function": {"name": "meteo",
                            "arguments": '{"citta": "Roma"}'}}]},
            "finish_reason": "tool_calls"}]}, "m")
        blocco = a["content"][0]
        self.assertEqual(blocco["type"], "tool_use")
        self.assertEqual(blocco["input"], {"citta": "Roma"})
        self.assertEqual(a["stop_reason"], "tool_use")

    def test_argomenti_illeggibili_non_diventano_un_oggetto_vuoto(self):
        """Un oggetto vuoto farebbe eseguire la chiamata con parametri sbagliati.
        Il testo grezzo va consegnato sotto una chiave riconoscibile."""
        a = aa.risposta_in_anthropic({"choices": [{"message": {"content": None,
            "tool_calls": [{"id": "t1", "function": {"name": "x",
                            "arguments": "{non json"}}]},
            "finish_reason": "tool_calls"}]}, "m")
        self.assertIn("_non_interpretabile", a["content"][0]["input"])

    def _eventi(self, righe, **kw):
        return [b.decode().split("\n")[0].removeprefix("event: ")
                for b in aa.eventi_da_chunk_openai(righe, "m", **kw)]

    def test_lo_streaming_emette_la_sequenza_completa(self):
        d = lambda x: b"data: " + json.dumps(x).encode()
        ev = self._eventi([
            d({"choices": [{"delta": {"content": "a"}}]}),
            d({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
            b"data: [DONE]"])
        self.assertEqual(ev, ["message_start", "content_block_start",
                              "content_block_delta", "content_block_stop",
                              "message_delta", "message_stop"])

    def test_un_flusso_interrotto_viene_comunque_chiuso(self):
        """Un client Anthropic che non riceve `message_stop` aspetta per sempre, e
        un flusso troncato a meta' e' proprio il caso in cui succede."""
        d = lambda x: b"data: " + json.dumps(x).encode()
        ev = self._eventi([d({"choices": [{"delta": {"content": "a"}}]})])
        self.assertEqual(ev[-1], "message_stop")
        self.assertIn("content_block_stop", ev)

    def test_un_flusso_vuoto_apre_e_chiude(self):
        ev = self._eventi([])
        self.assertEqual(ev, ["message_start", "message_delta", "message_stop"])
        self.assertNotIn("content_block_start", ev,
                         "senza testo non si apre un blocco di contenuto")

    def test_le_righe_sse_si_tagliano_sui_newline_non_sulle_letture(self):
        """Un chunk SSE puo' arrivare spezzato in mezzo a una riga: trattare ogni
        lettura come una riga intera perderebbe i pezzi."""
        pezzi = [b"data: {\"a\": 1}\ndata: {\"b\"", b": 2}\n", b""]
        it = iter(pezzi)
        righe = list(aa.righe_sse(lambda n: next(it, b"")))
        self.assertEqual(righe, [b'data: {"a": 1}', b'data: {"b": 2}'])


class TestRinomina(unittest.TestCase):
    """Il progetto si chiamava PicoStar16. Una rinomina rompe in silenzio: una
    variabile d'ambiente ignorata, uno storico perso, un formato che non si
    riconosce piu'. Questi test fissano le tre compatibilita' che ho deciso di
    tenere, perche' fra sei mesi nessuno si ricordera' perche' c'erano."""

    def setUp(self):
        self.tmp = Path(_TMP.name) / "rinomina"  # type: ignore[union-attr]
        self.tmp.mkdir(exist_ok=True)

    def test_il_vecchio_nome_di_una_variabile_viene_accettato(self):
        """Ignorarlo in silenzio sarebbe il comportamento peggiore: Argine
        ricadrebbe sulla build ufficiale senza dirlo, e due misure di velocita'
        prese su binari diversi sembrerebbero una differenza fra due modelli.
        E' successo, e il fattore 1,74 risultava dal cambio di binario/build; EXP-011 non isola quale differenza di build lo causasse."""
        vecchio = os.environ.pop("PICOSTAR_LLAMA_DIR", None)
        nuovo = os.environ.pop("ARGINE_LLAMA_DIR", None)
        try:
            os.environ["PICOSTAR_LLAMA_DIR"] = r"C:\vecchio\percorso"
            self.assertEqual(rtmod._variabile("ARGINE_LLAMA_DIR"),
                             r"C:\vecchio\percorso")
            os.environ["ARGINE_LLAMA_DIR"] = r"C:\nuovo"
            self.assertEqual(rtmod._variabile("ARGINE_LLAMA_DIR"), r"C:\nuovo",
                             "il nome nuovo deve vincere su quello vecchio")
        finally:
            os.environ.pop("PICOSTAR_LLAMA_DIR", None)
            os.environ.pop("ARGINE_LLAMA_DIR", None)
            if vecchio:
                os.environ["PICOSTAR_LLAMA_DIR"] = vecchio
            if nuovo:
                os.environ["ARGINE_LLAMA_DIR"] = nuovo

    def test_una_variabile_assente_resta_assente(self):
        self.assertIsNone(rtmod._variabile("ARGINE_NON_ESISTE_PROPRIO"))

    def test_il_formato_dello_store_non_e_cambiato(self):
        """PSXS nomina il formato, non il prodotto. Cambiarlo avrebbe obbligato
        a ricostruire 16,35 GiB di store per una modifica cosmetica, e avrebbe
        reso illeggibili i file gia' prodotti."""
        self.assertEqual(es.MAGIC, 0x53585350)
        testata = (Path(__file__).resolve().parents[1]
                   / "patches" / "argine-store.h").read_text(encoding="utf-8")
        self.assertIn("0x53585350u", testata,
                      "il lato C++ e quello Python devono leggere lo stesso formato")

    def test_il_codice_non_nomina_piu_il_vecchio_progetto(self):
        """Tranne dove serve: le compatibilita' col nome vecchio e i commenti
        che raccontano da dove viene un bug."""
        radice = Path(__file__).resolve().parents[1]
        colpevoli = []
        for p in sorted(radice.rglob("*.py")):
            if "__pycache__" in str(p):
                continue
            for n, riga in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                if "picostar" in riga.lower() and "PICOSTAR_" not in riga \
                        and "PicoStar16" not in riga and "picostar16" not in riga:
                    colpevoli.append(f"{p.name}:{n}")
        self.assertEqual(colpevoli, [], "nomi vecchi rimasti nel codice")

    def test_una_cartella_locale_inesistente_non_e_un_guasto_di_rete(self):
        """Capitato alla prima installazione dopo la rinomina: la variabile era
        rimasta a puntare la cartella col nome vecchio, `setup` ha detto
        "download non riuscito, oppure indica una copia locale con
        ARGINE_LLAMA_DIR" — cioe' ha consigliato la cosa che era impostata e
        rotta, e ha mandato a ritentare un download che non viene mai tentato
        finche' la variabile c'e'. Il messaggio era l'unica cosa sbagliata: il
        rifiuto in se' era corretto."""
        vecchio = os.environ.pop("PICOSTAR_LLAMA_DIR", None)
        nuovo = os.environ.pop("ARGINE_LLAMA_DIR", None)
        try:
            self.assertEqual(rtmod.llama_dir_configurato(), (None, None, False),
                             "senza variabili, setup e' libero di scaricare")

            os.environ["ARGINE_LLAMA_DIR"] = str(Path(self.tmp) / "non-esiste")
            valore, var, valido = rtmod.llama_dir_configurato()
            self.assertEqual(var, "ARGINE_LLAMA_DIR")
            self.assertFalse(valido, "la cartella non c'e': non e' una build")
            del os.environ["ARGINE_LLAMA_DIR"]

            # il nome vecchio va riconosciuto *come tale*, per poterlo dire
            os.environ["PICOSTAR_LLAMA_DIR"] = str(Path(self.tmp) / "non-esiste")
            valore, var, valido = rtmod.llama_dir_configurato()
            self.assertEqual(var, "PICOSTAR_LLAMA_DIR")
            self.assertFalse(valido)
            del os.environ["PICOSTAR_LLAMA_DIR"]

            # una cartella che contiene davvero llama-server e' valida
            buona = Path(self.tmp) / "build"
            buona.mkdir(parents=True, exist_ok=True)
            for n in rtmod.BIN_NAMES["server"]:
                (buona / n).write_bytes(b"x")
                break
            os.environ["ARGINE_LLAMA_DIR"] = str(buona)
            self.assertTrue(rtmod.llama_dir_configurato()[2])
        finally:
            for n in ("PICOSTAR_LLAMA_DIR", "ARGINE_LLAMA_DIR"):
                os.environ.pop(n, None)
            if vecchio:
                os.environ["PICOSTAR_LLAMA_DIR"] = vecchio
            if nuovo:
                os.environ["ARGINE_LLAMA_DIR"] = nuovo

    def test_il_consiglio_di_setup_nomina_la_variabile_giusta(self):
        """Il messaggio deve proporre il nome *nuovo* da impostare anche quando
        trova quello vecchio, altrimenti insegna all'utente a tenersi il nome
        che sta per sparire.

        Prima versione di questo test: cercava stringhe dentro il sorgente di
        `cmd_setup`. Si e' rotto cambiando una virgoletta, senza che il
        comportamento cambiasse di un carattere. Ora si guarda cio' che l'utente
        legge davvero."""
        vecchio = os.environ.pop("PICOSTAR_LLAMA_DIR", None)
        nuovo = os.environ.pop("ARGINE_LLAMA_DIR", None)
        buf = io.StringIO()
        try:
            os.environ["PICOSTAR_LLAMA_DIR"] = r"C:\posto\sparito\llama-noomp"
            args = argparse.Namespace(backend="both", llama_build=None)
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                rc = cli.cmd_setup(args)
        finally:
            for n in ("PICOSTAR_LLAMA_DIR", "ARGINE_LLAMA_DIR"):
                os.environ.pop(n, None)
            if vecchio:
                os.environ["PICOSTAR_LLAMA_DIR"] = vecchio
            if nuovo:
                os.environ["ARGINE_LLAMA_DIR"] = nuovo

        out = buf.getvalue()
        self.assertNotEqual(rc, 0, "setup non puo' dirsi riuscito")
        self.assertIn("setx ARGINE_LLAMA_DIR", out,
                      "va proposto il nome nuovo, non quello che sta sparendo")
        self.assertIn("terminale nuovo", out,
                      "setx non cambia il terminale corrente: va detto")
        self.assertNotIn("download", out.lower(),
                         "non e' un guasto di rete, e chiamarlo cosi' manda a "
                         "ritentare un download che non viene mai tentato")
        self.assertNotIn("indicare una copia locale", out,
                         "era il consiglio che ripeteva la cosa gia' rotta")

    def test_doctor_guarda_il_binario_che_partira_non_quelli_archiviati(self):
        """`doctor` leggeva solo la cartella dati, quindi diceva "runtime non
        installato" mentre ARGINE_LLAMA_DIR puntava a una build buona, e taceva
        del tutto quando puntava nel vuoto. Il README prometteva che doctor sa
        dire se il binario contiene l'expert store: era falso.

        E' la stessa distinzione che una volta e' costata due misure: conta il
        binario che parte, non quelli che abbiamo archiviato."""
        vecchio = os.environ.pop("PICOSTAR_LLAMA_DIR", None)
        nuovo = os.environ.pop("ARGINE_LLAMA_DIR", None)

        def diagnosi(**env) -> str:
            for n in ("PICOSTAR_LLAMA_DIR", "ARGINE_LLAMA_DIR"):
                os.environ.pop(n, None)
            os.environ.update(env)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                cli.cmd_doctor(argparse.Namespace())
            return buf.getvalue()

        try:
            # una build che contiene il marcatore, e una che no
            patchata = self.tmp / "patchata"
            patchata.mkdir(exist_ok=True)
            (patchata / "llama-server.exe").write_bytes(
                b"\x00ciao ARGINE_EXPERT_STORE mondo\x00")
            ufficiale = self.tmp / "ufficiale"
            ufficiale.mkdir(exist_ok=True)
            (ufficiale / "llama-server.exe").write_bytes(b"\x00nessun marcatore\x00")

            out = diagnosi(ARGINE_LLAMA_DIR=str(patchata))
            self.assertIn("ARGINE_LLAMA_DIR", out)
            self.assertIn("presente", out, "lo store c'e' e doctor deve dirlo")
            self.assertNotIn("non installato", out,
                             "una build indicata a mano E' un runtime")

            out = diagnosi(ARGINE_LLAMA_DIR=str(ufficiale))
            self.assertIn("assente", out,
                          "senza marcatore, --expert-store verrebbe ignorato in "
                          "silenzio: e' il punto di tutto il controllo")

            # percorso morto: va segnalato, non ignorato
            morto = str(self.tmp / "sparita")
            out = diagnosi(ARGINE_LLAMA_DIR=morto)
            self.assertIn("senza llama-server", out)

            # e va segnalato anche quando arriva dal nome vecchio
            out = diagnosi(PICOSTAR_LLAMA_DIR=morto)
            self.assertIn("PICOSTAR_LLAMA_DIR", out)
            self.assertIn("nome vecchio", out)
        finally:
            for n in ("PICOSTAR_LLAMA_DIR", "ARGINE_LLAMA_DIR"):
                os.environ.pop(n, None)
            if vecchio:
                os.environ["PICOSTAR_LLAMA_DIR"] = vecchio
            if nuovo:
                os.environ["ARGINE_LLAMA_DIR"] = nuovo


class TestGateUpFusi(unittest.TestCase):
    """llama.cpp accetta un GGUF con gate e up FUSI in `ffn_gate_up_exps`, e li
    preferisce quando ci sono. Per lo store e' il modo peggiore di fallire: quel
    nome non e' nessuno dei tre che cerchiamo, quindi lo store si costruirebbe
    con solo `down` — un terzo dei byte — e il tetto di RAM sarebbe falso del
    doppio, senza che una riga lo dica.

    Scoperto guardando llama.cpp *prima* di scaricare 35 GB di un modello che
    potrebbe averli fusi, non dopo."""

    def _modello_finto(self, nomi):
        t = [types.SimpleNamespace(name=n, dims=[8, 8, 4], n_bytes=4096,
                                   layer_index=0, abs_offset=0)
             for n in nomi]
        return types.SimpleNamespace(is_moe=True, expert_count=4, tensors=t)

    def test_un_gguf_fuso_viene_rifiutato_con_una_riga_che_spiega(self):
        m = self._modello_finto(["blk.0.ffn_gate_up_exps.weight",
                                 "blk.0.ffn_down_exps.weight"])
        problemi = es.check_model(m)  # type: ignore[arg-type]
        self.assertEqual(len(problemi), 1, "una riga, non una per layer")
        self.assertIn("FUSI", problemi[0])
        self.assertIn("ffn_gate_up_exps", problemi[0])

    def test_il_nome_fuso_non_viene_preso_per_up(self):
        """`blk.0.ffn_gate_up_exps.weight` non deve corrispondere alla regex di
        `up`: se lo facesse, lo store scriverebbe byte sbagliati invece di
        rifiutare, che e' peggio."""
        for kind, rx in es._KIND_RE.items():
            with self.subTest(kind=kind):
                self.assertIsNone(rx.search("blk.0.ffn_gate_up_exps.weight"))

    def test_un_gguf_normale_passa(self):
        m = self._modello_finto(["blk.0.ffn_gate_exps.weight",
                                 "blk.0.ffn_up_exps.weight",
                                 "blk.0.ffn_down_exps.weight"])
        self.assertEqual(es.check_model(m), [])  # type: ignore[arg-type]


class TestConsigliNonAvviabile(unittest.TestCase):
    """Quando il piano non sta nel budget, i consigli erano quattro e fissi. Nel
    caso reale di Qwen3-Next-80B con lo store e `--expert-budget 4G` su una
    macchina con 2,47 GiB liberi, **tre su quattro non si applicavano**:

    - "un GGUF piu' piccolo (attuale 35.67 GiB)": con lo store i residenti sono
      1,35 GiB, il file non e' il vincolo;
    - "--ctx 1024 --kv-type q4_0": le riduzioni automatiche avevano gia' portato
      ctx a 512 e la KV a Q4_0;
    - "lascia attivo l'expert streaming": era attivo.

    L'unica cosa da fare stava in un avviso trenta righe sopra. Stesso difetto del
    messaggio di `setup` che ripeteva l'impostazione rotta."""

    def _caso_reale(self):
        est = planmod.MemoryEstimate(
            weights_resident=int(1.35 * 2**30), repack=int(437.62 * 2**20),
            expert_cache=4 * 2**30, kv_cache=int(13.5 * 2**20),
            activations=int(24.3 * 2**20), runtime=384 * 2**20)
        p = planmod.RuntimePlan(
            ctx=512, kv_type_k="q4_0", kv_type_v="q4_0", expert_store=True,
            moe_streaming=True, expert_cache_budget=4 * 2**30,
            # quello che il planner lasciava da se', prima dell'override
            expert_cache_sostenibile=512 * 2**20,
            budget=int(1.97 * 2**30), estimate=est, feasible=False)
        hw = types.SimpleNamespace(ram_available=int(2.47 * 2**30),
                                   ram_total=int(15.8 * 2**30))
        m = types.SimpleNamespace(is_moe=True, total_size_on_disk=int(35.67 * 2**30),
                                  expert_bytes=int(34.31 * 2**30))
        args = argparse.Namespace(expert_budget="4G")
        return m, hw, p, args

    def test_i_tre_consigli_inapplicabili_non_compaiono(self):
        m, hw, p, args = self._caso_reale()
        righe = " | ".join(cli._consigli_non_avviabile(m, hw, p, args))
        self.assertNotIn("GGUF piu' piccolo", righe,
                         "con lo store i residenti sono 1,35 GiB su 35,67")
        self.assertNotIn("--ctx 1024", righe, "ctx era gia' a 512")
        self.assertNotIn("non usare --no-streaming", righe, "era attivo")
        self.assertNotIn("--kv-type", righe, "la KV era gia' Q4_0")

    def test_dice_la_causa_vera_per_prima(self):
        m, hw, p, args = self._caso_reale()
        righe = cli._consigli_non_avviabile(m, hw, p, args)
        self.assertIn("--expert-budget", righe[0])
        # e dice quanto lascia davvero agli expert, invece di un invito generico
        self.assertRegex(righe[0], r"\d")

    def test_il_numero_e_quello_del_planner_non_uno_ricalcolato(self):
        """La prima versione se lo ricalcolava e stampava "0 B" dove l'avviso del
        planner, trenta righe sopra, diceva 512 MiB: due numeri in contrasto nella
        stessa schermata, che e' esattamente il difetto che questa funzione
        doveva togliere."""
        m, hw, p, args = self._caso_reale()
        righe = cli._consigli_non_avviabile(m, hw, p, args)
        self.assertIn("512.00 MiB", righe[0])
        self.assertNotIn("0 B", righe[0])

    def test_senza_il_numero_del_planner_non_se_ne_inventa_uno(self):
        m, hw, p, args = self._caso_reale()
        p.expert_cache_sostenibile = 0
        righe = cli._consigli_non_avviabile(m, hw, p, args)
        self.assertIn("--expert-budget", righe[0])
        self.assertNotIn("lascia agli expert", righe[0])

    def test_dice_quanta_ram_liberare_e_perche_tenere_il_budget(self):
        m, hw, p, args = self._caso_reale()
        righe = " | ".join(cli._consigli_non_avviabile(m, hw, p, args))
        self.assertIn("libera", righe)
        self.assertIn("stesso budget", righe,
                      "abbassare il budget rende la misura non confrontabile: "
                      "va detto, altrimenti si sceglie la strada comoda")

    def test_i_consigli_tornano_quando_sono_applicabili(self):
        """Il contrario: su un modello denso senza store, con ctx alto e KV F16,
        i consigli di prima sono quelli giusti e devono esserci."""
        est = planmod.MemoryEstimate(weights_resident=14 * 2**30,
                                     kv_cache=2 * 2**30, runtime=384 * 2**20)
        p = planmod.RuntimePlan(ctx=8192, kv_type_k="f16", kv_type_v="f16",
                                expert_store=False, moe_streaming=False,
                                budget=8 * 2**30, estimate=est, feasible=False)
        hw = types.SimpleNamespace(ram_available=int(7.5 * 2**30),
                                   ram_total=int(15.8 * 2**30))
        m = types.SimpleNamespace(is_moe=True, total_size_on_disk=14 * 2**30,
                                  expert_bytes=12 * 2**30)
        args = argparse.Namespace(expert_budget=None)
        righe = " | ".join(cli._consigli_non_avviabile(m, hw, p, args))
        self.assertIn("--ctx 4096", righe)
        self.assertIn("--kv-type q4_0", righe)
        self.assertIn("non usare --no-streaming", righe)
        self.assertIn("GGUF piu' piccolo", righe)

    def test_non_resta_mai_senza_niente_da_dire(self):
        p = planmod.RuntimePlan(ctx=512, kv_type_k="q4_0", kv_type_v="q4_0",
                                expert_store=True, moe_streaming=True,
                                budget=2 * 2**30, feasible=False,
                                estimate=planmod.MemoryEstimate(runtime=3 * 2**30))
        hw = types.SimpleNamespace(ram_available=14 * 2**30, ram_total=16 * 2**30)
        m = types.SimpleNamespace(is_moe=True, total_size_on_disk=2 * 2**30,
                                  expert_bytes=1 * 2**30)
        righe = cli._consigli_non_avviabile(m, hw, p,
                                            argparse.Namespace(expert_budget=None))
        self.assertTrue(righe, "un elenco vuoto lascia l'utente senza appigli")


class TestCoppieDiVerifica(unittest.TestCase):
    """La verifica dello store confrontava 26 gruppi su 24576, e sceglieva i layer
    a caso: su 48 layer con 24 campioni ne lasciava intatti 25. Un layer scritto
    nel posto sbagliato passava la verifica piu' di una volta su due.

    Il difetto era vecchio ma invisibile: il conteggio dei campioni era fisso,
    quindi la copertura scendeva al crescere del modello, e su 6144 gruppi
    sembrava ragionevole. L'ha reso evidente un modello quattro volte piu' grande.

    La forma dell'errore dice come campionare: l'indice e' per layer, quindi un
    errore nella tabella dei layer rovina un layer intero. Quello si prende con
    certezza solo visitando tutti i layer."""

    def test_ogni_layer_e_coperto(self):
        for n_layer, n_exp in ((48, 512), (48, 128), (2, 8), (1, 1), (80, 256)):
            with self.subTest(layer=n_layer, exp=n_exp):
                c = es.coppie_di_verifica(list(range(n_layer)), n_exp, 24)
                self.assertEqual(len({l for l, _ in c}), n_layer,
                                 "un layer non visitato e' un layer in cui un "
                                 "errore passa inosservato")

    def test_la_vecchia_strategia_lasciava_scoperti_meta_dei_layer(self):
        """Il confronto quantitativo, per non perdere la ragione della modifica."""
        rnd = random.Random(0)
        vecchie = [(rnd.choice(list(range(48))), rnd.randrange(512))
                   for _ in range(24)]
        vecchie += [(0, 0), (47, 511)]
        self.assertLess(len({l for l, _ in vecchie}), 30)
        nuove = es.coppie_di_verifica(list(range(48)), 512, 24)
        self.assertEqual(len({l for l, _ in nuove}), 48)

    def test_gli_estremi_del_file_sono_sempre_dentro(self):
        c = es.coppie_di_verifica(list(range(48)), 512, 24)
        self.assertIn((0, 0), c)
        self.assertIn((47, 511), c)

    def test_e_riproducibile_e_senza_duplicati(self):
        a = es.coppie_di_verifica(list(range(48)), 512, 24)
        b = es.coppie_di_verifica(list(range(48)), 512, 24)
        self.assertEqual(a, b, "una verifica che cambia a ogni giro non si "
                              "puo' citare in un bug report")
        self.assertEqual(len(a), len(set(a)), "leggere due volte lo stesso "
                                             "gruppo e' tempo speso per nulla")

    def test_samples_zero_verifica_comunque_tutti_i_layer(self):
        c = es.coppie_di_verifica(list(range(48)), 512, 0)
        self.assertEqual(len({l for l, _ in c}), 48)

    def test_il_messaggio_non_ricalcola_il_conteggio(self):
        """Il messaggio diceva `samples + 2` per conto suo: un numero destinato a
        diventare falso al primo cambio nella scelta dei campioni, e lo e'
        diventato."""
        m = types.SimpleNamespace(
            expert_count=512,
            tensors=[types.SimpleNamespace(
                name=f"blk.{l}.ffn_{k}_exps.weight", layer_index=l,
                dims=[8, 8, 512], n_bytes=4096, abs_offset=0)
                for l in range(48) for k in ("gate", "up", "down")])
        quanti, quanti_layer = cli._conta_verifica(m, 24)  # type: ignore[arg-type]
        atteso = es.coppie_di_verifica(list(range(48)), 512, 24)
        self.assertEqual(quanti, len(atteso))
        self.assertEqual(quanti_layer, 48)
        self.assertNotEqual(quanti, 24 + 2, "il vecchio conteggio era 26")


class TestSlicePerTipo(unittest.TestCase):
    """La dimensione di una slice non e' una proprieta' del modello: dipende dalla
    quantizzazione, e i quantizzatori la variano per layer. Su
    Qwen3-Next-80B-A3B Q3_K_M `down` e' Q4_K su 45 layer e Q5_K su 3.

    Il piano stampava le tre slice del primo gruppo come se descrivessero tutto,
    e i numeri a schermo non si moltiplicavano piu' fra loro: 1,55 MiB x 512
    expert x 48 layer = 38,1 GiB contro i 34,31 dichiarati due righe sopra."""

    def _piano_misto(self):
        """48 layer: `down` da 704 KiB sul layer 0 e da 576 sugli altri 47."""
        gruppi = []
        for layer in range(48):
            giu = 704 * 1024 if layer == 0 else 576 * 1024
            slabs = tuple(
                types.SimpleNamespace(n_bytes=n)
                for n in (440 * 1024, 440 * 1024, giu))
            for e in range(4):
                gruppi.append(types.SimpleNamespace(layer=layer, expert=e,
                                                    slabs=slabs))
        raw = sum(sum(s.n_bytes for s in g.slabs) for g in gruppi)
        return types.SimpleNamespace(groups=gruppi, raw_bytes=raw)

    def test_le_dimensioni_variabili_vengono_dette_tutte(self):
        voci = dict(cli._slice_per_tipo(self._piano_misto()))
        self.assertEqual(voci["gate"], [(440 * 1024, 48)])
        self.assertEqual(voci["up"], [(440 * 1024, 48)])
        # la dimensione piu' diffusa va per prima, non quella del layer 0
        self.assertEqual(voci["down"], [(576 * 1024, 47), (704 * 1024, 1)])

    def test_la_media_moltiplicata_torna_al_totale(self):
        """La proprieta' che mancava: il numero stampato, moltiplicato per i
        gruppi, deve dare la dimensione dichiarata."""
        piano = self._piano_misto()
        medio = piano.raw_bytes / len(piano.groups)
        self.assertAlmostEqual(medio * len(piano.groups), piano.raw_bytes, places=3)

    def test_l_output_dice_su_quanti_layer(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli._stampa_slice_per_tipo(self._piano_misto())
        out = buf.getvalue()
        self.assertIn("in media", out)
        self.assertIn("su 47 layer", out)
        self.assertIn("su 1 layer", out)
        self.assertIn("lettura down", out)

    def test_quantizzazione_uniforme_resta_una_riga_semplice(self):
        piano = self._piano_misto()
        for g in piano.groups:  # tutti uguali: nessuna variazione da segnalare
            g.slabs = tuple(types.SimpleNamespace(n_bytes=440 * 1024)
                            for _ in range(3))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli._stampa_slice_per_tipo(piano)
        self.assertNotIn("su 48 layer", buf.getvalue())


class TestControllaReadme(unittest.TestCase):
    """Lo strumento che verifica il README e ne rigenera l'indice.

    Il README e' il risultato principale del progetto, e ha due modi di rompersi
    in silenzio: un link interno che non punta piu' a niente (rinominare un titolo
    non produce nessun errore) e un indice scritto a mano, che invecchia alla prima
    sezione aggiunta.

    La prima versione di questi controlli produsse cinque segnalazioni, **tutte
    false**: `meTODO` dentro "metodo" due volte, "placeholder" come termine tecnico
    di `VirtualAlloc2`, e due righe di tabella con `\\|` escapato contate come
    colonne in piu'. Uno strumento che grida al lupo e' uno che si impara a
    ignorare, quindi i tre casi stanno qui dentro."""

    @classmethod
    def setUpClass(cls):
        import importlib.util
        percorso = (Path(__file__).resolve().parents[1] / "strumenti"
                    / "controlla_readme.py")
        spec = importlib.util.spec_from_file_location("controlla_readme", percorso)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)
        cls.tmp = Path(_TMP.name) / "readme"  # type: ignore[union-attr]
        cls.tmp.mkdir(exist_ok=True)

    def _scrivi(self, testo):
        p = self.tmp / f"caso{abs(hash(testo)) % 10**8}.md"
        p.write_text(testo, encoding="utf-8")
        return p

    def test_trova_un_link_rotto_e_suggerisce(self):
        p = self._scrivi("# T\n\n## Alfa\n\nVedi [x](#alfaa).\n")
        problemi, _ = self.mod.controlla(p)
        self.assertEqual(len(problemi), 1)
        self.assertIn("#alfaa", problemi[0])
        self.assertIn("forse: #alfa", problemi[0])

    def test_trova_un_titolo_duplicato(self):
        p = self._scrivi("# T\n\n## Beta\n\n## Beta\n")
        problemi, _ = self.mod.controlla(p)
        self.assertTrue(any("duplicato" in x for x in problemi))

    def test_trova_un_blocco_di_codice_non_chiuso(self):
        p = self._scrivi("# T\n\n## A\n\n```\nqualcosa\n")
        problemi, _ = self.mod.controlla(p)
        self.assertTrue(any("non e' chiuso" in x for x in problemi))

    def test_le_pipe_escapate_non_sono_colonne_in_piu(self):
        p = self._scrivi("# T\n\n## A\n\n| o | s |\n|---|---|\n"
                         "| `--backend auto\\|cpu\\|vulkan` | va bene |\n")
        problemi, _ = self.mod.controlla(p)
        self.assertEqual(problemi, [], f"falso allarme: {problemi}")

    def test_metodo_e_placeholder_non_sono_segnaposto(self):
        p = self._scrivi("# T\n\n## A\n\nIl metodo usa un placeholder di "
                         "VirtualAlloc2.\n")
        problemi, _ = self.mod.controlla(p)
        self.assertEqual(problemi, [], f"falso allarme: {problemi}")

    def test_un_segnaposto_vero_viene_trovato(self):
        """Il controprova: con i confini di parola, TODO da solo deve uscire."""
        p = self._scrivi("# T\n\n## A\n\nTODO: finire questa sezione.\n")
        problemi, _ = self.mod.controlla(p)
        self.assertTrue(any("TODO" in x for x in problemi))

    def test_i_titoli_dentro_i_blocchi_di_codice_non_contano(self):
        """`# commento` dentro un blocco shell non e' un titolo, e metterlo in
        indice produrrebbe una voce che non porta da nessuna parte."""
        p = self._scrivi("# T\n\n## A\n\n```bash\n# questo e' un commento\nls\n```\n")
        tt = self.mod.titoli(p.read_text().splitlines())
        self.assertEqual([t[2] for t in tt], ["T", "A"])

    def test_l_indice_del_readme_vero_e_aggiornato(self):
        """Il controllo che vale: il README del progetto, cosi' com'e'."""
        readme = Path(__file__).resolve().parents[1] / "README.md"
        problemi, _ = self.mod.controlla(readme)
        self.assertEqual(problemi, [], "\n".join(problemi))
        testo = readme.read_text(encoding="utf-8")
        self.assertIn(self.mod.INIZIO, testo, "il README non ha l'indice generato")


class TestStatisticheCache(unittest.TestCase):
    """Le righe del resoconto dello store, tradotte in numeri e registrate in
    `runs.jsonl`.

    Esistono perche' il dato piu' importante prodotto dal progetto viveva solo
    nella scrollback di un terminale: su Qwen3-Next-80B l'11,7% di residenza ha
    dato il **91,2% di hit**, ed e' quel numero — non la velocita' — che ha
    spiegato perche' due previsioni di fila fossero sbagliate. Il log registrava
    velocita', TTFT, RAM e risorse, e non lo registrava.

    Le righe di questo test sono copiate **verbatim** da un'esecuzione reale del
    2026-10-03: un parser provato contro righe inventate da chi lo scrive non
    prova niente."""

    RIGHE = [
        "argine: expert cache (in corso): 1771302 hit, 170088 miss (91.2% hit), "
        "161511 sfratti, 81303 MiB letti dallo store, 4092/4096 MiB residenti",
        "argine: tempo (ms, somma sui thread): lettura 55444 (44%), pagine nuove 0 "
        "(0%), commit 10327 (8%), decommit 39391 (31%), attesa 0 (0%), lock 20888 (17%)",
        "argine: dentro il nostro codice 184433 ms su 57666 nodi MoE (3.20 ms per "
        "nodo); di cui I/O 126049 ms (68%) e attesa fra thread 58384 ms (32%)",
        "argine: parete 75167 ms; profondita' di coda media 2.14 (letture "
        "contemporanee), banda 1082 MB/s aggregati, 1466 MB/s per blocco di lettura",
        "argine: 58611 attese, 2.90 letture in volo per attesa; ns_read e' il parete "
        "dell'attesa, non la somma delle letture: la coda qui sopra ne tiene conto",
        "argine: per miss: 0.326 ms di lettura, 0.000 ms di pagine nuove, 0.061 ms "
        "di commit",
    ]

    def test_legge_tutti_i_numeri_che_contano(self):
        n = srv.LlamaServerProcess.statistiche_cache(self.RIGHE)
        atteso = {
            "hit": 1771302, "miss": 170088, "hit_pct": 91.2, "sfratti": 161511,
            "mib_letti": 81303, "mib_residenti": 4092, "mib_budget": 4096,
            "ms_lettura": 55444, "ms_pagine": 0, "ms_commit": 10327,
            "ms_decommit": 39391, "ms_attesa": 0, "ms_lock": 20888,
            "ms_nostro_codice": 184433, "nodi_moe": 57666, "ms_io": 126049,
            "ms_attesa_thread": 58384, "ms_parete": 75167, "coda": 2.14,
            "mbs_aggregati": 1082, "mbs_per_blocco": 1466,
            "attese": 58611, "letture_per_attesa": 2.90,
            "ms_lettura_miss": 0.326, "ms_pagine_miss": 0.0,
            "ms_commit_miss": 0.061,
        }
        for k, v in atteso.items():
            with self.subTest(campo=k):
                self.assertIn(k, n, f"{k} non estratto dalle righe vere")
                self.assertAlmostEqual(n[k], v, places=3)

    def test_il_tasso_di_hit_si_ricava_e_torna(self):
        """La coerenza interna: hit/(hit+miss) deve dare la percentuale stampata.
        Se il parser prendesse i due numeri da righe diverse non tornerebbe."""
        n = srv.LlamaServerProcess.statistiche_cache(self.RIGHE)
        calcolato = 100.0 * n["hit"] / (n["hit"] + n["miss"])
        self.assertAlmostEqual(calcolato, n["hit_pct"], places=1)

    def test_righe_assenti_danno_niente_non_zeri(self):
        """Un formato cambiato deve produrre numeri mancanti, non numeri
        sbagliati: uno zero finto in un log e' peggio di un buco."""
        n = srv.LlamaServerProcess.statistiche_cache(["argine: qualcosa d'altro"])
        self.assertEqual(n, {})
        self.assertEqual(srv.LlamaServerProcess.statistiche_cache([]), {})
        self.assertEqual(srv.LlamaServerProcess.statistiche_cache(None), {})

    def test_una_sola_fonte_per_i_tempi_per_miss(self):
        """`_tempi_per_miss` delegava a una regex sua: due letture della stessa
        riga sono due fonti che possono divergere."""
        st = {"resoconto": self.RIGHE,
              "numeri": srv.LlamaServerProcess.statistiche_cache(self.RIGHE)}
        t = cli._tempi_per_miss(st)
        self.assertAlmostEqual(t["ms_lettura_miss"], 0.326)
        self.assertAlmostEqual(t["ms_commit_miss"], 0.061)
        # e senza 'numeri' precalcolati deve arrivare allo stesso risultato
        self.assertEqual(cli._tempi_per_miss({"resoconto": self.RIGHE}), t)

    def test_il_resoconto_passa_per_lo_stato(self):
        proc = srv.LlamaServerProcess.__new__(srv.LlamaServerProcess)
        proc._lock = threading.RLock()
        proc._psx_lines = [
            "argine: expert store aperto: m.psxs, 48 layer x 512 expert, budget "
            "4096 MiB, 16 settori di cache, cache dell'OS scavalcata, lettura asincrono",
        ] + self.RIGHE + [
            "argine: copertura: 144 tensori di expert su 144 serviti dallo store",
        ]
        proc._load_lines = []
        st = proc.expert_store_status()
        self.assertEqual(st["numeri"]["hit_pct"], 91.2)
        self.assertEqual(st["copertura"], (144, 144))
        self.assertEqual(st["lettura"], "asincrono")


class TestCoperturaStore(unittest.TestCase):
    """Una copertura parziale e' il peggiore degli esiti possibili perche' si
    traveste da successo: "IN USO" compare, la cache riporta hit e miss
    plausibili, il budget risulta rispettato, e intanto una fetta degli expert
    vive in memoria mappata dove il tetto non vale.

    Le cause sono tutte silenziose e tutte reali: gate e up fusi, slice non
    allineate per un solo tipo, piu' expert del limite della build (Qwen3-Next ne
    ha 512, che e' esattamente il limite)."""

    def _stato(self, riga):
        proc = srv.LlamaServerProcess.__new__(srv.LlamaServerProcess)
        proc._lock = threading.RLock()
        proc._psx_lines = [riga]
        proc._load_lines = []
        return proc.expert_store_status()

    def test_la_riga_di_copertura_viene_letta(self):
        st = self._stato("argine: copertura: 96 tensori di expert su 144 serviti "
                         "dallo store")
        self.assertEqual(st["copertura"], (96, 144))

    def test_copertura_piena_non_avvisa(self):
        st = self._stato("argine: copertura: 144 tensori di expert su 144 serviti "
                         "dallo store")
        self.assertTrue(cli._avvisa_copertura(st, stampa=False))

    def test_copertura_parziale_avvisa_e_dice_quanto(self):
        st = self._stato("argine: copertura: 48 tensori di expert su 144 serviti "
                         "dallo store")
        self.assertFalse(cli._avvisa_copertura(st, stampa=False))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            cli._avvisa_copertura(st, stampa=True)
        out = buf.getvalue()
        self.assertIn("48", out)
        self.assertIn("144", out)
        self.assertIn("33%", out)
        self.assertIn("non e' un", out)

    def test_senza_la_riga_non_si_afferma_niente(self):
        """Un binario vecchio non la stampa: in quel caso non si deve dichiarare
        ne' copertura piena ne' parziale."""
        st = self._stato("argine: expert store aperto: x.psxs, 48 layer x 128 "
                         "expert, budget 2048 MiB")
        self.assertIsNone(st["copertura"])
        self.assertTrue(cli._avvisa_copertura(st, stampa=False))


class TestModoLettura(unittest.TestCase):
    """I tre strati di lettura dello store sono selezionabili a runtime perche'
    il loro confronto e' un risultato, e un risultato che nessuno puo' rifare e'
    un'opinione: le prime misure (+7-11% per l'asincrono) venivano da tre binari
    diversi in tre momenti diversi, e due di quei binari non esistono piu'.

    Qui si verifica la parte che puo' sbagliare in silenzio: accettare un modo
    scritto male, e — soprattutto — etichettare una misura col modo *chiesto*
    invece di quello *avvenuto*."""

    def test_default_asincrono(self):
        self.assertEqual(cli._modo_lettura(None), "asincrono")
        self.assertEqual(cli._modo_lettura(""), "asincrono")

    def test_nomi_validi_e_dimensione_del_blocco(self):
        self.assertEqual(cli._modo_lettura("sincrono"), "sincrono")
        self.assertEqual(cli._modo_lettura("BLOCCO"), "blocco")
        self.assertEqual(cli._modo_lettura(" blocco:4 "), "blocco:4")

    def test_un_modo_inventato_e_un_errore_non_un_default(self):
        """Ripiegare in silenzio sull'asincrono produrrebbe una riga di tabella
        etichettata 'sincrono' e misurata asincrona: il peggiore dei due esiti."""
        for cattivo in ("asincrna", "veloce", "1", "sincrono:2", "blocco:0",
                        "blocco:99", "blocco:x"):
            with self.subTest(cattivo=cattivo):
                with self.assertRaises(SystemExit) as e:
                    cli._modo_lettura(cattivo)
                self.assertEqual(e.exception.code, cli.EXIT_USAGE)

    def test_conferma_legge_il_log_non_la_variabile(self):
        # il motore dice cosa ha fatto: concorde
        self.assertTrue(cli._conferma_modo_lettura({"lettura": "sincrono"},
                                                   "sincrono"))
        # "blocco" chiesto senza dimensione: il motore stampa la sua (2)
        self.assertTrue(cli._conferma_modo_lettura({"lettura": "blocco:2"},
                                                   "blocco"))
        # discorde: va detto, perche' la riga che segue non misura cio' che dice
        self.assertFalse(cli._conferma_modo_lettura({"lettura": "asincrono"},
                                                    "sincrono"))
        # dimensione diversa da quella chiesta: ancora discorde
        self.assertFalse(cli._conferma_modo_lettura({"lettura": "blocco:2"},
                                                    "blocco:4"))
        # nessun log e store non aperto: il problema e' un altro, lo dice lui
        self.assertTrue(cli._conferma_modo_lettura({}, "sincrono"))
        self.assertTrue(cli._conferma_modo_lettura(None, "sincrono"))

    def test_un_binario_vecchio_non_passa_per_buono(self):
        """Il buco che la realta' ha infilato lo stesso giorno: il binario in uso
        era di tre ore prima e non conosceva ARGINE_EXPERT_READ. La prima versione
        tornava True ("nessun log da cui dedurlo") e uno sweep sui tre modi
        avrebbe prodotto tre colonne identiche spacciate per un confronto.

        L'assenza non e' ambigua: la riga "expert store aperto" la stampa
        qualunque binario capace di store, quindi se c'e' quella e manca il modo,
        il binario e' anteriore all'interruttore."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            esito = cli._conferma_modo_lettura({"aperto": True, "lettura": None},
                                               "sincrono")
        self.assertFalse(esito)
        self.assertIn("doctor", buf.getvalue())

    def test_la_copertura_assente_con_store_aperto_viene_dichiarata(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            cli._avvisa_copertura({"aperto": True, "copertura": None}, stampa=True)
        out = buf.getvalue()
        self.assertIn("non posso confermare", out.replace("\n", " "))

    def test_il_log_del_motore_viene_davvero_letto(self):
        """La riga del C++ e il parser di Python sono scritti in due file
        diversi: questo test li mette uno contro l'altro, perche' altrimenti un
        cambio di formulazione nel messaggio spegnerebbe il controllo senza che
        nessun test diventi rosso."""
        proc = srv.LlamaServerProcess.__new__(srv.LlamaServerProcess)
        proc._lock = threading.RLock()
        proc._psx_lines = [
            "argine: expert store aperto: m.psxs, 48 layer x 128 expert, "
            "budget 2048 MiB, 16 settori di cache, cache dell'OS scavalcata, "
            "lettura blocco:2",
        ]
        proc._load_lines = []
        st = proc.expert_store_status()
        self.assertTrue(st["aperto"])
        self.assertEqual(st["lettura"], "blocco:2")

    def test_il_modo_finisce_nell_ambiente_del_processo(self):
        """Si dichiara sempre, come PRETOUCH: un residuo in ambiente sposterebbe
        la velocita' del 7-11%, cioe' piu' dell'effetto che si sta misurando."""
        tmp = Path(_TMP.name) / "modo"  # type: ignore[union-attr]
        tmp.mkdir(exist_ok=True)
        gguf = tmp / "m.gguf"
        gguf.write_bytes(b"x")
        (tmp / "m.psxs").write_bytes(b"y")
        m = types.SimpleNamespace(path=gguf, expert_bytes=1 << 30)
        args = argparse.Namespace(expert_store="auto", expert_budget="1G",
                                  expert_direct=True, verbose=False,
                                  expert_stats=False, pretouch=False,
                                  read_mode=None)
        env = cli._expert_store_env(args, m)  # type: ignore[arg-type]
        self.assertEqual(env["ARGINE_EXPERT_READ"], "asincrono")
        args.read_mode = "sincrono"
        env = cli._expert_store_env(args, m)  # type: ignore[arg-type]
        self.assertEqual(env["ARGINE_EXPERT_READ"], "sincrono")


if __name__ == "__main__":
    unittest.main(verbosity=2)
