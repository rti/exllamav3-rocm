"""Concurrent long-context slots: jobs with natural (source-code) prompts sharing one paged cache.
usage: bench_slots.py --cache 131072 --prompt 120000,8000 [--tokens 1500,512] [--stagger 100] [--task code]
                      [--mtp | -dm DRAFT] [--kv 8] [--dkv 4]
--stagger N: enqueue jobs 1.. only after job 0 has decoded N tokens (second request arriving mid-generation);
job 0's decode rate is then reported per phase: alone / during job 1 prefill / both decoding / alone again."""
import argparse, glob, os, time, torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import ComboSampler
from exllamav3.cache import CacheLayer_quant

ap = argparse.ArgumentParser()
ap.add_argument("-m", default = "../models/Qwen3.8-27B-exl3-SC4.0")
ap.add_argument("-dm", default = None)
ap.add_argument("--mtp", action = "store_true")
ap.add_argument("--cache", type = int, required = True)
ap.add_argument("--prompt", required = True, help = "prompt tokens per slot, comma-separated (one value per job)")
ap.add_argument("--slots", type = int, default = None, help = "recurrent slots (default: number of jobs)")
ap.add_argument("--kv", type = int, default = 8)
ap.add_argument("--dkv", type = int, default = 4)
ap.add_argument("--tokens", default = "512", help = "max new tokens per job, comma-separated (last value repeats)")
ap.add_argument("--stagger", type = int, default = 0)
ap.add_argument("--task", choices = ["explain", "code"], default = "explain")
ap.add_argument("--temp", type = float, default = 0.6)
ap.add_argument("--load_only", action = "store_true")
args = ap.parse_args()
prompts = [int(p) for p in args.prompt.split(",")]
max_new = [int(t) for t in args.tokens.split(",")]
max_new += [max_new[-1]] * (len(prompts) - len(max_new))
args.slots = args.slots or len(prompts)

GiB = 1024 ** 3
def mem(tag):
    free, total = torch.cuda.mem_get_info()
    print(f"[{tag}] device used {(total - free) / GiB:.2f} / {total / GiB:.2f} GiB, "
          f"torch peak {torch.cuda.max_memory_allocated() / GiB:.2f} GiB", flush = True)

config = Config.from_directory(args.m)
model = Model.from_config(config)
dm = Model.from_config(config, component = "mtp") if args.mtp else \
     (Model.from_config(Config.from_directory(args.dm)) if args.dm else None)
mh = dm.caps.get("default_draft_size", 4) if dm else 0
qkw = lambda b: dict(layer_type = CacheLayer_quant, k_bits = b, v_bits = b)
cache = Cache(model, max_num_tokens = args.cache, max_history = mh, max_batch_size = args.slots, **qkw(args.kv))
model.load(progressbar = False)
dc = None
if dm:
    dc = Cache(dm, max_num_tokens = args.cache, **qkw(args.dkv))
    dm.load(progressbar = False)
tok = Tokenizer.from_config(config)
gen = Generator(model = model, cache = cache, tokenizer = tok, draft_model = dm, draft_cache = dc,
                max_batch_size = args.slots)
mem("loaded")
if args.load_only: raise SystemExit

# Natural long prompts: different slices of this repo's sources per slot
src = []
for pat in ["../exllamav3/**/*.py", "../exllamav3/**/*.cu", "../exllamav3/**/*.cuh", "../exllamav3/**/*.cpp"]:
    for f in sorted(glob.glob(pat, recursive = True)):
        src.append(f"# file: {f}\n" + open(f, errors = "ignore").read())
corpus = tok.encode("\n\n".join(src))
print("corpus tokens:", corpus.shape[-1])
questions = {
    "explain": ["Explain how the paged KV cache and the generator's job scheduling work in the code above.",
                "Describe how speculative decoding with a draft model is implemented in the code above."],
    "code": ["Write a new Python module, in the style of the code above, that adds an LRU-evicting prefix cache "
             "for the generator: class, methods, type hints, docstrings, and pytest unit tests. Output only code.",
             "Write a Python function, in the style of the code above, that validates a model config dict "
             "(required keys, types, ranges) and raises descriptive errors. Include pytest tests. Output only code."],
}[args.task]
head = tok.encode("<|im_start|>user\n", encode_special_tokens = True)
jobs = []
for i, plen in enumerate(prompts):
    q = tok.encode("\n\n" + questions[i % len(questions)])
    tail = tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", encode_special_tokens = True)
    n = plen - head.shape[-1] - q.shape[-1] - tail.shape[-1]
    off = (i * corpus.shape[-1] // len(prompts)) % max(corpus.shape[-1] - n, 1)
    body = corpus[:, off:off + n]
    assert body.shape[-1] == n, f"corpus too small ({corpus.shape[-1]}) for {n}"
    ids = torch.cat([head, body, q, tail], dim = -1)
    jobs.append(Job(input_ids = ids, max_new_tokens = max_new[i], identifier = i,
                    sampler = ComboSampler(temperature = args.temp, top_p = 0.95, top_k = 20),
                    stop_conditions = [tok.eos_token_id, "<|im_end|>"]))
pending = jobs[1:] if args.stagger else []
for j in (jobs[:1] if args.stagger else jobs): gen.enqueue(j)

t0 = time.perf_counter()
first, last_t, res, ntok = {}, {}, {}, {}
t_enq = {0: t0} if args.stagger else {i: t0 for i in range(len(jobs))}
trace0 = []  # (time, tokens) for job 0
while gen.num_remaining_jobs() or pending:
    for r in gen.iterate():
        if r.get("stage") == "error": raise r["error"]
        if "identifier" not in r:
            print("non-job result:", {k: v for k, v in r.items() if not torch.is_tensor(v)}, flush = True); continue
        i = r["identifier"]; now = time.perf_counter()
        k = r["token_ids"].shape[-1] if torch.is_tensor(r.get("token_ids")) else 0
        if k:
            first.setdefault(i, now); last_t[i] = now; ntok[i] = ntok.get(i, 0) + k
            if i == 0: trace0.append((now, k))
        if r.get("eos"): res[i] = r; last_t[i] = now
    if pending and ntok.get(0, 0) >= args.stagger:
        for j in pending: gen.enqueue(j); t_enq[j.identifier] = time.perf_counter()
        pending = []
mem("after run")
tot = 0
for i, j in enumerate(jobs):
    r = res[i]; n = r["new_tokens"]; acc = r.get("accepted_draft_tokens", 0); rej = r.get("rejected_draft_tokens", 0)
    tot += n
    print(f"slot {i}: prompt {prompts[i]} tok, TTFT {first[i] - t_enq[i]:.1f}s, {n} tok, "
          f"decode {(n - 1) / (last_t[i] - first[i]):.1f} tok/s, draft {acc}/{acc + rej}", flush = True)
    print("   ", r.get("full_completion", "")[:160].replace("\n", " "))

def rate(a, b):
    toks = sum(k for t, k in trace0 if a < t <= b)
    return f"{toks / (b - a):.1f} tok/s ({toks} tok in {b - a:.1f}s)" if b > a and toks else "n/a"
if args.stagger and len(jobs) > 1:
    e1, f1, l1 = t_enq[1], first[1], last_t[1]
    print("job 0 phases:")
    print("  alone (before job 1)    ", rate(first[0], e1))
    print("  during job 1 prefill    ", rate(e1, f1))
    print("  both decoding           ", rate(f1, min(l1, last_t[0])))
    print("  alone (after job 1)     ", rate(l1, last_t[0]))
else:
    overlap = max(0.0, min(last_t.values()) - max(first.values()))
    print(f"decode overlap window {overlap:.1f}s; aggregate {tot / (max(last_t.values()) - min(first.values())):.1f} tok/s over decode span")
