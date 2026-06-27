# AGENT.md

## Scope

This repository generates Japanese two-speaker dialogue data with
`Qwen/Qwen3-30B-A3B-Instruct-2507`.

The primary production workflow is direct local inference:

- load model weights from the local Hugging Face cache;
- load the model once with Transformers and PyTorch;
- generate several output files per GPU batch;
- do not send prompts or generated dialogue to an external API;
- do not use the local vLLM HTTP API unless the user explicitly asks for it.

There are two supported dataset workflows:

1. generate random travel dialogues from a prompt;
2. rewrite JMultiWOZ per-dialogue TSV files into natural JSON dialogues.

The expected workspace path on the current machine is:

```text
/mnt/kiso-qnap5/kitamura/projects/text-dialogue-qwen3
```

Run project commands from this directory unless a command explicitly says
otherwise.

## Current machine assumptions

- GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, about 96 GB VRAM
- Supported CUDA runtime: CUDA 12.8
- Local generation runtime: PyTorch 2.8.0+cu128 and Transformers 4.57.x
- Model cache size: approximately 57 GB
- Production model: `Qwen/Qwen3-30B-A3B-Instruct-2507`

Do not install the unpinned latest PyTorch from PyPI on this machine. It may
select a CUDA 13 build that the installed driver cannot use.

## Repository map

- `prompts/random_travel_dialogues_json.txt`: production prompt for random
  travel dialogues
- `prompts/rewrite_jmultiwoz_tsv_to_json_one_dialogue.txt`: prompt for
  rewriting one JMultiWOZ TSV dialogue into one JSON dialogue
- `jmultiwoz-tsv-by-dialogue/`: source JMultiWOZ TSV files copied into this
  repository so the rewrite workflow can run from this project alone
- `generate_random_travel_local_batch.py`: persistent direct-local batch
  generator; loads the model once
- `rewrite_jmultiwoz_tsv_to_json_batch.py`: persistent direct-local batch
  generator for JMultiWOZ rewriting; loads the model once
- `scripts/generate_random_travel_json_5000.sh`: production launcher for 1,000
  files / 5,000 dialogues
- `scripts/rewrite_jmultiwoz_tsv_to_json.sh`: production launcher for
  JMultiWOZ TSV-to-JSON rewriting
- `scripts/setup_local.sh`: creates the CUDA 12.8 local runtime in `/tmp`
- `normalize_dialogue_json.py`: mechanical JSON repair and normalization
- `normalize_jmultiwoz_json_text.py`: mechanical TTS-oriented text
  normalization for generated JMultiWOZ JSON outputs
- `repair_jmultiwoz_partial_json.py`: mechanical repair helper for malformed
  JMultiWOZ partial JSON outputs
- `validate_dialogue_json.py`: structural output validation
- `outputs/random_travel_dialogues_5x1000/`: generated production dataset; final
  four-digit JSON files are tracked by Git
- `outputs/jmultiwoz_rewritten_json/`: generated JMultiWOZ rewrite dataset;
  final JSON files may be tracked by Git
- `.hf-cache/`: local model cache; ignored by Git

The repository still contains vLLM/OpenAI-compatible scripts for optional
experiments. They are not the default production path. Do not start
`scripts/serve_vllm.sh` or use `scripts/generate_random_travel_json.sh` unless
the user explicitly requests the local HTTP API workflow.

The directory `.jmultiwoz-tsv-by-dialogue.bad-acl-copy/`, if present, is an
old failed copy with inherited ACL problems. It is ignored by Git and must not
be used as input. The correct source directory is `jmultiwoz-tsv-by-dialogue/`.

## Git exclusions

Never commit any of the following:

- `.hf-cache/` and model weights
- `.venv*`, `.uv-cache*`, or `/tmp` runtime environments
- generated outputs outside the production 5,000-dialogue dataset
- all `*.partial.json` files and malformed retry artifacts
- `.jmultiwoz-tsv-by-dialogue.bad-acl-copy/`
- `__pycache__/`, `*.pyc`, or `*.egg-info/`
- access tokens, credentials, local email, or machine-specific secrets

The model cache is roughly 57 GB and does not belong in Git. The completed
5,000-dialogue production dataset is expected to be about 5–10 MB and is
intended to be published in this repository. Only final files matching
`random_travel_dialogues.NNNN.json` are tracked; partial files remain ignored.

The JMultiWOZ source TSV files are intentionally copied into this repository
under `jmultiwoz-tsv-by-dialogue/`. This keeps the rewrite workflow
self-contained. Generated final JSON files under
`outputs/jmultiwoz_rewritten_json/` may also be published when the user wants
the rewritten dataset in the repository. Never commit `*.partial.json`.

## Local environment setup

The NAS workspace has previously produced permission errors while installing
large Python wheels. The supported approach is to put the disposable Python
environment under `/tmp` and keep only the model cache in the project.

Create or refresh the local CUDA 12.8 environment:

```bash
cd /mnt/kiso-qnap5/kitamura/projects/text-dialogue-qwen3
bash scripts/setup_local.sh
```

This creates:

```text
/tmp/text-dialogue-qwen3-local-venv
```

`/tmp` may be cleared by an OS reboot. After a reboot, rerun
`scripts/setup_local.sh`. Closing SSH, closing a terminal, detaching tmux, or
locking the screen does not remove this environment.

Verify CUDA and the GPU:

```bash
/tmp/text-dialogue-qwen3-local-venv/bin/python -c \
  "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected major values are PyTorch `2.8.0+cu128`, CUDA `12.8`, and
`torch.cuda.is_available() == True`.

## Model cache

The production launcher exports:

```text
HF_HOME=/mnt/kiso-qnap5/kitamura/projects/text-dialogue-qwen3/.hf-cache
```

The batch generator calls `from_pretrained(..., local_files_only=True)`. Once
the cache exists, generation must not download a model or access an inference
service.

Check that the current cache exists:

```bash
du -sh .hf-cache/hub/models--Qwen--Qwen3-30B-A3B-Instruct-2507
```

On a new machine, download the model once after running `setup_local.sh`:

```bash
HF_HOME="$PWD/.hf-cache" \
  /tmp/text-dialogue-qwen3-local-venv/bin/hf download \
  Qwen/Qwen3-30B-A3B-Instruct-2507
```

Set `HF_TOKEN` only if Hugging Face requires authentication. Never write the
token into this repository.

## Production generation

This section is for random prompt-based travel dialogue generation.

The production target is:

- 5 dialogues per JSON file;
- 1,000 JSON files;
- 5,000 dialogues total;
- filenames `random_travel_dialogues.0001.json` through
  `random_travel_dialogues.1000.json`;
- output directory `outputs/random_travel_dialogues_5x1000/`.

Start a tmux session:

```bash
tmux new -s qwen5000
```

If it already exists:

```bash
tmux attach -d -t qwen5000
```

Inside tmux, run:

```bash
cd /mnt/kiso-qnap5/kitamura/projects/text-dialogue-qwen3
bash scripts/generate_random_travel_json_5000.sh
```

Detach by pressing `Ctrl+b`, releasing both keys, and then pressing `d`.

The PC must remain awake. tmux survives terminal and SSH disconnection, but it
cannot continue GPU inference while the machine is suspended. Screen lock and
screen-off are safe. `systemd-inhibit` may return `Access denied` on this
machine; use the desktop power settings or an administrator-approved method to
disable sleep instead.

## Batch behavior

`generate_random_travel_local_batch.py` performs the following:

1. Reads the prompt once.
2. Loads tokenizer and model once from `.hf-cache`.
3. Sends four identical prompts through one Transformers GPU batch by default.
4. Writes each raw response to `*.partial.json`.
5. Mechanically normalizes the JSON.
6. Validates the normalized structure.
7. Renames a valid partial file to the final `*.json` filename.
8. Queues invalid partial output for retry, up to three attempts.

A `*.partial.json` file is not a completed dataset file. It is intentionally
kept for diagnosis after malformed model output. Do not count or publish it as
finished output. A successful retry creates the corresponding final `.json`.

The default `RESUME=1` behavior validates completed files and skips valid
ones. This makes rerunning the launcher safe after interruption. Use
`RESUME=0` only when the user explicitly wants every file regenerated.

Common controls:

```bash
BATCH_SIZE=2 bash scripts/generate_random_travel_json_5000.sh
BATCH_COUNT=20 bash scripts/generate_random_travel_json_5000.sh
MAX_RETRIES=5 bash scripts/generate_random_travel_json_5000.sh
RESUME=0 bash scripts/generate_random_travel_json_5000.sh
```

- Reduce `BATCH_SIZE` after a CUDA out-of-memory error.
- Keep `MAX_TOKENS=4096` unless the prompt or output size changes materially.
- Seeds are derived from `BASE_SEED`, file index, and retry number.
- Do not start two production generators on the same GPU and output directory.

## JMultiWOZ TSV rewrite generation

This workflow rewrites existing JMultiWOZ TSV dialogues into the same
one-dialogue JSON style used by the generated text-dialogue data. It does not
create the dialogue from scratch; it edits each source TSV dialogue into more
natural Japanese while preserving the important facts.

Input:

```text
jmultiwoz-tsv-by-dialogue/{train,validation,test}/*.tsv
```

Output:

```text
outputs/jmultiwoz_rewritten_json/{train,validation,test}/*.json
```

Each source TSV file becomes exactly one JSON file. The expected shape is:

```json
[
  [
    ["SYSTEM", "こんにちは、何か旅行の予定はありますか？"],
    ["USER", "..."],
    ["SYSTEM", "..."]
  ]
]
```

The first turn must be exactly:

```json
["SYSTEM", "こんにちは、何か旅行の予定はありますか？"]
```

Run the rewrite workflow:

```bash
cd /mnt/kiso-qnap5/kitamura/projects/text-dialogue-qwen3
bash scripts/rewrite_jmultiwoz_tsv_to_json.sh
```

For quality-first generation, prefer one prompt per GPU batch:

```bash
BATCH_SIZE=1 bash scripts/rewrite_jmultiwoz_tsv_to_json.sh
```

For a small smoke test:

```bash
SPLITS=validation LIMIT=3 BATCH_SIZE=1 \
  bash scripts/rewrite_jmultiwoz_tsv_to_json.sh
```

Default controls:

```text
BATCH_SIZE=2
MAX_TOKENS=2048
TEMPERATURE=0.55
TOP_P=0.9
MAX_RETRIES=3
RESUME=1
```

`RESUME=1` validates completed final `.json` files and skips valid ones.
Existing `.partial.json` files are not treated as completed outputs. Use
`RESUME=0` only when the user explicitly wants every output regenerated.

The rewrite prompt deliberately balances two goals:

- keep all important facts: locations, facilities, addresses, phone numbers,
  dates, times, people counts, nights, fees, reservation numbers, and
  conditions;
- avoid copying the TSV utterances verbatim; rewrite wording, endings, and
  turn flow into more natural spoken Japanese.

Do not weaken the fact-preservation rules just to make the dialogue sound more
natural. If outputs are too close to the source TSV, adjust the wording-copy
rule or sampling settings first.

The JMultiWOZ generator also normalizes TTS-sensitive text mechanically:

- `2/7` or `2／7` -> `2月7日`
- `13:40` or `13：40` -> `13時40分`
- `2名`, `2人`, `2泊` -> `二名`, `二人`, `二泊`
- trailing interruption dashes -> `、`

Repair existing malformed partial outputs after an interrupted run:

```bash
/tmp/text-dialogue-qwen3-local-venv/bin/python \
  repair_jmultiwoz_partial_json.py \
  --output-root outputs/jmultiwoz_rewritten_json \
  --delete-repaired
```

Normalize final JMultiWOZ JSON text for TTS without regenerating:

```bash
/tmp/text-dialogue-qwen3-local-venv/bin/python \
  normalize_jmultiwoz_json_text.py \
  --output-root outputs/jmultiwoz_rewritten_json
```

Count completed JMultiWOZ JSON files:

```bash
find outputs/jmultiwoz_rewritten_json \
  -type f -name '*.json' ! -name '*.partial.json' | wc -l
```

Count JMultiWOZ partial files:

```bash
find outputs/jmultiwoz_rewritten_json \
  -type f -name '*.partial.json' | wc -l
```

## Monitoring

Count completed files without counting partial files:

```bash
find outputs/random_travel_dialogues_5x1000 \
  -maxdepth 1 -type f -name 'random_travel_dialogues.[0-9][0-9][0-9][0-9].json' \
  | wc -l
```

Count partial files:

```bash
find outputs/random_travel_dialogues_5x1000 \
  -maxdepth 1 -type f -name '*.partial.json' | wc -l
```

Monitor the GPU:

```bash
nvidia-smi
```

Inspect the tmux output:

```bash
tmux attach -d -t qwen5000
```

Stop generation with `Ctrl+c`. Restart with the same launcher; valid completed
files are skipped.

## Normalization and validation limits

`normalize_dialogue_json.py` only performs mechanical repairs:

- removes a Markdown JSON code fence;
- combines multiple sequential JSON arrays into the expected outer array;
- splits a flattened `[speaker, utterance, speaker, utterance]` record;
- replaces a trailing interruption dash with `、`.

It does not invent, rewrite, fact-check, or stylistically improve dialogue.

`validate_dialogue_json.py` currently checks:

- expected dialogue count;
- each dialogue is an array;
- each turn is `[speaker, utterance]`;
- speaker is `SYSTEM` or `USER`;
- utterance is a non-empty string;
- each dialogue has at least six turns.

It intentionally does not enforce the prompt's approximate 12–16 turn range.
It also does not enforce style, factual accuracy, speaker alternation, or the
number of topic rejections. Add explicit validation before treating any of
those prompt conditions as hard dataset guarantees.

Validate one completed production file:

```bash
/tmp/text-dialogue-qwen3-local-venv/bin/python \
  -m validate_dialogue_json \
  outputs/random_travel_dialogues_5x1000/random_travel_dialogues.0001.json \
  --expected-count 5
```

## Safe development rules

- Preserve a running generation process unless the user asks to stop it or a
  confirmed defect would corrupt the requested dataset.
- Inspect `nvidia-smi`, tmux logs, completed count, and partial count before
  diagnosing a stalled run.
- Model loading from NAS can appear frozen for tens of seconds per shard.
- Never delete `.hf-cache` to solve an environment problem.
- Do not overwrite completed outputs during code-only changes.
- Test shell scripts with `bash -n`.
- Test Python entry points with the CUDA 12.8 local Python under `/tmp`.
- Use `rg` for repository searches.

Useful non-generating checks:

```bash
bash -n scripts/setup_local.sh \
  scripts/generate_random_travel_json_5000.sh \
  scripts/rewrite_jmultiwoz_tsv_to_json.sh
/tmp/text-dialogue-qwen3-local-venv/bin/python \
  -m generate_random_travel_local_batch --help
/tmp/text-dialogue-qwen3-local-venv/bin/python \
  rewrite_jmultiwoz_tsv_to_json_batch.py --help
```

## Publishing this repository

The repository should contain source, prompts, scripts, lock files,
documentation, small examples, and the completed production dataset under
`outputs/random_travel_dialogues_5x1000/`. It may also contain the copied
JMultiWOZ source TSV files under `jmultiwoz-tsv-by-dialogue/` and final
rewritten JSON files under `outputs/jmultiwoz_rewritten_json/` when the user
wants the rewritten dataset published.

Keep caches, environments, single-run test outputs, and partial retry artifacts
local. Before publishing the random travel dataset, confirm that all 1,000
expected final files exist and validate successfully. Before publishing the
JMultiWOZ rewrite dataset, confirm that all 4,246 expected final JSON files
exist and no `*.partial.json` files remain.
