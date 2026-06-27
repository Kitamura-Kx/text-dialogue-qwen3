# text-dialogue-qwen3
Qwen3-30B-A3B-Instruct-2507を用いて、音声合成に用いる二話者の日本語のテキスト対話を生成するディレクトリです。  
デフォルトの話題ドメインは旅行・温泉・観光になっています。プロンプトで調整してください。

公式モデルカードでは、このモデルは `qwen3_moe` アーキテクチャなので `transformers>=4.51.0` が必要です。ローカルで OpenAI 互換 API として配信する場合は `vllm>=0.8.5` または `sglang>=0.4.6.post1` が案内されています。30.5B total / 3.3B activated の MoE モデルなので、NVIDIA RTX PRO 6000 Blackwell 96GB のような大きめの単一 GPU では vLLM サーバ経由がおすすめです。

## セットアップ

```bash
cd /mnt/kiso-qnap5/kitamura/projects/text-dialogue-qwen3
UV_CACHE_DIR=/tmp/text-dialogue-qwen3-uv-cache UV_LINK_MODE=copy uv sync
```

transformers で直接ロードする場合:

```bash
UV_CACHE_DIR=/tmp/text-dialogue-qwen3-uv-cache UV_LINK_MODE=copy uv sync --extra local
```

vLLMはCUDA 12.8用の専用環境 `/tmp/text-dialogue-qwen3-vllm-venv` へインストールします。OSを再起動して `/tmp` が消えた場合は、同じコマンドを再実行してください。

```bash
bash scripts/setup_vllm.sh
```

## vLLM サーバを起動

GPU が使える環境で起動します。`scripts/serve_vllm.sh` は RTX PRO 6000 Blackwell 96GB を想定して、既定で `MAX_MODEL_LEN=65536`, `GPU_MEMORY_UTILIZATION=0.92`, `MAX_NUM_SEQS=4`, `DTYPE=bfloat16` にしています。複数ファイルの生成時はvLLMの連続バッチ処理で最大4リクエストを並行処理します。

```bash
cd /mnt/kiso-qnap5/kitamura/projects/text-dialogue-qwen3
bash scripts/serve_vllm.sh
```

OOM する場合は短いコンテキストに下げます:

```bash
MAX_MODEL_LEN=32768 bash scripts/serve_vllm.sh
```

安定したら長くします。96GB ではまず `65536` で確認し、余裕があれば `131072`、さらに余裕があればネイティブ上限の `262144` を試してください。

```bash
MAX_MODEL_LEN=131072 bash scripts/serve_vllm.sh
```

別ポートで起動する例:

```bash
PORT=8001 bash scripts/serve_vllm.sh
```

GPU が見えているか確認:

```bash
nvidia-smi
```

## TSV 対話を書き換え生成

vLLM サーバ起動後:

```bash
UV_CACHE_DIR=/tmp/text-dialogue-qwen3-uv-cache UV_LINK_MODE=copy uv run --no-sync python -m qwen_dialogue \
  --backend openai \
  --input-file examples/input.tsv \
  --output-file outputs/64.generated.tsv
```

transformers で直接ロードする場合:

```bash
UV_CACHE_DIR=/tmp/text-dialogue-qwen3-uv-cache UV_LINK_MODE=copy uv run --no-sync --extra local python -m qwen_dialogue \
  --backend local \
  --input-file examples/input.tsv \
  --output-file outputs/64.generated.tsv
```

出力 TSV を軽く検査:

```bash
UV_CACHE_DIR=/tmp/text-dialogue-qwen3-uv-cache UV_LINK_MODE=copy uv run --no-sync python -m validate_dialogue_tsv outputs/64.generated.tsv
```

## 任意プロンプトで生成

```bash
UV_CACHE_DIR=/tmp/text-dialogue-qwen3-uv-cache UV_LINK_MODE=copy uv run --no-sync python -m qwen_dialogue \
  --backend openai \
  --prompt "USERとSYSTEMの自然な短い対話を10ターン生成してください。話題は名古屋旅行の買い物相談。"
```

## ランダム対話JSONを5個生成

旅行・お出かけを話題に、例のような自然で短いやりとりを5個生成して、1つのJSONファイルに保存します。

生成後は `normalize_dialogue_json.py` が、コードフェンス、分断された複数のJSON配列、1配列に連結された複数ターン、発話末尾の途切れ用ダッシュを機械的に正規化します。その後 `validate_dialogue_json.py` で検証します。対話内容自体の書き換えは行いません。

```bash
bash scripts/generate_random_travel_json.sh
```

出力先は既定で `outputs/random_travel_dialogues.json` です。別名にする場合:

```bash
OUTPUT_FILE=outputs/travel_001.json bash scripts/generate_random_travel_json.sh
```

Hugging Face から transformers で直接ロードして生成する場合は、先に local 依存を入れてから実行します。初回はモデル重みのダウンロードが走ります。

```bash
UV_CACHE_DIR=/tmp/text-dialogue-qwen3-uv-cache UV_LINK_MODE=copy uv sync --extra local
bash scripts/generate_random_travel_json_local.sh
```

手動で検査する場合:

```bash
UV_CACHE_DIR=/tmp/text-dialogue-qwen3-uv-cache UV_LINK_MODE=copy uv run --no-sync python -m validate_dialogue_json outputs/random_travel_dialogues.json
```

### 1,000ファイル、合計5,000対話を生成

HTTP APIは使いません。Transformersが `.hf-cache` のモデルを直接読み込み、モデルを一度だけGPUへロードしたまま1,000ファイルを生成します。初回のみCUDA 12.8用のローカル生成環境を用意します。

```bash
bash scripts/setup_local.sh
```

```bash
bash scripts/generate_random_travel_json_5000.sh
```

出力先は `outputs/random_travel_dialogues_5x1000/`、ファイル名は `random_travel_dialogues.0001.json` から `random_travel_dialogues.1000.json` です。各ファイルは5対話です。途中で止まって再実行した場合、検証済みの既存ファイルはスキップします。最初から上書きする場合は `RESUME=0` を指定します。

既定では同じプロンプトを4件まとめてGPUへ渡します。GPUメモリが不足する場合はバッチサイズを下げます。

```bash
BATCH_SIZE=2 bash scripts/generate_random_travel_json_5000.sh
```

## 主なオプション

- `--backend openai`: vLLM/SGLang などの OpenAI 互換 API に送ります。
- `--backend local`: transformers でモデルを直接ロードします。
- `--model`: 既定は `Qwen/Qwen3-30B-A3B-Instruct-2507`。
- `--prompt-file`: 既定は `prompts/rewrite_kyowa.txt`。
- `--input-file`: `{{DIALOGUE_TSV}}` に差し込む TSV。
- `--max-tokens`: 生成トークン数。既定は `4096`。
- `--temperature`: 既定は `0.4`。TSVを崩したくない場合は低めがおすすめです。
- `--base-url`: OpenAI 互換 API のURL。既定は `http://localhost:8000/v1`。
- `--json-mode`: OpenAI 互換 API が対応している場合、JSON出力を要求します。

## 注意

このリポジトリはモデル重みを含みません。初回実行時に Hugging Face からモデルがダウンロードされます。必要に応じて `HF_TOKEN` を設定してください。

NAS 上で `.venv/.lock` の permission warning が出る場合があるため、上の例では `UV_CACHE_DIR` と `UV_LINK_MODE=copy` を指定しています。warning が出ても `python -m ...` の実行が成功していれば問題ありません。

