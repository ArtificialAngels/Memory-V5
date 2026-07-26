# models/

模型目录。V5 的本地 LLM（聊天 / 认知任务）与向量语义检索的 embedding 模型都放在这里。

## 目录内容

- `model_config.py` — 本地 LLM 的加载配置（扫描目录、自动选定初始模型、生成 llama-server 参数）。
- `model_config.json` — 由 `model_config.py` 首跑自动生成 / 可手动修改的当前模型选择。
- `download_models.py` — 从 HuggingFace 下载所需 GGUF（聊天 + embedding）。
- `serve_embeddings.py` — 用 llama-server 自托管 nomic embedding 服务（:8587）。

> **注意**：`.gguf` 模型文件**不入库**（体积数 GB）。请运行 `python download_models.py` 下载，
> 或把已有的 GGUF 直接放进本目录。

## 聊天模型（:8080）

默认用 `Qwen3-1.7B-Q4_K_M.gguf`。放好之后 `model_config.py` 会自动识别并写入 `model_config.json`；
也可手动改 `model_config.json` 的 `initial_model` 切换。V5 只把**非 embedding** 的 `.gguf` 当作聊天模型
（按文件名关键字 embed/nomic/e5/bge… 排除）。

## Embedding 模型（:8587）

语义检索需要 embedding 端点。两种用法任选其一：

1. **外部服务**：设环境变量 `IKAROS_EMBED_URL=http://你的端点/embedding`（OpenAI / Ollama / 远程 llama-server 均可）。
2. **自托管**：`python serve_embeddings.py`（需先 `python download_models.py --only embed`）。
   启动后监听 `127.0.0.1:8587/embedding`，把 `IKAROS_EMBED_URL` 指向它即可。

## llama-server 二进制

V5 通过 `IKAROS_LLAMA_SERVER` 环境变量定位 `llama-server` 二进制；未设置时按顺序尝试：
`PATH` 中的 `llama-server` → 仓库内 `runtime/llama/b10000-cuda/llama-server.exe`。
