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

## 安装 llama.cpp（提供 llama-server 二进制）

embedding 自托管服务与可选的本地聊天 LLM 都依赖 `llama.cpp` 的 `llama-server` 二进制。
**新用户部署时请先安装 llama.cpp**，否则 `serve_embeddings.py` 与本地 LLM 会报「找不到 llama-server」。

**1. 下载预编译二进制**
- Windows：到 [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases) 下载含 `llama-server` 的包
  （如 `llama-server-bin-win-cuda-cu12.x` 带 GPU 加速，或 CPU 版），解压得到 `llama-server.exe`。
- Linux / macOS：`brew install llama.cpp`，或下载 release 中的 `llama-server`，或 `cmake` 自行编译。

**2. 让 V5 能找到它（二选一）**
- 把 `llama-server`（或 `llama-server.exe`）所在目录加入系统 `PATH`；
- 或设置环境变量 `IKAROS_LLAMA_SERVER` 指向完整路径：
  - Windows：`set IKAROS_LLAMA_SERVER=C:\tools\llama.cpp\llama-server.exe`
  - Linux/macOS：`export IKAROS_LLAMA_SERVER=/usr/local/bin/llama-server`

定位顺序（未设置 `IKAROS_LLAMA_SERVER` 时）：`PATH` 中的 `llama-server` → 仓库内
`runtime/llama/b10000-cuda/llama-server.exe`（新仓库通常没有该路径，忽略即可）。

**3. 启动 embedding 服务（语义检索需要）**
```bash
python models/download_models.py --only embed   # 取得 nomic-embed-text-v2-moe.f16.gguf
python models/serve_embeddings.py                # 监听 127.0.0.1:8587/embedding
```
然后把 `IKAROS_EMBED_URL` 指向 `http://127.0.0.1:8587/embedding`（install.py 已默认设好）。
