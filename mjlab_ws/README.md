# 如何新增一个项目并加入 workspace

本目录是一个 **uv workspace**：`mjlab`（上游库）和 `cartpole`（自己的项目）共用一份依赖锁和同一个 `.venv`。所有成员都以 editable 方式装进环境，因此任意成员里都可以直接 `import mjlab`，不需要设置 `PYTHONPATH`。

## 目录结构

```text
mjlab_ws/
├── pyproject.toml     # workspace 根（package = false，只做容器）
├── uv.lock            # 整个 workspace 共用的锁文件
├── .venv/             # 唯一的虚拟环境（当前 Python 3.13）
├── mjlab/             # 上游 mjlab clone，保持干净，不要往里加自己的代码
└── cartpole/          # 自己的项目（示例）
    ├── pyproject.toml
    └── src/cartpole/
        ├── __init__.py
        ├── asset_zoo/...
        └── tasks/...
```
## 核心步骤

假设新项目叫 `myproj`，放在 `mjlab_ws/myproj/`。

### 1. 建目录

```text
mjlab_ws/myproj/
├── pyproject.toml
└── src/
    └── myproj/
        └── __init__.py # 必须有这个py
```

`uv_build` 默认要求包目录是 `src/<模块名>/`，模块名默认由项目名转换而来（`my-proj` → `my_proj`）。目录不匹配时 `uv sync` 会报：

```text
Expected a Python module at: myproj/src/myproj/__init__.py
```

### 2. 写 `myproj/pyproject.toml`

```toml
[build-system]
requires = ["uv_build>=0.12.1,<0.13.0"]
build-backend = "uv_build"

[project]
name = "myproj"
version = "0.1.0"
requires-python = ">=3.10,<3.15"
dependencies = ["mjlab"]

[project.entry-points."mjlab.tasks"]
myproj = "myproj.tasks"
```

### 3. 把项目注册到 workspace 根 [pyproject.toml](pyproject.toml)

三处都要加：

```toml
[project]
dependencies = [
  "mjlab[cu130]",
  "cartpole",
  "myproj",                                    # ① 作为根项目的依赖
]

[tool.uv.workspace]
members = ["mjlab", "cartpole", "myproj"]      # ② 声明为 workspace 成员

[tool.uv.sources]
mjlab = { workspace = true }
cartpole = { workspace = true }
myproj = { workspace = true }                  # ③ 使用本地目录，而不是去 PyPI 下载
```

### 4. 同步并验证

```bash
cd /home/u2024210044/RL-Learn-Hub/mjlab_ws
uv sync
uv run python -c "import myproj; print(myproj.__file__)"
```

输出应为 `.../mjlab_ws/myproj/src/myproj/__init__.py`。看到这个就说明已经"加入路径"，之后在任何成员里直接 `import myproj` / `import mjlab` 即可。

### 5. 训练与测试
在创建完模型和任务后可运行该条查看是否创建成功
```
uv run list-envs
```
**train**
```
uv run python -m mjlab.scripts.train Mjlab-Cartpole --agent.max-iterations 5000 --log-root <save-path>
```
**play**
```
uv run python -m mjlab.scripts.play Mjlab-Cartpole --checkpoint_file <path> --num_envs 4 --viewer viser
``` 

### 6. vscode显示问题
写入./vscode/settings.json
```
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.analysis.extraPaths": [
    "${workspaceFolder}/mjlab/src",
    "${workspaceFolder}/cartpole/src"
  ]
}
```
再选择.venv的python解释器，如果选不了再在settings.json加上
```
"python.useEnvironmentsExtension": false
```