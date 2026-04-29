#!/usr/bin/env bash
set -euo pipefail

# 基于脚本位置定位项目根目录
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# 使用 uv 安装 Python 依赖（符合平台规范）
if command -v uv &>/dev/null; then
    uv pip install --system -r requirements.txt
else
    pip install -r requirements.txt
fi

echo "Dependencies installed successfully"
