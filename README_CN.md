# 🔍 ReasoningLens

<div align="center">

### **Escape the "CoT Maze": Unmasking Model Reasoning at a Glance**

![reasoninglens-github](assets/reasoninglens-github.png)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-green.svg)](https://www.python.org/)
[![Node.js](https://img.shields.io/badge/node.js-22.10+-green.svg)](https://nodejs.org/)

[**🇬🇧 English**](README.md) | [**中文**](#-reasoninglens)

</div>

---

> **太长不看：** 长链推理（CoT）是一把双刃剑。虽然 OpenAI o1 和 DeepSeek-R1 等模型比以往更加智能，但调试一个 10,000 token 的推理轨迹简直是噩梦。**ReasoningLens** 将「文字墙」转化为交互式的层级结构图。

https://github.com/user-attachments/assets/f85a110f-d800-4a70-9f50-ffb16552987f


## 🤯 问题：当透明度成为负担

**大型推理模型（LRMs）** 的时代已经到来。我们喜欢它们自我纠正和规划的能力，但有一个问题：**理解模型*如何*得出结论变得越来越困难。**

当模型产生海量推理轨迹时，「关键」逻辑往往被淹没在重复的过程性步骤中。找到一个幻觉或逻辑转折点，就像大海捞针一样困难。


## 💡 ReasoningLens 简介

基于 **[Open WebUI](https://github.com/open-webui/open-webui.git)** 构建，ReasoningLens 是一个面向开发者的工具包，旨在帮助开源社区**可视化、理解和调试**模型推理链，而不会让人抓狂。

> **「ReasoningLens 不仅展示模型说了什么，更展示模型*如何思考*。」**

<div align="center">
<img src="assets/reasoninglens-framework.png" alt="ReasoningLens 框架" width="800"/>
</div>


## ✨ 核心功能

### 🗺️ 层级可视化：从混乱到清晰

大多数 CoT token 只是「执行」（进行计算），而只有少数是「策略性」的（决定改变方向）。ReasoningLens 将信号从噪声中分离出来：

- **规划单元分割：** 我们自动检测逻辑关键词，如 *「等等，让我重新检查...」* 或 *「或者...」*。
- **宏观视图（探索）：** 查看高层策略——模型在哪里回溯、在哪里验证、在哪里遇到困难。
- **微观视图（执行）：** 仅在需要时深入查看具体的算术或替换步骤。

<div align="center">
<img src="assets/reasoning-structure.png" alt="层级可视化" width="800"/>
</div>

### 🕵️ 自动错误检测：「智能体」审计员

更长的推理并不总是意味着更好的推理。「长度扩展」可能引入难以发现的幻觉。我们的 **SectionAnalysisAgent** 充当你的推理轨迹的专业审计员：

- **⚡ 批量分析：** 高效解析海量推理轨迹而不丢失上下文，使大规模调试成为可能。
-	**🧠 滚动摘要记忆：** 记住前序部分的上下文，能够捕捉到人工审阅者容易忽略的非局部不一致和逻辑漂移。
-	**🧮 工具增强验证：** 还在为模型连基础数学都算错而头疼吗？ReasoningLens 集成了计算器，可自动验证算术推理步骤。

<div align="center">
<img src="assets/automated-error-detection.png" alt="自动错误检测" width="800"/>
</div>

### 📊 模型画像：超越单次轨迹

单次调试很好，但**系统性模式**更重要。ReasoningLens 聚合多个对话的数据，为你的模型构建**推理画像**：

1. **聚合：** 跨不同领域（编程、数学、逻辑）收集轨迹。
2. **压缩：** 将重复模式提炼成紧凑的记忆状态。
3. **报告：** 生成结构化的 Markdown 报告，突出模型的「盲区」和「稳定优势」。

<div align="center">
<img src="assets/reasoning-profile.png" alt="模型画像" width="800"/>
</div>


## 🚀 快速开始

### 环境要求

- **Python**：版本 **3.11 或更高**（后端服务必需）
- **Node.js**：版本 **22.10 或更高**（前端开发必需）
- **Docker** 和 **Docker Compose**（容器化部署）


## 📦 安装方式

### 方式一：Conda 环境（开发模式）

#### 1. 克隆仓库

```bash
git clone https://github.com/icip-cas/reasoning-lens.git
cd reasoning-lens
```

#### 2. 后端配置

```bash
cd backend

# 创建并激活 conda 环境
conda create --name open-webui python=3.11
conda activate open-webui

# 安装依赖
pip install -r requirements.txt -U

# 启动后端服务
sh dev.sh
```

后端运行地址：`http://localhost:8080`

#### 3. 前端配置

打开新终端：

```bash
# 安装前端依赖
npm install --force

# 启动开发服务器
npm run dev
```

前端运行地址：`http://localhost:5173`


### 方式二：Docker Compose（推荐）

#### 快速启动

```bash
# 添加执行权限
chmod +x dev-docker.sh

# 启动开发环境
./dev-docker.sh
```

这将自动：

- 清理旧容器
- 创建必要的数据卷
- 启动前端和后端服务

**访问地址：**

- 🌐 前端：`http://localhost:5173`
- 🔧 后端：`http://localhost:8080`

#### Docker 常用命令

```bash
# 查看所有日志
docker-compose -f docker-compose.dev.yaml logs -f

# 仅查看后端日志
docker-compose -f docker-compose.dev.yaml logs -f backend

# 仅查看前端日志
docker-compose -f docker-compose.dev.yaml logs -f frontend

# 停止所有服务
docker-compose -f docker-compose.dev.yaml down

# 重启后端
docker-compose -f docker-compose.dev.yaml restart backend

# 重启前端
docker-compose -f docker-compose.dev.yaml restart frontend
```


### 方式三：Docker 构建（生产环境）

#### 构建 Docker 镜像

```bash
# 基础构建（仅 CPU）
docker build -t reasoning-lens:latest .

# 启用 CUDA 支持构建
docker build --build-arg USE_CUDA=true -t reasoning-lens:cuda .

# 集成 Ollama 构建
docker build --build-arg USE_OLLAMA=true -t reasoning-lens:ollama .

# 精简版构建（不预下载模型）
docker build --build-arg USE_SLIM=true -t reasoning-lens:slim .
```

#### 构建参数

| 参数                  | 默认值                                   | 说明                                      |
| --------------------- | ---------------------------------------- | ----------------------------------------- |
| `USE_CUDA`            | `false`                                  | 启用 CUDA/GPU 支持                        |
| `USE_CUDA_VER`        | `cu128`                                  | CUDA 版本（如 `cu117`、`cu121`、`cu128`） |
| `USE_OLLAMA`          | `false`                                  | 在镜像中包含 Ollama                       |
| `USE_SLIM`            | `false`                                  | 跳过预下载嵌入模型                        |
| `USE_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | RAG 的句子转换器模型                      |
| `USE_RERANKING_MODEL` | `""`                                     | RAG 的重排序模型                          |

#### 运行容器

```bash
# 运行容器
docker run -d \
  --name reasoning-lens \
  -p 8080:8080 \
  -v reasoning-lens-data:/app/backend/data \
  reasoning-lens:latest

# 使用 GPU 支持运行
docker run -d \
  --name reasoning-lens \
  --gpus all \
  -p 8080:8080 \
  -v reasoning-lens-data:/app/backend/data \
  reasoning-lens:cuda
```

#### 环境变量

| 变量                  | 说明                                  |
| --------------------- | ------------------------------------- |
| `OPENAI_API_KEY`      | 您的 OpenAI API 密钥                  |
| `OPENAI_API_BASE_URL` | 自定义 OpenAI 兼容 API 端点           |
| `WEBUI_SECRET_KEY`    | 会话管理的密钥                        |
| `DEFAULT_USER_ROLE`   | 新用户的默认角色（`user` 或 `admin`） |


## 🛠️ 开发指南

### 项目结构

```
reasoning-lens/
├── backend/                 # Python 后端 (FastAPI)
│   ├── open_webui/          # 主应用程序
│   │   ├── routers/         # API 路由
│   │   ├── models/          # 数据模型
│   │   └── utils/           # 工具函数
│   └── requirements.txt     # Python 依赖
├── src/                     # Svelte 前端
│   ├── lib/                 # 共享组件
│   └── routes/              # 页面路由
├── static/                  # 静态资源
├── Dockerfile               # 生产环境 Docker 构建文件
├── docker-compose.dev.yaml  # 开发环境 compose 文件
```

### 技术栈

- **后端**：Python 3.11+、FastAPI、SQLAlchemy
- **前端**：Svelte 5、TypeScript、TailwindCSS
- **数据库**：SQLite（默认）、PostgreSQL（可选）
- **容器化**：Docker、Docker Compose


## 📄 开源协议

本项目基于 MIT 协议开源 - 详见 [LICENSE](LICENSE) 文件。


## 📚 引用

如果 ReasoningLens 对你的研究有帮助，请考虑引用：

```bibtex
@software{Zhang_ReasoningLens_2026,
  author = {Zhang, Jun and Zheng, Jiasheng and Lu, Yaojie and Cao, Boxi},
  license = {MIT},
  month = feb,
  title = {{ReasoningLens}},
  url = {https://github.com/icip-cas/ReasoningLens},
  version = {0.1.0},
  year = {2026}
}
```


## 👥 团队与贡献者

- **Jun Zhang** - 主要贡献者
- **Jiasheng Zheng** - 贡献者
- **Yaojie Lu** - 贡献者
- **Boxi Cao** - 项目负责人

## 致谢

我们感谢 **[Open WebUI](https://github.com/open-webui/open-webui.git)** 社区以及所有早期用户和贡献者所提供的反馈与支持。我们期待开源社区持续的贡献。正是你们的时间与好奇心，让 ReasoningLens 变得更加出色。

## 💬 加入我们

有问题或想讨论想法？在 GitHub 上提交 Issue 或加入我们的社区讨论！让我们携手为社区设计更有效的工具. 🌟
