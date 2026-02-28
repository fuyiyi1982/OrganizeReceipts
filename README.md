# Toll Organizer Web

一个可部署在 Ubuntu 的 Web 项目：上传“通行费电子发票及详情”压缩包后，系统会自动解析并按 `年/月/日/行程` 分类保存，同时支持按日打包 ZIP 下载。

## 功能

- 上传一个或多个顶层 ZIP 文件
- 自动识别行程段目录（`apply.zip + detail.zip`）
- 从 `apply.zip` 内提取全部发票 PDF 与 XML
- 从 `detail.zip` 提取 `trans.pdf` 作为行程单
- 自动按 `storage/YYYY/MM/DD/trip_xxx` 存储
- Web 页面展示每日段次、金额、文件列表
- 提供“按日打包 ZIP”下载接口

## 目录结构

```text
toll-organizer-web/
  app/
    main.py
    templates/
    static/
  data/
    storage/   # 运行后生成，按年/月/日/行程保存
    tmp/       # 临时目录
  requirements.txt
  Dockerfile
  docker-compose.yml
```

## 本地运行

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

浏览器访问：`http://<服务器IP>:8000`

## Ubuntu 部署（systemd）

1. 克隆项目

```bash
git clone <your-repo-url> toll-organizer-web
cd toll-organizer-web
```

2. 安装依赖并启动

```bash
sudo apt update
sudo apt install -y python3 python3-venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

3. 可选：使用 Nginx 反向代理到 8000 端口。

## Docker 部署

```bash
docker compose up -d --build
```

访问：`http://<服务器IP>:8000`

## 数据规则

- 每段行程目录保存：
  - `itinerary.pdf`
  - `invoice_001.pdf`, `invoice_002.pdf`, ...
  - `metadata.json`
- 金额汇总来自 XML 含税金额字段：
  - `TotalTax-includedAmount`
  - 兼容 `TotaltaxIncludedAmount`

## 典型使用流程

1. 打开首页上传多个当月 ZIP 文件
2. 系统自动完成分类
3. 在“已整理日期”列表里查看某天明细
4. 点击“按日下载ZIP”获取报销打包文件
