# 新疆大学语音感知与智能计算实验室网站

展示新疆大学语音感知与智能计算实验室语音方向的项目、论文成果和 Demo 音频，内容主要围绕语音增强、语音合成、低资源语音处理等方向，支持项目筛选、项目卡片展示、论文列表展示以及单独的 Demo 详情页。

## 项目简介

网站由纯前端静态文件构成，不依赖后端服务，适合部署到 GitHub Pages、学校服务器、Nginx 静态目录或其他静态网站托管平台。

主要功能包括：

- 展示 B415 课题组简介与联系方式
- 按研究方向展示语音项目与 Demo
- 支持项目搜索与方向筛选
- 展示精选论文成果
- 提供项目 Demo 详情页与音频示例播放
- 支持跳转到外部论文、代码仓库或项目页面

## 目录结构

```text
.
├── index.html      # 网站首页
├── demo.html       # 项目 Demo 详情页
├── styles.css      # 页面样式
├── script.js       # 网站数据与交互逻辑
├── assets/         # 项目音频、图片和独立 Demo 页面等静态资源
├── CLAUDE.md       # Claude Code 项目规范
├── AGENTS.md       # 通用 Agent 项目规范
└── README.md       # 项目说明文档
```

## 本地预览

直接用浏览器打开 `index.html`。如果音频资源、相对路径或浏览器安全策略导致部分内容无法正常加载，建议使用本地静态服务器预览。

### 方法一：直接打开

在项目根目录中双击打开：

```text
index.html
```

### 方法二：使用 Python 启动本地服务

在项目根目录运行：

```bash
python3 -m http.server 8000
```

然后在浏览器访问：

```text
http://localhost:8000
```

如需查看某个项目 Demo 页面，可以访问：

```text
http://localhost:8000/demo.html?project=项目ID
```

其中 `项目ID` 来自 `script.js` 中对应项目的 `id` 字段。

## 页面内容维护

网站的大部分展示内容集中维护在 `script.js` 中，页面结构在 `index.html` 和 `demo.html` 中，视觉样式在 `styles.css` 中。

### 修改实验室信息

实验室名称、简介、办公室、邮箱和教师主页等信息维护在 `script.js` 顶部的 `labInfo` 对象中。

常见可修改字段包括：

- `name`：课题组名称
- `subtitle`：副标题
- `office`：办公室位置
- `emails`：联系邮箱
- `profile`：教师主页链接

修改后刷新首页即可查看效果。

### 添加或修改研究方向

研究方向信息维护在 `script.js` 中的 `directions` 数据结构中。

每个方向通常包含：

- `id`：方向唯一标识，用于筛选和锚点跳转
- `title`：中文方向名称
- `english`：英文方向名称
- `description`：方向简介
- `accent`：该方向使用的主题色
- `projects`：该方向下的项目列表

如果新增方向，需要同时确认首页方向筛选按钮是否也需要在 `index.html` 中增加对应入口。

### 添加或修改项目

项目内容主要维护在 `script.js` 的项目列表中。

一个项目通常包含以下信息：

- `id`：项目唯一标识，Demo 页面通过该字段查找项目
- `title`：项目标题
- `kind`：项目类型，例如语音增强、语音合成等
- `summary`：项目简介
- `tags`：项目标签
- `result`：项目亮点或结果说明
- `links`：外部链接或本地页面链接
- `samples`：Demo 音频示例

添加项目时请注意：

1. `id` 不要和已有项目重复。
2. 本地资源路径应相对于项目根目录填写，例如 `assets/example/audio.wav`。
3. 外部链接应使用完整 URL，例如 `https://example.com`。
4. 如果项目需要详情页展示，请确保项目数据中包含可渲染的 Demo 信息。

### 维护项目入口链接

项目卡片中的 `Demo`、`Paper`、`Code` 等入口由项目对象的 `links` 字段控制：

```js
links: [
  { label: "Paper", url: "https://doi.org/..." },
  { label: "Code", url: "https://github.com/..." },
  { label: "Demo", url: "assets/project/index.html" }
]
```

常见入口含义如下：

- `Demo`：项目展示页或本地 Demo 页面
- `Paper`：论文 DOI、IEEE、ACM、Springer 等官方页面
- `Code`：GitHub、GitLab 或其他代码仓库

`url` 以 `http` 开头时会在新窗口打开；本地相对路径会在当前页面打开。

### 添加或修改论文成果

论文成果维护在 `script.js` 的论文列表数据中。

常见字段包括：

- `year`：发表年份
- `title`：论文标题
- `venue`：会议、期刊或发表平台
- `authors`：作者信息
- `link`：论文链接，可选；建议使用 DOI 链接或出版社官方页面链接

添加论文后，首页“论文成果”区域会根据脚本渲染对应内容。若填写了 `link`，论文标题会变为可点击链接，并在新窗口打开。

### 添加 Demo 音频资源

Demo 音频建议放在 `assets/` 下对应项目目录中，例如：

```text
assets/项目名/audio/example.wav
```

在 `script.js` 的项目 `samples` 字段中引用该音频路径：

```js
{
  label: "示例名称",
  src: "assets/项目名/audio/example.wav"
}
```

建议使用浏览器兼容性较好的音频格式，例如 `.wav`、`.mp3` 或 `.ogg`。

## 部署说明

项目为静态网站，部署时只需要上传项目根目录下的 HTML、CSS、JS 和 `assets/` 资源即可。

### 部署到 GitHub Pages

1. 将项目提交到 GitHub 仓库。
2. 打开仓库的 `Settings`。
3. 进入 `Pages` 设置。
4. 选择部署分支，例如 `main`。
5. 选择根目录 `/` 作为发布目录。
6. 保存后等待 GitHub Pages 构建完成。

### 部署到服务器

如果部署到 Nginx、Apache 或学校服务器，只需要将以下文件和目录上传到网站根目录：

```text
index.html
demo.html
styles.css
script.js
assets/
```

确保服务器能正确访问音频、图片、HTML 和 JavaScript 文件。

## 注意事项

- 不要随意修改项目 `id`，否则已有的 `demo.html?project=项目ID` 链接可能失效。
- 添加外部链接时，请使用 HTML 或 JavaScript 中正确的 URL 格式，不要使用 Markdown 链接语法。
- 添加音频文件后，建议在本地服务器中测试播放效果。
- 资源文件路径区分大小写，部署到 Linux 服务器后大小写不一致可能导致文件无法加载。
- `.DS_Store` 是 macOS 自动生成的系统文件，通常不需要提交到仓库。
- 如果删除或移动 `assets/` 中的资源，需要同步更新 `script.js` 中对应路径。

## 许可证与引用说明

本网站用于课题组项目展示和学术 Demo 展示。部分 Demo 内容可能来自公开论文、开源仓库或合作项目，使用时请保留原作者、论文和代码仓库引用信息。

如页面中使用了第三方项目资源，请在项目卡片、Demo 页面或相关说明中注明来源。
