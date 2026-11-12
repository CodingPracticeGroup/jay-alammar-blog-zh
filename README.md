# jay-alammar-blog-zh

Jay Alammar（[jalammar.github.io](https://jalammar.github.io) / [Language Models & Co.](https://newsletter.languagemodels.co)）所著 “The Illustrated”（图解）系列技术博客的**简体中文翻译**静态站点。

- 在线地址（GitHub Pages，gh-pages 分支）：https://codingpracticegroup.github.io/jay-alammar-blog-zh/
- 首页为按原文发表时间倒序排列的文章列表；点击进入各篇简体中文译文。
- 图片、视频一律直接引用原站资源（`jalammar.github.io/images/...` 与 `substackcdn.com/...`），本仓库不存储图片。

## 已收录文章

| 原文发布日期 | 中文标题 | 英文原文 |
|---|---|---|
| 2025-08-19 | 图解 GPT-OSS | The Illustrated GPT-OSS |
| 2025-01-27 | 图解 DeepSeek-R1 | The Illustrated DeepSeek-R1 |
| 2022-10-04 | 图解 Stable Diffusion | The Illustrated Stable Diffusion |
| 2022-01-03 | 图解检索增强 Transformer（RETRO） | The Illustrated Retrieval Transformer |
| 2019-08-12 | 图解 GPT-2 | The Illustrated GPT-2 |
| 2019-03-27 | 图解 Word2vec | The Illustrated Word2vec |
| 2018-12-03 | 图解 BERT、ELMo 等 | The Illustrated BERT, ELMo, and co. |
| 2018-06-27 | 图解 Transformer | The Illustrated Transformer |

说明：原仓库中另有一篇 `The Illustrated Gradient Descent`（内题 *How Machines Learn*），其 front-matter 为 `published: False`，即作者从未在博客上发布（对应 URL 返回 404），故未收录。

## 目录结构

```
original/en/      英文原文（博客仓库 Markdown 副本；Substack 文章为 HTML 转换的 Markdown），供校对译文用
translation/zh/   简体中文译文 Markdown（站点的唯一内容来源，不带 front-matter）
tools/build.py    构建脚本：pandoc 转换 + 模板 + 生成首页
site/             构建产物（静态 HTML），部署时同步到 gh-pages 分支
```

## 本地构建

依赖 `pandoc`（≥ 2.x）：

```bash
python3 tools/build.py      # 生成 site/
python3 -m http.server -d site 8000   # 本地预览 http://localhost:8000
```

## 新增一篇文章

1. 把英文原文放入 `original/en/`，译文写入 `translation/zh/<slug>.zh.md`（正文即可，不要 front-matter；图片保持原文 URL，构建时会自动改写为原站绝对地址；文中 `/illustrated-*` 相对链接会被自动改写为本站译文页）。
2. 在 `tools/build.py` 的 `POSTS` 列表中登记该篇（slug、日期、中英文标题、原文链接）。
3. 重新构建并同步 gh-pages（见下）。

## 部署（GitHub Pages → gh-pages 分支）

```bash
python3 tools/build.py
git add -A && git commit -m "..."        # 提交 main（源码）
# 将 site/ 同步到 gh-pages 分支
rm -rf /tmp/gh && git worktree add /tmp/gh gh-pages 2>/dev/null || \
  git worktree add --orphan /tmp/gh gh-pages
rsync -a --delete --exclude .git site/ /tmp/gh/
git -C /tmp/gh add -A && git -C /tmp/gh commit -m "site update"
git push origin main gh-pages
git worktree remove /tmp/gh
```

然后在 GitHub 仓库 Settings → Pages 选择 `gh-pages` 分支、根目录作为发布来源（仅首次需要）。

## 许可与出处

- 原文版权归 **Jay Alammar** 所有；其博客在每篇文章页脚声明全文以 [CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/) 发布。
- 本仓库的译文按 **CC BY-NC-SA 4.0**（署名—非商业性使用—相同方式共享）发布：须署名并链接原文、不得用于商业目的、衍生作品须采用相同许可。译文中的翻译差错由本翻译项目负责，并非作者原意。
- `tools/` 下的构建脚本为本项目辅助代码，按 MIT 许可发布。
- 本仓库与 Jay Alammar 本人无隶属关系。
