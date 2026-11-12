#!/usr/bin/env python3
"""构建 jay-alammar-blog-zh 静态站点。

用法：
    python3 tools/build.py            # 生成 site/ 下的静态页面

依赖：pandoc（用于 Markdown -> HTML 转换）。

文章元数据集中在下方 POSTS 列表中；新增翻译时：
1. 在 translation/zh/ 放入 <slug>.zh.md（不带 front-matter 的正文）；
2. 在 POSTS 中登记一条记录；
3. 重新运行本脚本。
"""
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZH_DIR = os.path.join(ROOT, "translation", "zh")
SITE_DIR = os.path.join(ROOT, "site")
POSTS_DIR = os.path.join(SITE_DIR, "posts")

# 站点元信息
SITE_TITLE = "Jay Alammar 博客 · 简体中文翻译"
JALAMMAR_BASE = "https://jalammar.github.io"

# 文章登记表（首页按 date 倒序自动生成）
POSTS = [
    {
        "slug": "illustrated-gpt-oss",
        "date": "2025-08-19",
        "title_en": "The Illustrated GPT-OSS",
        "title_zh": "图解 GPT-OSS",
        "original_url": "https://newsletter.languagemodels.co/p/the-illustrated-gpt-oss",
        "source": "Substack「Language Models & Co.」",
    },
    {
        "slug": "illustrated-deepseek-r1",
        "date": "2025-01-27",
        "title_en": "The Illustrated DeepSeek-R1",
        "title_zh": "图解 DeepSeek-R1",
        "original_url": "https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1",
        "source": "Substack「Language Models & Co.」",
    },
    {
        "slug": "illustrated-stable-diffusion",
        "date": "2022-10-04",
        "title_en": "The Illustrated Stable Diffusion",
        "title_zh": "图解 Stable Diffusion",
        "original_url": JALAMMAR_BASE + "/illustrated-stable-diffusion/",
        "source": "jalammar.github.io",
    },
    {
        "slug": "illustrated-retrieval-transformer",
        "date": "2022-01-03",
        "title_en": "The Illustrated Retrieval Transformer",
        "title_zh": "图解检索增强 Transformer（RETRO）",
        "original_url": JALAMMAR_BASE + "/illustrated-retrieval-transformer/",
        "source": "jalammar.github.io",
    },
    {
        "slug": "illustrated-gpt2",
        "date": "2019-08-12",
        "title_en": "The Illustrated GPT-2 (Visualizing Transformer Language Models)",
        "title_zh": "图解 GPT-2（可视化 Transformer 语言模型）",
        "original_url": JALAMMAR_BASE + "/illustrated-gpt2/",
        "source": "jalammar.github.io",
    },
    {
        "slug": "illustrated-word2vec",
        "date": "2019-03-27",
        "title_en": "The Illustrated Word2vec",
        "title_zh": "图解 Word2vec",
        "original_url": JALAMMAR_BASE + "/illustrated-word2vec/",
        "source": "jalammar.github.io",
    },
    {
        "slug": "illustrated-bert",
        "date": "2018-12-03",
        "title_en": "The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)",
        "title_zh": "图解 BERT、ELMo 等（NLP 如何攻克迁移学习）",
        "original_url": JALAMMAR_BASE + "/illustrated-bert/",
        "source": "jalammar.github.io",
    },
    {
        "slug": "illustrated-transformer",
        "date": "2018-06-27",
        "title_en": "The Illustrated Transformer",
        "title_zh": "图解 Transformer",
        "original_url": JALAMMAR_BASE + "/illustrated-transformer/",
        "source": "jalammar.github.io",
    },
]

CC_BADGE = ('<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">'
            '<img alt="Creative Commons License" style="border-width:0" '
            'src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>')

LICENSE_FOOTER = f"""
<footer class="site-footer">
  <p>本站为 Jay Alammar 博客文章的简体中文翻译合集，非官方站点，不用于商业用途。<br />
  翻译与译文按 <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a> 许可发布；原文版权归 Jay Alammar 所有，同样依 CC BY-NC-SA 4.0 发布。</p>
  <p class="badge">{CC_BADGE}</p>
</footer>
"""

STYLE = """
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --fg:#1f2328; --muted:#656d76; --accent:#0b6bcb; --bg:#ffffff; --box:#f6f8fa; --border:#d8dee4; }
  @media (prefers-color-scheme: dark) {
    :root { --fg:#e6edf3; --muted:#9198a1; --accent:#6cb2f0; --bg:#0d1117; --box:#161b22; --border:#30363d; }
  }
  * { box-sizing: border-box; }
  body { margin:0; padding:0; background:var(--bg); color:var(--fg);
         font: 16px/1.75 -apple-system,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif; }
  .wrap { max-width: 46em; margin: 0 auto; padding: 1.5rem 1.2rem 4rem; }
  header.site-header { border-bottom:1px solid var(--border); padding-bottom:.8rem; margin-bottom:1.5rem; }
  header.site-header a.brand { color:var(--fg); text-decoration:none; font-weight:700; font-size:1.15rem; }
  header.site-header .sub { color:var(--muted); font-size:.85rem; }
  h1 { line-height:1.35; font-size:1.7rem; }
  h2 { margin-top:2.2em; border-bottom:1px solid var(--border); padding-bottom:.2em; font-size:1.3rem; }
  h3 { margin-top:1.8em; font-size:1.1rem; }
  a { color: var(--accent); }
  img, video { max-width:100%; height:auto; }
  .img-div, .img-div-any-width, .img-div-any-size, .cite { text-align:center; margin:1.2em 0; }
  .img-div img, .img-div-any-width img, .img-div-any-size img { max-width: 90%; }
  figcaption, .img-div br + *, .img-div-any-width br + * { color:var(--muted); }
  blockquote.subtle, blockquote { border-left:4px solid var(--border); margin:1em 0; padding:.3em 1em; color:var(--muted); }
  pre { background:var(--box); border:1px solid var(--border); border-radius:8px; padding:1em; overflow-x:auto; }
  code { background:var(--box); border-radius:4px; padding:.1em .3em; font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:.9em; }
  pre code { background:none; padding:0; }
  table { border-collapse:collapse; margin:1em 0; }
  td, th { border:1px solid var(--border); padding:.4em .7em; }
  .post-list { list-style:none; padding:0; }
  .post-list li { padding:1rem 0; border-bottom:1px solid var(--border); }
  .post-list .date { color:var(--muted); font-size:.85rem; }
  .post-list a.title { font-size:1.15rem; font-weight:600; text-decoration:none; }
  .post-list a.title:hover { text-decoration:underline; }
  .post-list .orig { color:var(--muted); font-size:.85rem; }
  .notice { background:var(--box); border:1px solid var(--border); border-radius:8px; padding:.8em 1.1em; font-size:.92rem; color:var(--muted); }
  .notice a { color:var(--muted); }
  .post-meta { color:var(--muted); font-size:.9rem; margin-bottom:1em; }
  .site-footer { margin-top:4rem; padding-top:1.2rem; border-top:1px solid var(--border); color:var(--muted); font-size:.85rem; }
  .site-footer .badge img { display:inline-block; }
  .backlink { font-size:.9rem; }
</style>
"""


def page(title: str, body_html: str, canonical: str = "") -> str:
    """组装整页 HTML（相对资源均用相对路径，便于部署到项目子路径）。"""
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
{STYLE}<title>{title}</title>
</head>
<body>
<div class="wrap">
{body_html}
</div>
</body>
</html>
"""


def site_header(active: bool = True) -> str:
    home = "./index.html" if not active else "index.html"
    return f"""<header class="site-header">
  <a class="brand" href="{home}">Jay Alammar 博客 · 简体中文翻译</a>
  <div class="sub">原文作者 Jay Alammar（jalammar.github.io / languagemodels.co）· 译文按 CC BY-NC-SA 4.0 共享</div>
</header>"""


def pandoc(md_path: str) -> str:
    out = subprocess.run(
        ["pandoc", "-f", "gfm+tex_math_dollars", "-t", "html", "--wrap=none", md_path],
        capture_output=True, text=True, check=True)
    return out.stdout


def fix_links(html: str, slug: str) -> str:
    # 相对图片/视频 -> 原站绝对地址
    html = html.replace('src="/images/', f'src="{JALAMMAR_BASE}/images/')
    html = html.replace('href="/images/', f'href="{JALAMMAR_BASE}/images/')
    html = html.replace('src="/videos/', f'src="{JALAMMAR_BASE}/videos/')
    # 指向本站已翻译文章的旧站绝对链接 -> 本地译文页
    for p in POSTS:
        pat = re.compile(r'href="(?:' + re.escape(JALAMMAR_BASE) + r')?/(' + re.escape(p["slug"]) + r')/"')
        html = pat.sub(lambda m: f'href="{m.group(1)}.html"' if m.group(1) != slug
                       else f'href="#{m.group(1)}"', html)
    # 其余站内相对链接 -> 原站绝对地址
    html = re.sub(r'href="/(?!/|#)([^"]*)"', rf'href="{JALAMMAR_BASE}/\1"', html)
    return html


def build_post(p: dict) -> None:
    md_path = os.path.join(ZH_DIR, p["slug"] + ".zh.md")
    body = pandoc(md_path)
    body = fix_links(body, p["slug"])
    notice = f"""
<div class="notice">
  本文由社区翻译，原文 <a href="{p["original_url"]}">{p["title_en"]}</a> 作者是
  <a href="https://jalammar.github.io">Jay Alammar</a>（来源：{p["source"]}）。
  原文与本文译文均按 <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a> 许可发布；
  译文力求忠实，如有出入以原文为准，翻译错误由本站负责。
</div>"""
    meta = f"""
<h1>{p["title_zh"]}</h1>
<p class="post-meta">原文：{p["title_en"]} · {p["date"]} ·
  <a href="{p["original_url"]}">阅读原文</a></p>"""
    back = '<p class="backlink"><a href="../index.html">← 返回文章列表</a></p>'
    html = page(p["title_zh"] + " | Jay Alammar 博客中文翻译",
                site_header(active=False) + back + meta + notice +
                '<article class="post-body">' + body + "</article>" + LICENSE_FOOTER)
    out = os.path.join(POSTS_DIR, p["slug"] + ".html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print("  built posts/%s.html" % p["slug"])


def build_index() -> None:
    posts = sorted(POSTS, key=lambda p: p["date"], reverse=True)
    items = "\n".join(
        f'<li><div class="date">{p["date"]}</div>'
        f'<a class="title" href="posts/{p["slug"]}.html">{p["title_zh"]}</a><br />'
        f'<span class="orig">原文：{p["title_en"]}（{p["source"]} · '
        f'<a href="{p["original_url"]}">English</a>）</span></li>'
        for p in posts)
    body = site_header() + f"""
<h1>文章列表</h1>
<p>这里是 <a href="https://jalammar.github.io">Jay Alammar</a> 所著 “The Illustrated”（图解）系列技术博客的简体中文翻译，按原文发表时间倒序排列。点击标题阅读简体中文译文；图片沿用原站资源。</p>
<ul class="post-list">
{items}
</ul>
<p class="notice">本站是非官方、非商业的社区翻译项目。译文按 <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a> 许可共享——署名、非商业性使用、相同方式共享。</p>
""" + LICENSE_FOOTER
    with open(os.path.join(SITE_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(page(SITE_TITLE, body))
    print("  built index.html")


def main() -> None:
    if shutil.which("pandoc") is None:
        sys.exit("需要 pandoc，请先安装。")
    if os.path.isdir(SITE_DIR):
        shutil.rmtree(SITE_DIR)
    os.makedirs(POSTS_DIR)
    build_index()
    for p in POSTS:
        build_post(p)
    print("完成：site/ 目录已生成（index.html + %d 个文章页）" % len(POSTS))


if __name__ == "__main__":
    main()
