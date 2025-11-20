#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aozora Bunko downloader (balanced random + sidecar)
- 隨機下載 N 本（每位作者至多 per_author 本，預設 1）
- 以作者關鍵字下載
- 以作者 + 標題關鍵字下載
- 直接指定作品卡頁 URL 下載
- 優先抓「テキストファイル(ルビあり)」zip，退回任何 zip
- 自動處理 Shift-JIS/CP932，輸出 UTF-8 .txt
- 另存 *.meta.json（title/author/card_url）方便 RAG 顯示

用法：
  python aozora_downloader.py 3
  python aozora_downloader.py 5 --author 夏目漱石
  python aozora_downloader.py 1 --author 夏目漱石 --title 吾輩は猫
  python aozora_downloader.py 1 --card https://www.aozora.gr.jp/cards/000148/cardXXXX.html
  python aozora_downloader.py 50 --per-author 1
"""

import argparse
import io
import os
import random
import re
import unicodedata
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

INDEX_ALL = "https://www.aozora.gr.jp/index_pages/person_all.html"
INDEX_FALLBACKS = [
    "https://www.aozora.gr.jp/index_pages/person_a.html",
    "https://www.aozora.gr.jp/index_pages/person_ka.html",
    "https://www.aozora.gr.jp/index_pages/person_sa.html",
    "https://www.aozora.gr.jp/index_pages/person_ta.html",
    "https://www.aozora.gr.jp/index_pages/person_na.html",
    "https://www.aozora.gr.jp/index_pages/person_ha.html",
    "https://www.aozora.gr.jp/index_pages/person_ma.html",
    "https://www.aozora.gr.jp/index_pages/person_ya.html",
    "https://www.aozora.gr.jp/index_pages/person_ra.html",
    "https://www.aozora.gr.jp/index_pages/person_wa.html",
]
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X) AozoraScraper/1.2"

def _get_soup(url: str, session: requests.Session | None = None) -> BeautifulSoup:
    sess = session or requests.Session()
    r = sess.get(url, timeout=25, headers={"User-Agent": UA})
    ct = r.headers.get("Content-Type", "").lower()
    head = r.content[:800].lower()
    if "shift_jis" in ct or b"shift_jis" in head or b"x-sjis" in head:
        r.encoding = "cp932"
    else:
        r.encoding = r.apparent_encoding or "utf-8"
    return BeautifulSoup(r.text, "html.parser")

_norm_space = re.compile(r"[\s\u3000・･]+")
def jnorm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = _norm_space.sub("", s)
    return s

def sanitize_filename(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).strip()
    s = re.sub(r"[\\/:*?\"<>|]", "_", s)
    s = re.sub(r"\s+", "_", s)
    return s or "untitled"

def get_all_author_pages() -> list[dict]:
    sess = requests.Session()
    authors: list[dict] = []
    soup = _get_soup(INDEX_ALL, session=sess)
    links = soup.select("ol li a[href^='person']")
    if not links:
        print("ℹ️ person_all 無內容，改用後備假名子頁清單。")
        for u in INDEX_FALLBACKS:
            s2 = _get_soup(u, session=sess)
            links += s2.select("ol li a[href^='person']")
    for a in links:
        href = a.get("href")
        name = a.get_text(strip=True)
        if not href or not name:
            continue
        authors.append({"name": name, "url": urljoin(INDEX_ALL, href)})
    # 去重
    seen, dedup = set(), []
    for x in authors:
        if x["url"] in seen:
            continue
        seen.add(x["url"])
        dedup.append(x)
    return dedup

def find_authors_by_keyword(keyword: str) -> list[dict]:
    key = jnorm(keyword)
    return [row for row in get_all_author_pages() if key in jnorm(row["name"])]

def get_author_card_pages(author_page_url: str) -> list[dict]:
    soup = _get_soup(author_page_url)
    out: list[dict] = []
    # 公開中の作品
    anchor = soup.select_one("a[name='sakuhin_list_1']")
    if anchor:
        ol = anchor.find_next("ol")
        if ol:
            for a in ol.select("a[href*='cards/'][href*='card']"):
                out.append({"title": a.get_text(strip=True), "url": urljoin(author_page_url, a.get("href"))})
    if not out:
        for a in soup.select("a[href*='cards/'][href*='card']"):
            out.append({"title": a.get_text(strip=True), "url": urljoin(author_page_url, a.get("href"))})
    # 去重
    seen, dedup = set(), []
    for x in out:
        if x["url"] in seen:
            continue
        seen.add(x["url"])
        dedup.append(x)
    return dedup

def get_card_page_title(card_url: str) -> str:
    soup = _get_soup(card_url)
    text = soup.get_text("\n", strip=True)
    m = re.search(r"作[　\s]*品[　\s]*名[:：]\s*([^\n\r]+)", text)
    if m:
        return m.group(1).strip()
    for tag in ["h2", "h3"]:
        h = soup.find(tag)
        if h and h.get_text(strip=True) and not re.search(r"図書カード", h.get_text()):
            return h.get_text(strip=True)
    t = soup.find("title")
    if t:
        return re.sub(r"図書カード[:：]?\s*", "", t.get_text(strip=True))
    return "work"

def get_card_page_author(card_url: str) -> str:
    soup = _get_soup(card_url)
    a = soup.select_one("a[href*='index_pages/person']")
    if a and a.get_text(strip=True):
        return a.get_text(strip=True)
    lab = soup.find(string=re.compile(r"著者名|作者名"))
    if lab and lab.parent:
        txt = lab.parent.get_text(" ", strip=True)
        m = re.search(r"(著者名|作者名)[:：]\s*([^\s　]+)", txt)
        if m:
            return m.group(2)
    return ""

def find_zip_from_card(card_url: str) -> tuple[str | None, str]:
    soup = _get_soup(card_url)
    page_title = get_card_page_title(card_url)
    cand = soup.select("a[href$='.zip']")
    ruby_first, any_zip = [], []
    for a in cand:
        href = a.get("href", "")
        text = a.get_text(" ", strip=True)
        if re.search(r"ruby|ルビ", href, re.I) or re.search(r"ルビ", text):
            ruby_first.append(urljoin(card_url, href))
        else:
            any_zip.append(urljoin(card_url, href))
    if ruby_first:
        return ruby_first[0], page_title
    if any_zip:
        return any_zip[0], page_title
    return None, page_title

# 替換原本的 download_and_extract_text，加上簡單重試與退避
def download_and_extract_text(zip_url: str, out_dir: str, base_title: str, max_retries: int = 3) -> list[str]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.get(zip_url, timeout=40, headers={"User-Agent": UA})
            r.raise_for_status()
            saved = []
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                for info in zf.infolist():
                    if not info.filename.lower().endswith(".txt"):
                        continue
                    data = zf.read(info.filename)
                    try:
                        text = data.decode("cp932")
                    except UnicodeDecodeError:
                        text = data.decode("shift_jis", errors="ignore")
                    inner_name = os.path.basename(info.filename)
                    stem = os.path.splitext(inner_name)[0]
                    fn = f"{sanitize_filename(stem)}.txt"
                    if len(stem) < 2 or re.fullmatch(r"\d+_ruby_\d+", stem):
                        fn = f"{sanitize_filename(base_title)}.txt"
                    out_path = Path(out_dir) / fn
                    out_path.write_text(text, encoding="utf-8")
                    saved.append(str(out_path))
            return saved
        except Exception as e:
            last_err = e
            # 指數退避（0.8~1.4 隨機抖動）
            import time, random
            time.sleep((0.8 + 0.6 * random.random()) * (2 ** attempt))
    # 重試用盡，丟回讓上層打印「⚠️ 失敗」
    raise last_err


def write_sidecar(txt_path: Path, title: str, author: str, card_url: str):
    meta = {"title": title, "author": author, "card_url": card_url}
    txt_path.with_suffix(".meta.json").write_text(
        __import__("json").dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def download_by_card(card_url: str, out_dir: str) -> int:
    zip_url, title = find_zip_from_card(card_url)
    if not zip_url:
        print("❌ 這個卡頁沒有可下載的 zip。")
        return 0
    print("🔗 Fetching", card_url)
    try:
        saved = download_and_extract_text(zip_url, out_dir, base_title=title)
        if saved:
            author = get_card_page_author(card_url) or ""
            for p in saved:
                write_sidecar(Path(p), title, author, card_url)
            print(f"✅ Saved: {os.path.basename(saved[0])}")
            return 1
    except Exception as e:
        print("⚠️ 失敗：", e)
    return 0

def download_by_author(keyword: str, count: int, out_dir: str, title_kw: str | None = None) -> int:
    matches = find_authors_by_keyword(keyword)
    if not matches:
        print(f"❌ 找不到符合作者關鍵字：{keyword}")
        return 0
    author = matches[0]
    print(f"👤 命中作者：{author['name']} → {author['url']}")

    cards = get_author_card_pages(author["url"])
    if not cards:
        print("❌ 找不到公開中的作品。")
        return 0
    if title_kw:
        key = jnorm(title_kw)
        filtered = [c for c in cards if key in jnorm(c["title"]) or key in jnorm(get_card_page_title(c["url"]))]
        if not filtered:
            print(f"❌ 在此作者下找不到標題包含「{title_kw}」的作品。")
            return 0
        cards = filtered

    downloaded = 0
    for c in cards:
        if downloaded >= count:
            break
        zip_url, title = find_zip_from_card(c["url"])
        if not zip_url:
            continue
        print("🔗 Fetching", c["url"])
        try:
            saved = download_and_extract_text(zip_url, out_dir, base_title=title)
            if saved:
                for p in saved:
                    write_sidecar(Path(p), title, author["name"], c["url"])
                print(f"✅ Saved: {os.path.basename(saved[0])}")
                downloaded += 1
        except Exception as e:
            print("⚠️ 失敗：", e)
    print(f"🎉 完成，共下載 {downloaded} 本。")
    return downloaded

def download_random(count: int, out_dir: str, per_author: int = 1) -> int:
    """隨機多作者下載：每位作者至多 per_author 本（預設 1）"""
    authors = get_all_author_pages()
    if not authors:
        print("❌ 取得作者清單失敗。")
        return 0
    random.shuffle(authors)
    downloaded = 0
    for a in authors:
        if downloaded >= count:
            break
        cards = get_author_card_pages(a["url"])
        if not cards:
            continue
        random.shuffle(cards)
        taken = 0
        for c in cards:
            if downloaded >= count or taken >= max(1, per_author):
                break
            zip_url, title = find_zip_from_card(c["url"])
            if not zip_url:
                continue
            print(f"👤 {a['name']} → {c['url']}")
            try:
                saved = download_and_extract_text(zip_url, out_dir, base_title=title)
                if saved:
                    for p in saved:
                        write_sidecar(Path(p), title, a["name"], c["url"])
                    print(f"✅ Saved: {os.path.basename(saved[0])}")
                    downloaded += 1
                    taken += 1
            except Exception as e:
                print("⚠️ 失敗：", e)
    print(f"🎉 完成，共下載 {downloaded} 本。")
    return downloaded

def main():
    ap = argparse.ArgumentParser(description="Aozora Bunko downloader")
    ap.add_argument("count", type=int, help="要下載的本數")
    ap.add_argument("--author", type=str, default=None, help="作者關鍵字（例：夏目漱石）")
    ap.add_argument("--title", type=str, default=None, help="（搭配 --author）作品標題關鍵字")
    ap.add_argument("--card", type=str, default=None, help="直接指定作品卡頁 URL")
    ap.add_argument("--out", type=str, default="data", help="輸出資料夾（預設 data）")
    ap.add_argument("--per-author", type=int, default=1, help="隨機模式：每位作者最多幾本（預設 1）")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    if args.card:
        # 直接卡頁模式：至多下載 1 本
        return download_by_card(args.card, args.out)

    if args.author:
        return download_by_author(args.author, args.count, args.out, title_kw=args.title)

    return download_random(args.count, args.out, per_author=args.per_author)

if __name__ == "__main__":
    main()
