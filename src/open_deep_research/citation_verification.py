"""citation_verification.py —— 离线核验最终报告里引用的URL是不是真的在这次
研究过程中被检索到过（fetched_sources 来自 researcher_tools 对 ToolMessage
内容的结构化解析，见 state.py 的 FETCHED_SOURCES_DOC）。

设计原则：
- 纯离线、纯字符串/数据处理，不发任何网络请求，不调用任何模型API。
- 只标记，不修改/不删除 final_report 里的任何内容——这是只读分析工具，
  调用方自己决定拿到结果之后要怎么用。
- URL 归一化用于比较，不是"猜测两个URL是不是同一个东西"的模糊匹配——
  只做明确、可解释的归一化规则（见 normalize_url），归一化不掉的差异就
  判定为不同URL，不做启发式近似匹配。
"""

import re
from urllib.parse import urlsplit, urlunsplit

_URL_RE = re.compile(r"https?://\S+")

# URL 末尾常见的标点，在正文里URL后面经常紧跟着这些符号（句号、右括号、
# 逗号等），属于正文标点不属于URL本身，归一化时需要剥掉，否则会把
# "https://x.com." 和 "https://x.com" 误判成不同URL。
_TRAILING_PUNCTUATION = ".,;:)]}>\"'"


def normalize_url(url: str) -> str:
    """明确规则的URL归一化，不是模糊匹配：
    - 剥掉URL末尾可能混入的正文标点。
    - scheme/host 转小写（host本身大小写不敏感，scheme同理）；path/query
      保持原样（这两部分可能是大小写敏感的，不能一律转小写）。
    - 去掉默认端口（http:80 / https:443）。
    - 去掉末尾的单个 "/"（但根路径"/"本身保留，避免把 "https://x.com"
      和 "https://x.com/" 之外的情况也错误合并）。
    """
    url = url.rstrip(_TRAILING_PUNCTUATION)
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[: -len(":80")]
    elif scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[: -len(":443")]
    path = parts.path
    if path.endswith("/") and path != "/":
        path = path[:-1]
    return urlunsplit((scheme, netloc, path, parts.query, ""))  # fragment丢弃，不参与比较


def extract_cited_urls(final_report_text: str) -> list[str]:
    """从最终报告正文里找出所有看起来像URL的子串（http/https开头，直到
    空白字符为止），不区分是不是在"Sources"小节里——不同模型/不同报告的
    来源小节格式不统一，直接扫全文找URL比依赖某个固定小节标题更稳健。
    返回的是归一化之后的URL列表，可能含重复（重复由调用方按需处理）。
    """
    return [normalize_url(m) for m in _URL_RE.findall(final_report_text)]


def verify_citations(final_report_text: str, fetched_sources: list[dict]) -> dict:
    """核心校验函数。返回结构：
    {
        "valid": [{"url": ..., "source": ..., "research_topic": ...}, ...],   # 报告里引用的URL，确实在fetched_sources里出现过
        "invalid": [url, ...],                                                # 报告里引用的URL，fetched_sources里完全没有——可能是模型编的
        "missing": [{"url": ..., "source": ..., "research_topic": ...}, ...], # fetched_sources里真实检索到过，但报告正文完全没提到
    }
    只标记，不改final_report_text本身。
    """
    cited_urls = set(extract_cited_urls(final_report_text))

    normalized_fetched = {}  # normalized_url -> 原始fetched_sources条目（取第一次出现的）
    for entry in fetched_sources:
        norm = normalize_url(entry.get("url", ""))
        if norm and norm not in normalized_fetched:
            normalized_fetched[norm] = entry

    valid = []
    invalid = []
    for url in cited_urls:
        if url in normalized_fetched:
            entry = normalized_fetched[url]
            valid.append({
                "url": url,
                "source": entry.get("source"),
                "research_topic": entry.get("research_topic"),
            })
        else:
            invalid.append(url)

    missing = [
        {"url": norm_url, "source": entry.get("source"), "research_topic": entry.get("research_topic")}
        for norm_url, entry in normalized_fetched.items()
        if norm_url not in cited_urls
    ]

    return {"valid": valid, "invalid": invalid, "missing": missing}
