"""公司AI调研与报告生成工具"""
import json
import os
import re
from urllib.parse import urljoin, urlparse
from langchain.tools import tool
from coze_coding_dev_sdk import LLMClient, SearchClient
from coze_coding_dev_sdk.fetch import FetchClient
from coze_coding_dev_sdk import DocumentGenerationClient
from coze_coding_utils.runtime_ctx.context import new_context
from coze_coding_utils.log.write_log import request_context
from tools.feishu_crm_client import FeishuCrmClient, get_default_app_token, get_default_table_id


def _get_ctx():
    ctx = request_context.get()
    if ctx is None:
        ctx = new_context(method="company_research_tools")
    return ctx


def _extract_company_core_name(company_name: str) -> str:
    """提取公司名的核心词（去掉常见后缀），用于相关性校验"""
    suffixes = [
        " gmbh", " ag", " inc", " ltd", " llc", " co.", " corporation",
        " limited", " holding", " holdings", " group", " s.a.", " s.a",
        " plc", " pty", " bv", " ab", " oy", " as", " asa"
    ]
    lower = company_name.lower()
    for s in suffixes:
        if lower.endswith(s):
            lower = lower[:-len(s)].strip()
            break
    return lower


def _is_relevant_search_result(company_name: str, title: str, snippet: str, content: str) -> bool:
    """判断搜索结果是否与目标公司相关"""
    core_name = _extract_company_core_name(company_name)
    # 构建待检查的文本
    check_text = " ".join(filter(None, [title, snippet, content])).lower()
    # 核心词必须出现在标题或摘要中（防止同名不同公司）
    if core_name in check_text:
        return True
    # 如果公司名本身（含后缀）完全匹配，也算相关
    if company_name.lower() in check_text:
        return True
    return False


def _extract_subpage_urls(homepage_html: str, base_url: str) -> list:
    """从首页HTML中提取关键子页面链接"""
    parsed_base = urlparse(base_url)
    base_domain = parsed_base.netloc.lower()
    # 去掉www前缀做域名匹配
    if base_domain.startswith("www."):
        base_domain = base_domain[4:]

    keywords = [
        "about", "team", "company", "leadership", "management", "people",
        "executive", "board", "director",
        "products", "solutions", "services", "technology",
        "news", "press", "blog", "media",
        "careers", "jobs", "work",
        "contact", "office", "location"
    ]

    found = set()
    # 匹配 <a href="..."> 标签
    for match in re.finditer(r'href\s*=\s*["\']([^"\']+)["\']', homepage_html, re.IGNORECASE):
        href = match.group(1).strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            continue
        # 解析完整URL
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        # 只保留同域名下的页面
        if domain != base_domain and not domain.endswith("." + base_domain):
            continue
        # 只保留HTML页面（排除图片、PDF等）
        path_lower = parsed.path.lower()
        if any(path_lower.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".pdf", ".zip", ".css", ".js"]):
            continue
        # 检查路径是否包含关键字
        for kw in keywords:
            if kw in path_lower:
                found.add(full_url)
                break
    return list(found)[:8]  # 最多取8个子页面，防止过多


def _fetch_url_text(fetch_client, url: str, max_chars: int = 4000) -> str:
    """抓取URL并提取文本内容"""
    try:
        resp = fetch_client.fetch(url=url)
        text_parts = []
        for item in resp.content:
            if hasattr(item, "type") and item.type == "text" and hasattr(item, "text"):
                text_parts.append(item.text)
        full_text = "\n".join(text_parts)
        return full_text[:max_chars] if full_text else ""
    except Exception as e:
        return f"抓取失败: {e}"


@tool
def research_company_online(company_name: str, country: str = "", website: str = "",
                            extra_keywords: str = "") -> str:
    """对公司进行在线公开信息调研，收集基本面、规模、动态、关键人等维度的原始资料。

    核心策略：
    1. 优先深度抓取官网（首页 + about/team/products/news 等子页面），获取一手结构化信息。
    2. 多维度公开搜索，但对结果进行与公司名的相关性过滤，丢弃同名无关噪音。

    Args:
        company_name: 公司名称（必填）
        country: 国家/地区，用于缩小搜索范围
        website: 公司官网，如提供会优先抓取官网内容
        extra_keywords: 额外搜索关键词（如 executive, financial, LinkedIn 等）

    Returns:
        JSON字符串，汇总搜索结果和官网抓取内容
    """
    ctx = _get_ctx()
    search_client = SearchClient(ctx=ctx)
    fetch_client = FetchClient(ctx=ctx)

    results = {
        "company_name": company_name,
        "country": country,
        "website": website,
        "searches": [],
        "website_content": {},
        "notes": []
    }

    # ── 阶段1：深度抓取官网（首页 + 关键子页面） ──
    if website:
        try:
            base_url = website if website.startswith("http") else f"https://{website}"
            # 1.1 抓取首页
            homepage_text = _fetch_url_text(fetch_client, base_url, max_chars=4000)
            results["website_content"]["homepage"] = homepage_text[:3000]

            # 1.2 如果 fetch 返回了原始HTML，从中提取子页面链接
            # 注意：FetchClient 可能只返回文本，我们需要尝试从文本中反推链接
            # 实际上 FetchClient 返回的是结构化内容，不一定有原始HTML
            # 这里我们直接用 search_client 搜索 site:官网 about 等关键词来补充
            subpages = {}
            # 子页面关键词路径映射
            subpage_queries = [
                ("about", f'site:{base_url} "{company_name}" about'),
                ("team", f'site:{base_url} "{company_name}" team leadership executive'),
                ("products", f'site:{base_url} "{company_name}" products solutions'),
                ("news", f'site:{base_url} "{company_name}" news press'),
            ]
            for page_key, sub_q in subpage_queries:
                try:
                    sub_resp = search_client.search(
                        query=sub_q,
                        search_type="web",
                        count=3,
                        need_content=True,
                        need_summary=False
                    )
                    for item in (sub_resp.web_items or []):
                        # 只保留同域名页面
                        item_url = item.url or ""
                        if base_url.replace("https://", "").replace("http://", "").split("/")[0] not in item_url:
                            continue
                        text = (item.content or "")[:2000] if item.content else ""
                        if text and page_key not in subpages:
                            subpages[page_key] = {
                                "url": item_url,
                                "text": text
                            }
                            break
                except Exception:
                    continue
            results["website_content"]["subpages"] = subpages

        except Exception as e:
            results["website_content"] = {"error": f"官网抓取失败: {e}"}

    # ── 阶段2：多维度公开搜索，带相关性过滤 ──
    # 构建更精准的搜索词，避免通用词污染
    queries = [
        f'"{company_name}" company profile about us',
        f'"{company_name}" {country} CEO founder management team',
        f'"{company_name}" news 2025 2026',
    ]
    if extra_keywords:
        queries.append(f'"{company_name}" {extra_keywords}')

    for q in queries:
        try:
            resp = search_client.search(
                query=q,
                search_type="web",
                count=5,
                need_content=True,
                need_summary=True
            )
            items = []
            for item in (resp.web_items or []):
                title = item.title or ""
                snippet = item.snippet or ""
                content = (item.content or "")[:1500] if item.content else ""

                # 相关性过滤：标题/摘要中必须出现公司名核心词
                if not _is_relevant_search_result(company_name, title, snippet, content):
                    # 不相关的结果只保留标题和URL作为参考，不保留正文
                    items.append({
                        "title": title,
                        "url": item.url,
                        "snippet": snippet,
                        "relevance": "low",
                        "note": "内容未出现目标公司名核心词，可能为同名无关结果，已丢弃正文"
                    })
                    continue

                summary = (item.summary or "")[:500] if item.summary else ""
                items.append({
                    "title": title,
                    "url": item.url,
                    "snippet": snippet,
                    "summary": summary,
                    "content": content,
                    "relevance": "high"
                })
            results["searches"].append({"query": q, "items": items})
        except Exception as e:
            results["searches"].append({"query": q, "error": str(e)})

    raw_result = json.dumps(results, ensure_ascii=False, indent=2)
    # 强制截断防止Token爆炸撑爆Agent上下文
    if len(raw_result) > 14000:
        raw_result = raw_result[:14000] + "\n...[数据过长，底层代码已强制截断]..."
    return raw_result


@tool
def generate_company_report(company_name: str, research_data: str,
                            existing_info: str = "", record_id: str = "") -> str:
    """基于调研数据生成公司背调报告PDF，并回写CRM。

    Args:
        company_name: 公司名称
        research_data: research_company_online 返回的JSON字符串
        existing_info: CRM中已有信息的JSON字符串（可选）
        record_id: CRM记录ID（回写报告链接时使用）

    Returns:
        JSON字符串，包含报告URL和核心摘要
    """
    ctx = _get_ctx()
    llm_client = LLMClient(ctx=ctx)
    doc_client = DocumentGenerationClient()

    # 解析调研数据
    try:
        research = json.loads(research_data)
    except Exception:
        research = {"raw": research_data}

    # 构建报告生成 prompt
    from langchain_core.messages import SystemMessage, HumanMessage
    system_prompt = (
        "你是一位资深的商业分析师，专精于海外市场公司背调。"
        "你的任务是基于提供的调研数据，生成一份严谨、有条理、便于阅读的商业背调报告。"
        "报告必须使用Markdown格式，每一条核心结论必须尽可能标注信息来源URL。"
        "如果信息源不足，必须在报告顶部标注 [置信度：低]。"
        "严禁编造没有来源的数据。"
    )

    user_prompt = f"""请为以下公司生成商业背调报告：

公司名称：{company_name}
已有CRM信息：{existing_info or '无'}

调研原始数据：
{json.dumps(research, ensure_ascii=False, indent=2)[:12000]}

请按以下结构生成Markdown报告：

# 商业背调报告：{company_name}

> 数据生成时间：当前日期 | 背调依据：列出使用的信息源
> [置信度：高/中/低]

## 1. 基本面与赛道定位
- 主营业务
- 商业模式
- 行业地位
- 核心竞争对手

## 2. 规模与健康度
- 工商实体与法律状态
- 财务状况（如可获取）
- 员工规模估算（如可获取）

## 3. 组织架构与关键决策人
- CEO/创始人
- 销售/采购切入点

## 4. 近期动态与口碑
- 近3-6个月重大新闻
- 招聘动态
- 合作案例

## 5. 风险提示与销售建议
- 信息缺失提醒
- 销售切入建议

注意：所有带具体数字/事实的论断，尽量在句末用 `[来源：URL]` 标注依据。
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    try:
        response = llm_client.invoke(
            messages=messages,
            model="doubao-seed-2-0-pro-260215",
            temperature=0.4,
            max_completion_tokens=12000
        )
    except Exception as e:
        return json.dumps({"error": f"LLM报告生成失败: {e}"}, ensure_ascii=False)

    report_md = response.content
    if isinstance(report_md, list):
        text_parts = [item.get("text", "") for item in report_md if isinstance(item, dict) and item.get("type") == "text"]
        report_md = "\n".join(text_parts)
    else:
        report_md = str(report_md)

    # 生成 PDF
    try:
        pdf_url = doc_client.create_pdf_from_markdown(report_md, f"{company_name}_背调报告")
    except Exception as e:
        return json.dumps({"error": f"PDF生成失败: {e}", "report_markdown_preview": report_md[:500]}, ensure_ascii=False)

    # 回写CRM报告链接（强制从环境变量读取，剥夺大模型传参权）
    app_token = get_default_app_token()
    table_id = get_default_table_id()
    if app_token and table_id and record_id:
        try:
            crm_client = FeishuCrmClient()
            crm_client.update_record(
                app_token, table_id, record_id,
                {
                    "AI调研报告链接": {
                        "link": pdf_url,
                        "text": "查看背调报告"
                    },
                    "数据状态": "调研完成"
                }
            )
        except Exception as e:
            # 回写失败不阻断，记录到结果中
            return json.dumps({
                "success": True,
                "report_url": pdf_url,
                "report_markdown_length": len(report_md),
                "crm_write_warning": f"报告链接回写CRM失败: {e}"
            }, ensure_ascii=False)

    return json.dumps({
        "success": True,
        "report_url": pdf_url,
        "report_markdown_length": len(report_md),
        "crm_status": "已回写" if (app_token and table_id and record_id) else "未提供CRM信息，跳过回写"
    }, ensure_ascii=False)
