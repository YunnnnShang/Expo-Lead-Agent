"""公司AI调研与报告生成工具"""
import json
import os
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


@tool
def research_company_online(company_name: str, country: str = "", website: str = "",
                            extra_keywords: str = "") -> str:
    """对公司进行在线公开信息调研，收集基本面、规模、动态、关键人等维度的原始资料。

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
        "website_content": ""
    }

    # 定义搜索策略
    queries = [
        f'"{company_name}" company profile about us',
        f'"{company_name}" {country} CEO executive team',
        f'"{company_name}" news recent 2025 2026',
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
                items.append({
                    "title": item.title,
                    "url": item.url,
                    "snippet": item.snippet,
                    "summary": item.summary,
                    "content": item.content
                })
            results["searches"].append({"query": q, "items": items})
        except Exception as e:
            results["searches"].append({"query": q, "error": str(e)})

    # 抓取官网内容
    if website:
        try:
            # 确保有协议前缀
            url = website if website.startswith("http") else f"https://{website}"
            fetch_resp = fetch_client.fetch(url=url)
            text_parts = []
            for item in fetch_resp.content:
                if hasattr(item, "type") and item.type == "text" and hasattr(item, "text"):
                    text_parts.append(item.text)
            results["website_content"] = "\n".join(text_parts)[:8000]
        except Exception as e:
            results["website_content"] = f"官网抓取失败: {e}"

    return json.dumps(results, ensure_ascii=False, indent=2)


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
                {"AI调研报告链接": pdf_url, "数据状态": "调研完成"}
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
