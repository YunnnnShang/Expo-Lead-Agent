"""名片解析与 CRM 录入工具"""
import json
import os
import re
from difflib import SequenceMatcher
from langchain.tools import tool
from coze_coding_dev_sdk import LLMClient
from coze_coding_utils.runtime_ctx.context import new_context
from coze_coding_utils.log.write_log import request_context
from tools.feishu_crm_client import FeishuCrmClient, get_default_app_token, get_default_table_id


def _get_ctx():
    ctx = request_context.get()
    if ctx is None:
        ctx = new_context(method="business_card_tools")
    return ctx


def _normalize_website(website: str) -> str:
    """清洗官网字符串，去除协议前缀和尾部斜杠，转小写"""
    if not website:
        return ""
    w = website.strip().lower()
    w = re.sub(r"^https?://", "", w)
    w = w.rstrip("/")
    return w


def _extract_domain_from_email(email: str) -> str:
    """从邮箱提取域名"""
    if not email or "@" not in email:
        return ""
    return email.split("@")[1].strip().lower()


def _similarity(a: str, b: str) -> float:
    """计算两个字符串的相似度 (0~1)"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _run_llm_parse(ctx, messages, model: str, temperature: float, max_tokens: int):
    """调用 LLM 解析，内部函数供 tool 使用"""
    client = LLMClient(ctx=ctx)
    return client.invoke(
        messages=messages,
        model=model,
        temperature=temperature,
        max_completion_tokens=max_tokens
    )


def _extract_json_from_content(content) -> dict:
    """从 LLM 响应中提取 JSON"""
    if isinstance(content, list):
        text_parts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
        text = " ".join(text_parts)
    else:
        text = str(content)
    try:
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return json.loads(text)
    except Exception:
        return {"raw_parse_result": text, "error": "JSON parse failed"}


@tool
def parse_business_card(image_url: str, supplement_text: str = "") -> str:
    """解析名片图片和补充文本，提取结构化客户/代理商线索信息。

    Args:
        image_url: 名片图片的URL地址（支持jpg/png等格式）
        supplement_text: 销售现场补充的碎片化文本（如意向产品、需求量、下一步行动等），可为空

    Returns:
        JSON字符串，包含解析出的名片字段和语义槽填充结果
    """
    ctx = _get_ctx()

    system_prompt = (
        "你是一位专业的名片信息提取助手，擅长从名片图片中精准提取信息，"
        "并能理解销售人员的现场补充文本。即使名片图片模糊、反光、包含手写修改或小语种，"
        "你也能准确识别。对于手写修改的内容（如电话号码），优先采用手写值覆盖印刷值。"
        "你必须以JSON格式输出，不要包含任何解释性文字。"
    )

    base_text = (
        "请提取名片信息并结合销售补充文本进行语义槽填充。\n\n"
        f"销售补充文本：{supplement_text or '（无）'}\n\n"
        "请输出如下JSON格式（无法确定的字段留空字符串）。\n"
        "重要约束：以下字段是CRM中真实存在的列，严禁输出任何不在这个列表中的字段名：\n"
        "- company_name: 公司全称\n"
        "- company_alias: 公司别名/简称\n"
        "- country_region: 国家/地区\n"
        "- city: 城市\n"
        "- address: 公司地址\n"
        "- website: 官网链接（仅域名，去除协议前缀）\n"
        "- contact_name: 联系人姓名\n"
        "- contact_title: 职位\n"
        "- email: 邮箱\n"
        "- phone: 电话/手机（优先采用手写修改后的号码）\n"
        "- social_account: 社交账号/即时通讯\n"
        "- customer_type: 客户分类（直接客户/代理商/分销商/独立采购商）\n"
        "- value_tag: 价值标签（战略级/高管熟人/高意向/普通）\n"
        "- scene_notes: 现场跟进备注（必须包含：意向产品、预计需求量、下一步行动、线索来源等所有销售碎片信息，合并写入此字段）\n"
        "\n"
        "注意：意向产品、预计需求量、下一步行动、线索来源等信息不要作为独立字段输出，"
        "必须全部合并写入 scene_notes（现场跟进备注）中，格式清晰便于阅读。\n"
        "\n"
        "JSON格式示例：\n"
        "{\n"
        '  "company_name": "Müller Automation GmbH",\n'
        '  "company_alias": "Müller Auto",\n'
        '  "country_region": "德国",\n'
        '  "city": "慕尼黑",\n'
        '  "address": "",\n'
        '  "website": "mueller-auto.de",\n'
        '  "contact_name": "Hans Müller",\n'
        '  "contact_title": "CEO",\n'
        '  "email": "h.mueller@mueller-auto.de",\n'
        '  "phone": "",\n'
        '  "social_account": "",\n'
        '  "customer_type": "直接客户",\n'
        '  "value_tag": "高意向",\n'
        '  "scene_notes": "[线索来源] 2026海外展会\\n[意向产品] 工业网关\\n[预计需求量] 500台/年\\n[下一步行动] 下周发样品\\n[销售补充] 对工业网关很感兴趣"\n'
        "}"
    )

    from langchain_core.messages import SystemMessage, HumanMessage

    # 尝试图片+文本解析
    if image_url:
        user_content = [
            {"type": "text", "text": base_text},
            {"type": "image_url", "image_url": {"url": image_url}}
        ]
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ]
        try:
            response = _run_llm_parse(ctx, messages, "doubao-seed-2-0-pro-260215", 0.3, 4096)
            parsed = _extract_json_from_content(response.content)
            if "error" not in parsed:
                return json.dumps(parsed, ensure_ascii=False, indent=2)
            # JSON解析失败但API调用成功，继续降级
        except Exception as e:
            err_str = str(e).lower()
            if "download" in err_str or "404" in err_str or "status code" in err_str:
                # 图片下载失败，降级为纯文本解析
                pass
            else:
                # 其他错误直接返回
                return json.dumps({
                    "error": f"名片解析失败: {e}",
                    "supplement_text": supplement_text
                }, ensure_ascii=False)

    # 纯文本解析（图片不可用或为空时）
    fallback_text = (
        "名片图片暂时无法访问或解析。请仅基于以下销售补充文本，"
        "尽可能提取和推断结构化线索信息，并以相同JSON格式输出。\n\n"
        f"销售补充文本：{supplement_text or '（无）'}\n\n"
        "重要约束：只输出以下字段：company_name, company_alias, country_region, city, address, "
        "website, contact_name, contact_title, email, phone, social_account, customer_type, value_tag, scene_notes。"
        "严禁输出 interest_product, estimated_volume, next_step, source 等独立字段。"
        "所有销售碎片信息（意向产品、预计需求量、下一步行动、线索来源）必须全部合并写入 scene_notes。"
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=fallback_text)
    ]
    try:
        response = _run_llm_parse(ctx, messages, "doubao-seed-2-0-pro-260215", 0.3, 4096)
        parsed = _extract_json_from_content(response.content)
        parsed["_parse_note"] = "图片不可用，仅基于补充文本解析"
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "error": f"名片解析失败（含降级重试）: {e}",
            "supplement_text": supplement_text
        }, ensure_ascii=False)


@tool
def check_crm_duplicate(company_name: str, website: str = "", email: str = "",
                        app_token: str = "", table_id: str = "") -> str:
    """检查飞书CRM中是否已存在该公司记录，支持多维度去重。

    Args:
        company_name: 待检查的公司名称（必填）
        website: 公司官网，用于精确匹配
        email: 联系人邮箱，用于后缀匹配
        app_token: 飞书多维表格的 app_token（如未提供则从环境变量读取）
        table_id: 飞书多维表格的 table_id（如未提供则从环境变量读取）

    Returns:
        JSON字符串，包含匹配结果列表和去重建议
    """
    # 强类型校验：飞书 filter value 只接受字符串/列表
    if not company_name or not isinstance(company_name, str):
        return json.dumps({"matches": [], "recommendation": "NO_MATCH", "reason": "公司名称为空，无法去重"}, ensure_ascii=False)

    app_token = app_token or get_default_app_token()
    table_id = table_id or get_default_table_id()
    if not app_token or not table_id:
        return json.dumps({"error": "缺少 app_token 或 table_id，请通过参数传入或配置环境变量"}, ensure_ascii=False)

    client = FeishuCrmClient()
    all_matches = []

    seen_ids = set()

    # 1. 按公司名称模糊搜索
    try:
        name_results = client.search_by_company_name(app_token, table_id, str(company_name))
        for r in name_results:
            rid = r.get("record_id")
            if rid in seen_ids:
                continue
            fields = r.get("fields", {})
            db_name = fields.get("公司名称", "")
            sim = _similarity(company_name, db_name)
            if sim >= 0.5:
                seen_ids.add(rid)
                all_matches.append({
                    "record_id": rid,
                    "fields": fields,
                    "match_type": "name_similarity",
                    "similarity": round(sim, 2)
                })
    except Exception as e:
        return json.dumps({"error": f"按名称搜索失败: {e}"}, ensure_ascii=False)

    # 2. 按官网精确搜索
    norm_web = _normalize_website(website)
    if norm_web:
        try:
            web_results = client.search_by_website(app_token, table_id, website)
            for r in web_results:
                rid = r.get("record_id")
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    all_matches.append({
                        "record_id": rid,
                        "fields": r.get("fields", {}),
                        "match_type": "website_exact",
                        "similarity": 1.0
                    })
        except Exception:
            pass

    # 3. 按邮箱后缀搜索
    domain = _extract_domain_from_email(email)
    if domain:
        try:
            mail_results = client.search_by_email_suffix(app_token, table_id, email)
            for r in mail_results:
                rid = r.get("record_id")
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    all_matches.append({
                        "record_id": rid,
                        "fields": r.get("fields", {}),
                        "match_type": "email_domain",
                        "similarity": 1.0
                    })
        except Exception:
            pass

    # 按相似度排序
    all_matches.sort(key=lambda x: x.get("similarity", 0), reverse=True)

    result = {
        "company_name": company_name,
        "website": website,
        "email": email,
        "matches_count": len(all_matches),
        "matches": all_matches,
        "suggestion": ""
    }

    if all_matches:
        top = all_matches[0]
        if top.get("similarity", 0) >= 0.8:
            result["suggestion"] = "HIGH_SIMILARITY: 发现高度相似记录，建议合并更新"
        else:
            result["suggestion"] = "MEDIUM_SIMILARITY: 发现可能相关记录，建议人工确认是否合并"
    else:
        result["suggestion"] = "NO_MATCH: 未发现重复记录，建议新建"

    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def write_crm_record(record_json: str, mode: str = "create", record_id: str = "",
                     app_token: str = "", table_id: str = "") -> str:
    """向飞书CRM写入或更新记录。

    Args:
        record_json: JSON字符串，包含要写入的字段键值对
        mode: 操作模式，create（新增）或 update（更新）
        record_id: 更新模式时需要提供现有记录ID
        app_token: 飞书多维表格的 app_token
        table_id: 飞书多维表格的 table_id

    Returns:
        JSON字符串，包含操作结果和记录ID
    """
    app_token = app_token or get_default_app_token()
    table_id = table_id or get_default_table_id()
    if not app_token or not table_id:
        return json.dumps({"error": "缺少 app_token 或 table_id"}, ensure_ascii=False)

    try:
        fields = json.loads(record_json)
    except Exception as e:
        return json.dumps({"error": f"record_json 解析失败: {e}"}, ensure_ascii=False)

    client = FeishuCrmClient()
    try:
        if mode == "update":
            if not record_id:
                return json.dumps({"error": "update 模式需要提供 record_id"}, ensure_ascii=False)
            result = client.update_record(app_token, table_id, record_id, fields)
            return json.dumps({
                "success": True,
                "mode": "update",
                "record_id": result.get("record_id"),
                "fields": result.get("fields", {})
            }, ensure_ascii=False)
        else:
            result = client.create_record(app_token, table_id, fields)
            return json.dumps({
                "success": True,
                "mode": "create",
                "record_id": result.get("record_id"),
                "fields": result.get("fields", {})
            }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"CRM写入失败: {e}"}, ensure_ascii=False)
