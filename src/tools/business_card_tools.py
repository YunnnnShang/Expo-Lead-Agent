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
        "请输出如下JSON格式。\n"
        "【绝对约束】飞书CRM中只有以下6个字段是真实存在的，严禁输出任何其他键名：\n"
        "- 公司名称: 公司全称（无法确定则留空字符串\"\"）\n"
        "- 官网: 官网链接（仅域名，去除协议前缀；无法确定则留空字符串\"\"，绝对禁止传null）\n"
        "- 邮箱: 联系人邮箱（无法确定则留空字符串\"\"，绝对禁止传null）\n"
        "- 跟进记录/展会备注: 大文本字段，必须按下方Markdown格式写入所有碎片信息\n"
        "- 数据状态: 固定枚举，只能从 [\"待跟进\", \"调研中\", \"调研完成\"] 中选择，名片录入时默认为\"待跟进\"\n"
        "\n"
        "【碎片信息合并规则】名片和补充文本中除了公司名称、官网、邮箱之外的所有信息，"
        "包括但不限于：联系人姓名、职位、电话、国家/地区、城市、地址、社交账号、客户分类、价值标签、"
        "意向产品、预计需求量、下一步行动、线索来源等，"
        "必须全部合并到\"跟进记录/展会备注\"中，采用如下严格格式（不要遗漏任何信息）：\n\n"
        "【YYYY年MM月DD日展会录入】\n"
        "- 联系人姓名：[具体姓名，若无写未知]\n"
        "- 职位：[具体职位，若无写未知]\n"
        "- 电话：[具体电话，若无写未知]\n"
        "- 国家/地区：[如：德国/英国/未知]\n"
        "- 城市：[具体城市，若无写未知]\n"
        "- 公司地址：[具体地址，若无写未知]\n"
        "- 社交账号/即时通讯：[如有填写，若无写未知]\n"
        "- 客户分类：[直接客户/代理商/分销商/独立采购商/未知]\n"
        "- 价值标签：[战略级/高管熟人/高意向/普通/未知]\n"
        "- 线索来源：[如：展会现场/代理商推荐/未知]\n"
        "- 意向产品：[具体产品，若无写未知]\n"
        "- 预计需求量：[具体需求量描述，若无写未知]\n"
        "- 下一步行动：[后续跟进计划，若无写未知]\n"
        "- 销售现场补充：[原文或摘要]\n"
        "\n"
        "JSON格式示例：\n"
        "{\n"
        '  "公司名称": "Müller Automation GmbH",\n'
        '  "官网": "mueller-auto.de",\n'
        '  "邮箱": "h.mueller@mueller-auto.de",\n'
        '  "跟进记录/展会备注": "【2026年05月23日展会录入】\\n- 联系人姓名：Hans Müller\\n- 职位：CEO\\n- 电话：未知\\n- 国家/地区：德国\\n- 城市：慕尼黑\\n- 公司地址：未知\\n- 社交账号/即时通讯：未知\\n- 客户分类：直接客户\\n- 价值标签：高意向\\n- 线索来源：2026海外展会\\n- 意向产品：工业网关\\n- 预计需求量：500台/年\\n- 下一步行动：下周发样品\\n- 销售现场补充：对工业网关很感兴趣，预计今年采购500台，下周回国后要发样品。",\n'
        '  "数据状态": "待跟进"\n'
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
        "【绝对约束】飞书CRM中只有以下6个字段是真实存在的，严禁输出任何其他键名：\n"
        "- 公司名称、官网、邮箱、跟进记录/展会备注、数据状态\n"
        "除公司名称/官网/邮箱外，所有碎片信息（联系人、职位、电话、国家/地区、城市、地址、"
        "社交账号、客户分类、价值标签、意向产品、预计需求量、下一步行动、线索来源等）"
        "必须全部合并到\"跟进记录/展会备注\"字段中，采用如下严格格式：\n\n"
        "【YYYY年MM月DD日展会录入】\n"
        "- 联系人姓名：[具体姓名，若无写未知]\n"
        "- 职位：[具体职位，若无写未知]\n"
        "- 电话：[具体电话，若无写未知]\n"
        "- 国家/地区：[如：德国/英国/未知]\n"
        "- 城市：[具体城市，若无写未知]\n"
        "- 公司地址：[具体地址，若无写未知]\n"
        "- 社交账号/即时通讯：[如有填写，若无写未知]\n"
        "- 客户分类：[直接客户/代理商/分销商/独立采购商/未知]\n"
        "- 价值标签：[战略级/高管熟人/高意向/普通/未知]\n"
        "- 线索来源：[如：展会现场/代理商推荐/未知]\n"
        "- 意向产品：[具体产品，若无写未知]\n"
        "- 预计需求量：[具体需求量描述，若无写未知]\n"
        "- 下一步行动：[后续跟进计划，若无写未知]\n"
        "- 销售现场补充：[原文或摘要]\n"
        "\n数据状态只能从 [\"待跟进\", \"调研中\", \"调研完成\"] 中选择，名片录入时默认为\"待跟进\"。"
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

    # 2. 按官网精确搜索（空值/非字符串直接跳过）
    norm_web = _normalize_website(website)
    if norm_web and isinstance(website, str):
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

    # 3. 按邮箱后缀搜索（空值/非字符串直接跳过）
    domain = _extract_domain_from_email(email)
    if domain and isinstance(email, str):
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


# 飞书CRM允许写入的字段白名单（严格）
_CRM_FIELD_WHITELIST = {
    "公司名称", "官网", "邮箱",
    "跟进记录/展会备注", "AI调研报告链接", "数据状态"
}

# 数据状态合法枚举值
_CRM_STATUS_ENUM = {"待跟进", "调研中", "调研完成"}


def _sanitize_crm_fields(fields: dict) -> dict:
    """清洗并校验要写入CRM的字段，只保留白名单中的键，过滤非法值"""
    sanitized = {}
    for k, v in fields.items():
        if k not in _CRM_FIELD_WHITELIST:
            continue
        # 强制字符串类型处理（官网/邮箱/公司名称/备注/报告链接）
        if v is None or v is False:
            continue
        if k == "数据状态":
            sv = str(v).strip()
            if sv in _CRM_STATUS_ENUM:
                sanitized[k] = sv
            else:
                sanitized[k] = "待跟进"
        else:
            sanitized[k] = str(v).strip() if v else ""
    return sanitized


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

    if not isinstance(fields, dict):
        return json.dumps({"error": "record_json 必须是一个对象字典"}, ensure_ascii=False)

    # 硬拦截：只保留白名单字段，过滤空值/null
    safe_fields = _sanitize_crm_fields(fields)
    removed = [k for k in fields if k not in _CRM_FIELD_WHITELIST]

    if not safe_fields:
        return json.dumps({
            "error": "字段校验失败：record_json 中无可合法写入的字段",
            "removed_keys": removed,
            "hint": "只允许以下字段：公司名称、官网、邮箱、跟进记录/展会备注、AI调研报告链接、数据状态"
        }, ensure_ascii=False)

    client = FeishuCrmClient()
    try:
        if mode == "update":
            if not record_id:
                return json.dumps({"error": "update 模式需要提供 record_id"}, ensure_ascii=False)
            result = client.update_record(app_token, table_id, record_id, safe_fields)
            return json.dumps({
                "success": True,
                "mode": "update",
                "record_id": result.get("record_id"),
                "fields": result.get("fields", {}),
                "removed_keys": removed
            }, ensure_ascii=False)
        else:
            result = client.create_record(app_token, table_id, safe_fields)
            return json.dumps({
                "success": True,
                "mode": "create",
                "record_id": result.get("record_id"),
                "fields": result.get("fields", {}),
                "removed_keys": removed
            }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"CRM写入失败: {e}"}, ensure_ascii=False)
