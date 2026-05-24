"""飞书多维表格 CRM 客户端封装（普通类，供 Tool 调用）"""
import os
import requests
from functools import wraps
from typing import Any, Optional
from cozeloop.decorator import observe
from coze_workload_identity import Client


def _get_access_token() -> str:
    client = Client()
    return client.get_integration_credential("integration-feishu-base")


def _require_token(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.access_token = _get_access_token()
        if not self.access_token:
            raise ValueError("Failed to get feishu-base access token")
        return func(self, *args, **kwargs)
    return wrapper


class FeishuCrmClient:
    """飞书 CRM 多维表格客户端"""

    def __init__(self, base_url: str = "https://open.larkoffice.com/open-apis", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.access_token = ""

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.access_token}" if self.access_token else "",
            "Content-Type": "application/json; charset=utf-8",
        }

    @observe
    def _request(self, method: str, path: str, params: dict | None = None, json: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        resp = requests.request(method, url, headers=self._headers(), params=params, json=json, timeout=self.timeout)
        resp_data = resp.json()
        if resp_data.get("code") != 0:
            raise Exception(f"Feishu API error: {resp_data}")
        return resp_data

    @_require_token
    def search_by_company_name(self, app_token: str, table_id: str, company_name: str, page_size: int = 100) -> list:
        """按公司名称模糊搜索记录"""
        filter_body = {
            "conditions": [
                {"field_name": "公司名称", "operator": "contains", "value": company_name}
            ],
            "conjunction": "and"
        }
        resp = self._request(
            "POST",
            f"/bitable/v1/apps/{app_token}/tables/{table_id}/records/search",
            params={"page_size": page_size},
            json={"filter": filter_body}
        )
        return resp.get("data", {}).get("items", [])

    @_require_token
    def search_by_website(self, app_token: str, table_id: str, website: str, page_size: int = 100) -> list:
        """按官网精确搜索记录"""
        if not website:
            return []
        filter_body = {
            "conditions": [
                {"field_name": "官网", "operator": "is", "value": website}
            ],
            "conjunction": "and"
        }
        resp = self._request(
            "POST",
            f"/bitable/v1/apps/{app_token}/tables/{table_id}/records/search",
            params={"page_size": page_size},
            json={"filter": filter_body}
        )
        return resp.get("data", {}).get("items", [])

    @_require_token
    def search_by_email_suffix(self, app_token: str, table_id: str, email: str, page_size: int = 100) -> list:
        """按邮箱后缀搜索记录"""
        if not email or "@" not in email:
            return []
        suffix = email.split("@")[1]
        filter_body = {
            "conditions": [
                {"field_name": "邮箱", "operator": "contains", "value": suffix}
            ],
            "conjunction": "and"
        }
        resp = self._request(
            "POST",
            f"/bitable/v1/apps/{app_token}/tables/{table_id}/records/search",
            params={"page_size": page_size},
            json={"filter": filter_body}
        )
        return resp.get("data", {}).get("items", [])

    @_require_token
    def create_record(self, app_token: str, table_id: str, fields: dict) -> dict:
        """新增单条记录"""
        resp = self._request(
            "POST",
            f"/bitable/v1/apps/{app_token}/tables/{table_id}/records/batch_create",
            json={"records": [{"fields": fields}]}
        )
        records = resp.get("data", {}).get("records", [])
        return records[0] if records else {}

    @_require_token
    def update_record(self, app_token: str, table_id: str, record_id: str, fields: dict) -> dict:
        """更新单条记录"""
        resp = self._request(
            "POST",
            f"/bitable/v1/apps/{app_token}/tables/{table_id}/records/batch_update",
            json={"records": [{"record_id": record_id, "fields": fields}]}
        )
        records = resp.get("data", {}).get("records", [])
        return records[0] if records else {}

    @_require_token
    def get_record(self, app_token: str, table_id: str, record_id: str) -> dict:
        """获取单条记录详情"""
        resp = self._request(
            "POST",
            f"/bitable/v1/apps/{app_token}/tables/{table_id}/records/batch_get",
            json={"record_ids": [record_id]}
        )
        records = resp.get("data", {}).get("records", [])
        return records[0] if records else {}


def get_default_app_token() -> str:
    return os.getenv("FEISHU_CRM_APP_TOKEN", "")


def get_default_table_id() -> str:
    return os.getenv("FEISHU_CRM_TABLE_ID", "")
