# -*- coding: utf-8 -*-
"""
电商客服数据扩充脚本
读取现有 customer_service.json（30 条）作为 few-shot 示例，
按意图矩阵调用智谱 GLM 生成变体，去重、校验后合并输出。
"""
import json
import os
import re
import time
from difflib import SequenceMatcher

from zhipuai import ZhipuAI

# ============ 配置 ============
INPUT_FILE = "data/customer_service.json"
OUTPUT_FILE = "data/customer_service_300.json"
API_KEY = os.environ.get("ZHIPU_API_KEY", "")
MODEL = "glm-4-flash"
TARGET_TOTAL = 300
SIMILARITY_THRESHOLD = 0.85  # 相似度超过此值视为重复

# 意图矩阵：每个意图指定参考示例的索引（对应你现有 30 条的序号）
INTENT_MATRIX = [
    {
        "intent": "物流查询与催发货",
        "count": 35,
        "example_idx": [0, 8, 15, 24, 29],
        "variants": "问何时到、问物流不更新、问签收未收到、问怎么查物流、催发货、问能否指定快递、问快递丢件、问改地址后物流",
    },
    {
        "intent": "退货退款",
        "count": 40,
        "example_idx": [1, 6, 14, 23, 25, 28],
        "variants": "问退货操作、问质量问题退货、问退货邮费、问退款到账时间、问价保补差、问7天无理由、问已拆封能否退、问换货流程、问退款失败",
    },
    {
        "intent": "支付相关",
        "count": 35,
        "example_idx": [2, 9, 18],
        "variants": "问支付失败、问支付方式、问货到付款、问重复扣款、问优惠券未抵扣、问支付限额、问分期付款、问支付安全",
    },
    {
        "intent": "会员权益",
        "count": 35,
        "example_idx": [3, 11, 17],
        "variants": "问会员优惠、问会员升级、问积分获取、问积分兑换、问会员有效期、问会员降级、问专属客服、问会员生日礼",
    },
    {
        "intent": "发票",
        "count": 35,
        "example_idx": [4, 13, 27],
        "variants": "问电子发票、问专票、问公司抬头、问发票抬头修改、问发票重开、问开票时间、问发票丢失、问个人抬头发票",
    },
    {
        "intent": "订单修改与取消",
        "count": 35,
        "example_idx": [5, 10, 16, 12],
        "variants": "问改地址、问改商品、问取消订单、问补货提醒、问改收货人、问合并订单、问订单超时未支付、问改配送时间",
    },
    {
        "intent": "商品与售后",
        "count": 35,
        "example_idx": [7, 19, 6],
        "variants": "问保修、问质量问题、问优惠券使用、问商品真伪、问商品缺件、问安装服务、问三包政策、问维修进度",
    },
    {
        "intent": "账号与其他",
        "count": 30,
        "example_idx": [20, 21, 26],
        "variants": "问账号被盗、问联系人工、问活动时间、问账号解封、问修改手机号、问注销账号、问投诉渠道、问隐私保护",
    },
]


def load_existing(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_prompt(intent_info, examples):
    examples_text = json.dumps(examples, ensure_ascii=False, indent=2)
    return f"""你是电商客服数据标注员。以下是现有客服问答示例，请严格模仿其话术风格、语气和格式：

{examples_text}

请围绕「{intent_info['intent']}」意图，生成 {intent_info['count']} 条客服问答。
覆盖以下问法角度：{intent_info['variants']}

硬性要求：
1. 每条必须是 JSON 对象，包含且仅包含三个字段：instruction、input、output
2. instruction 格式统一为「用户问：...」，要有多样性：口语化、带情绪、带订单信息、问原因、问操作、问政策等
3. input 是补充信息（订单号、状态、会员等级等），可以为空字符串
4. output 是客服回答，必须与示例话术风格一致，语气礼貌专业
5. 绝对不得编造示例中不存在的政策。现有政策边界：7天无理由、满200减30、黄金会员95折、铂金会员年消费5000、专票3个工作日寄出、价保7天、电子产品保修1年、退款微信支付宝1-3工作日/银行卡3-7工作日
6. 只输出 JSON 数组，不要任何解释文字、不要 markdown 代码块标记

输出格式示例：
[
  {{"instruction": "用户问：...", "input": "...", "output": "..."}},
  ...
]"""


def parse_json_response(text):
    """从模型输出中提取 JSON 数组，容忍 markdown 代码块包裹。"""
    text = text.strip()
    # 去掉 ```json ... ``` 包裹
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    # 找到第一个 [ 和最后一个 ]
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []


def validate_item(item):
    """校验单条数据字段完整、类型正确、非空。"""
    if not isinstance(item, dict):
        return False
    if set(item.keys()) != {"instruction", "input", "output"}:
        return False
    for k in ("instruction", "input", "output"):
        if not isinstance(item[k], str):
            return False
    if not item["instruction"].strip() or not item["output"].strip():
        return False
    return True


def is_duplicate(item, existing_texts, threshold=SIMILARITY_THRESHOLD):
    """基于 instruction+output 的文本相似度去重。"""
    text = item["instruction"] + item["output"]
    for t in existing_texts:
        if SequenceMatcher(None, text, t).ratio() > threshold:
            return True
    return False


def main():
    if not API_KEY:
        raise SystemExit("请先设置环境变量 ZHIPU_API_KEY")

    client = ZhipuAI(api_key=API_KEY)
    existing = load_existing(INPUT_FILE)
    print(f"读取现有数据：{len(existing)} 条")

    # 用现有数据的 instruction+output 初始化去重池
    existing_texts = [x["instruction"] + x["output"] for x in existing]
    all_items = list(existing)

    for intent_info in INTENT_MATRIX:
        if len(all_items) >= TARGET_TOTAL:
            break

        examples = [existing[i] for i in intent_info["example_idx"] if i < len(existing)]
        prompt = build_prompt(intent_info, examples)
        print(f"\n生成意图「{intent_info['intent']}」，目标 {intent_info['count']} 条...")

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
            )
            raw = response.choices[0].message.content
        except Exception as e:
            print(f"  API 调用失败：{e}")
            continue

        candidates = parse_json_response(raw)
        print(f"  模型返回 {len(candidates)} 条候选")

        added = 0
        for item in candidates:
            if len(all_items) >= TARGET_TOTAL:
                break
            if not validate_item(item):
                continue
            if is_duplicate(item, existing_texts):
                continue
            all_items.append(item)
            existing_texts.append(item["instruction"] + item["output"])
            added += 1
        print(f"  校验去重后新增 {added} 条，当前总数 {len(all_items)}")

        time.sleep(1)  # 避免请求过快

    # 写输出
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_items, f, ensure_ascii=False, indent=2)

    print(f"\n完成。总条数 {len(all_items)}，已写入 {OUTPUT_FILE}")
    if len(all_items) < TARGET_TOTAL:
        print(f"提示：未达到 {TARGET_TOTAL} 条，可重复运行或调高各意图 count 后补生成。")


if __name__ == "__main__":
    main()