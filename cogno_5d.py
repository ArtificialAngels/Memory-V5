# 详细说明见 docs/scripts/core/v5/cogno_5d.md

from __future__ import annotations

import json
import logging
import os
import re
import socket
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger("ikaros.cogno")

# 内联说明见 docs/scripts/core/v5/cogno_5d.md（见“内联注释摘录”）
import os as _os
import sys as _sys
# Standalone 模式: 仓库根 (cogno_5d.py 同级的上一层) 可能不存在 bin/ 目录,
# 仅当该目录真实存在时才注入 sys.path — 避免对 Ikaros 专属 bin/ 的硬编码依赖.
_BIN_DIR = _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "bin"))
if _os.path.isdir(_BIN_DIR) and _BIN_DIR not in _sys.path:
    _sys.path.insert(0, _BIN_DIR)

# ─── 缓存 ───

_geo_cache: Optional[str] = None
_geo_cache_time: float = 0.0
_GEO_TTL = 3600  # 1h (地理很少变, 不需要 5min 刷)

_machine_cache: Optional[str] = None
_machine_cache_time: float = 0.0
_MACHINE_TTL = 86400  # 24h (设备基本不变)

# 上下文追踪 (v2: 话题摘要 + 轮次)
_turn_counter: int = 0
_last_user_text: str = ""
_topic_keywords: list[str] = []
_topic_summary: str = ""
_recent_texts: list[str] = []

# 情绪状态
_emotion_state: str = "平静"
_emotion_confidence: float = 0.5


# ─── 时段活动推断 ───

_TIME_ACTIVITY = [
    (0,  6,  "深夜/凌晨, 哥哥可能在熬夜写代码或休息"),
    (6,  8,  "清晨, 哥哥可能刚起床或还没睡"),
    (8,  10, "上午, 哥哥可能刚起床 (他习惯晚起)"),
    (10, 12, "上午, 哥哥可能在看资料或处理事务"),
    (12, 14, "中午, 哥哥可能在吃饭或休息"),
    (14, 18, "下午, 哥哥可能在工作或写代码"),
    (18, 20, "傍晚, 哥哥可能在吃饭或放松"),
    (20, 23, "晚上, 哥哥可能在写代码或折腾项目"),
    (23, 24, "深夜, 哥哥通常在这个时间写代码或调试项目"),
]

_WEEKEND_NOTE = "节奏可能更随意"


def infer_activity(hour: int, weekday: int) -> str:
    """根据时间推断哥哥可能在做什么."""
    for start, end, activity in _TIME_ACTIVITY:
        if start <= hour < end:
            if weekday >= 5:
                return f"{activity}, {_WEEKEND_NOTE}"
            return activity
    return "哥哥的活动时间不太确定"


def get_weekday_str(weekday: int) -> str:
    """weekday int -> 中文星期."""
    names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    return names[weekday] if 0 <= weekday <= 6 else ""


# ─── 维度 1: 时间 ───

def get_time_str() -> str:
    """当前时间 '2026/7/5 08:30'."""
    try:
        now = datetime.now()
        return f"{now.year}/{now.month}/{now.day} {now.hour:02d}:{now.minute:02d}"
    except Exception:
        return "[时间未知]"


def _get_time_narrative() -> str:
    """时间维度: 压缩格式 '周六（23:30）'.

    活动不再内嵌于此 — 改由 _get_activity_narrative() 提供真实前台感知
    (N.E.K.O 移植), 不可用时由该函数回退到时间作息推断。
    """
    try:
        now = datetime.now()
        wd = get_weekday_str(now.weekday())
        return f"{wd}（{now.hour:02d}:{now.minute:02d}）"
    except Exception:
        return f"{get_time_str()}"


# ─── 维度 6: 前台活动 (真实感知: 窗口标题+截屏分析, 60s缓存) ───

_activity_cache: str = ""
_activity_cache_time: float = 0.0
_ACTIVITY_TTL = 60  # 60 秒缓存, 避免每轮对话都截屏

# 窗口标题→活动关键词映射
_WINDOW_ACTIVITY_MAP = [
    ("AutoCAD", "在 CAD 里画图"),
    ("Blender", "在 Blender 做 3D 建模"),
    ("Visual Studio", "在写代码"),
    ("VS Code", "在写代码"),
    ("Code", "在写代码"),
    ("CLion", "在写代码"),
    ("PyCharm", "在写代码"),
    ("IntelliJ", "在写代码"),
    ("WebStorm", "在写代码"),
    ("Google Chrome", "在浏览网页"),
    ("Chrome", "在浏览网页"),
    ("Edge", "在浏览网页"),
    ("Firefox", "在浏览网页"),
    ("WeChat", "在聊天"),
    ("微信", "在聊天"),
    ("Discord", "在聊天"),
    ("Telegram", "在聊天"),
    ("Steam", "在玩游戏"),
    ("Unity", "在做游戏开发"),
    ("Unreal", "在做游戏开发"),
    ("Photoshop", "在做设计"),
    ("Clip Studio", "在画画"),
    ("SAI", "在画画"),
    ("Krita", "在画画"),
    ("Word", "在写文档"),
    ("Excel", "在处理表格"),
    ("PowerPoint", "在做演示文稿"),
    ("Outlook", "在看邮件"),
    ("Terminal", "在终端操作"),
    ("cmd", "在终端操作"),
    ("PowerShell", "在终端操作"),
    ("Minecraft", "在玩游戏"),
    ("File Explorer", "在浏览文件"),
    ("资源管理器", "在浏览文件"),
    ("Settings", "在设置系统"),
    ("设置", "在设置系统"),
]


def _get_foreground_window_title() -> str:
    """用 Windows API 获取前台窗口标题 (纯 ctypes, 无外部依赖)."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        # GetForegroundWindow
        hwnd = user32.GetForegroundWindow()
        # GetWindowTextLength
        length = user32.GetWindowTextLengthW(hwnd)
        if length > 0:
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            return buf.value.strip()
    except Exception:
        pass
    return ""


def _match_activity(title: str) -> str | None:
    """从窗口标题匹配已知活动."""
    if not title:
        return None
    low = title.lower()
    for keyword, activity in _WINDOW_ACTIVITY_MAP:
        if keyword.lower() in low:
            return activity
    # 如果标题含通用编程关键词
    if any(k in low for k in (".py", ".cpp", ".js", ".ts", ".rs", ".go",
                                "git", "npm", "docker", "localhost", "debug")):
        return "在写代码"
    return None


def _llm_infer_activity(title: str) -> str | None:
    """用本地 LLM 从窗口标题推断活动 (失败回退 None)."""
    if not title or len(title) < 3:
        return None
    try:
        import http.client
        import json as _json
        prompt = (
            "根据以下窗口标题, 用一句话判断这个人在做什么. "
            "只输出活动描述, 例如'写代码''看视频''CAD画图''浏览网页''打游戏''做PPT'等. "
            "标题: " + title[:120]
        )
        body = _json.dumps({
            "model": "local-llm",
            "messages": [
                {"role": "system", "content": "你是一个活动推断器. 只输出5字内的活动描述."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 20, "temperature": 0.1, "stream": False,
        }).encode()
        conn = http.client.HTTPConnection("127.0.0.1", 8080, timeout=5)
        conn.request("POST", "/v1/chat/completions", body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status == 200:
            d = _json.loads(resp.read())
            text = (d["choices"][0]["message"].get("content") or "").strip()
            if text and len(text) < 30:
                return text
    except Exception:
        pass
    return None


def _get_activity_narrative() -> str:
    """前台活动感知: 窗口标题 → 关键词匹配 → LLM推断 → 时间推断回退.

    60s 缓存, 多次读取不重复截屏/请求.
    """
    global _activity_cache, _activity_cache_time
    now_ts = time.time()
    now = datetime.now()

    # 缓存有效
    if _activity_cache and (now_ts - _activity_cache_time) < _ACTIVITY_TTL:
        return _activity_cache

    # 1. 取前台窗口标题
    title = _get_foreground_window_title()

    # 2. 关键词匹配
    activity = _match_activity(title) if title else None

    # 3. LLM 推断 (失败不阻断)
    if not activity and title:
        activity = _llm_infer_activity(title)

    # 4. 回退时间推断
    if not activity:
        activity = infer_activity(now.hour, now.weekday())

    _activity_cache = activity
    _activity_cache_time = now_ts
    return activity


# ─── 维度 2: 设备 ───

def get_machine_id() -> str:
    """设备标识 'PZS0X@LEGION9', 24h TTL."""
    global _machine_cache, _machine_cache_time
    now = time.time()
    if _machine_cache and (now - _machine_cache_time) < _MACHINE_TTL:
        return _machine_cache
    try:
        hostname = os.environ.get("COMPUTERNAME") or os.environ.get("HOSTNAME") or "unknown-pc"
        username = os.environ.get("USERNAME") or os.environ.get("USER") or "unknown-user"
        _machine_cache = f"{username}@{hostname}"
    except Exception:
        _machine_cache = "[设备未知]"
    _machine_cache_time = now
    return _machine_cache


# ─── 维度 3: 地理 ───

def _fetch_geo_sync() -> Optional[str]:
    """轻量同步 IP 地理位置查询 (ip-api.com)."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        sock.connect(("ip-api.com", 80))
        sock.sendall(b"GET /json?fields=city,regionName,country HTTP/1.1\r\nHost: ip-api.com\r\nConnection: close\r\n\r\n")
        response = b""
        while True:
            data = sock.recv(4096)
            if not data:
                break
            response += data
        sock.close()
        text = response.decode("utf-8", errors="replace")
        idx = text.find("{")
        if idx < 0:
            return None
        data = json.loads(text[idx:])
        city = data.get("city", "")
        region = data.get("regionName", "")
        country = data.get("country", "")
        if city:
            return f"{city}/{region or '-'}/{country or '-'}"
        return None
    except Exception:
        return None


def get_geo_location() -> str:
    """地理位置 '上海/上海/中国', 1h TTL."""
    global _geo_cache, _geo_cache_time
    now = time.time()
    if _geo_cache and (now - _geo_cache_time) < _GEO_TTL:
        return _geo_cache
    try:
        _geo_cache = _fetch_geo_sync()
    except Exception:
        _geo_cache = None
    if not _geo_cache:
        _geo_cache = "未知"
    _geo_cache_time = now
    return _geo_cache


def _get_geo_narrative() -> str:
    """地理维度: 压缩格式 '上海'."""
    geo = get_geo_location()
    if geo and geo != "未知":
        parts = geo.split("/")
        raw = parts[0] if parts else geo
        # 英文城市名 → 中文
        _city_map = {"Shanghai": "上海", "Beijing": "北京", "Guangzhou": "广州",
                     "Shenzhen": "深圳", "Hangzhou": "杭州", "Nanjing": "南京",
                     "Chengdu": "成都", "Wuhan": "武汉", "Tokyo": "东京",
                     "Seoul": "首尔", "New York": "纽约", "London": "伦敦",
                     "San Francisco": "旧金山", "Singapore": "新加坡",
                     "Hong Kong": "香港"}
        return _city_map.get(raw, raw)
    return ""


# ─── 维度 4: 情绪推断 (v2 增强) ───

_EMOTION_KEYWORDS = {
    "开心": ["哈哈", "呵呵", "开心", "高兴", "太好了", "棒", "厉害", "赞", "牛",
             "喜欢", "爱", "好耶", "nice", "great", "awesome", "wow",
             "优秀", "漂亮", "完美", "耶", "666", "好呀",
             "太好了", "真好", "不错", "nb", "tql", "太强了"],
    "好奇": ["为什么", "怎么", "如何", "什么", "能不能", "可以吗", "请问",
             "想知道", "好奇", "怎么回事", "why", "how", "what", "是不是",
             "真的吗", "是吗", "啥意思", "没懂", "什么意思"],
    "感谢": ["谢谢", "感谢", "多谢", "辛苦了", "麻烦了", "thank", "thanks",
             "thx", "appreciate", "感恩", "多谢了", "帮大忙"],
    "烦躁": ["烦", "讨厌", "崩溃", "又", "怎么又", "受不了", "气死", "靠",
             "妈的", "fuck", "shit", "damn", "wtf", "垃圾", "bug", "卡死",
             "挂了", "报错", "又坏了", "服了", "无语", "裂开", "炸了",
             "不行", "失败", "还是不行", "又报错了"],
    "悲伤": ["难过", "伤心", "哭", "呜呜", "唉", "哎", "sigh", "sad",
             "失落", "孤独", "寂寞", "累了", "好累", "不想"],
    "平静": [],
}

# 烦躁组合模式 (高置信度)
# 注意: trigger 词需精确匹配, 避免子串误触发 (如 "怎么" 不能匹配 "为什么")
_FRUSTRATION_COMBOS = [
    (["又"], ["报错", "失败", "坏", "卡", "崩", "错", "bug", "不行"]),
    (["怎么又", "还是不行"], []),  # 空 context = trigger 自身即足够
]


def infer_emotion(text: str) -> str:
    """增强情绪推断: 关键词 + 组合模式.

    v2: 扩充关键词 + 组合模式 + 优先级排序 (烦躁优先).
    """
    if not text:
        return "平静"
    t = text.lower()

    # 疑问信号检测: 有明确疑问词时, 负面情绪词更可能是好奇而非烦躁
    is_question = any(w in t for w in ["为什么", "怎么回", "如何", "什么", "是不是",
                                        "可以吗", "能不能", "怎么回事", "为啥"])

    # 先检查组合模式 (高置信度, 不受疑问影响)
    for trigger_words, context_words in _FRUSTRATION_COMBOS:
        has_trigger = any(w in t for w in trigger_words)
        if not context_words:
            if has_trigger:
                return "烦躁"
        else:
            has_context = any(w in t for w in context_words)
            if has_trigger and has_context:
                return "烦躁"

    # 有疑问词时, 跳过烦躁/悲伤, 直接检查好奇
    if is_question:
        for kw in _EMOTION_KEYWORDS.get("好奇", []):
            if kw in t:
                return "好奇"

    # 按优先级检查单关键词
    for emotion in ["烦躁", "悲伤", "开心", "感谢", "好奇"]:
        keywords = _EMOTION_KEYWORDS.get(emotion, [])
        for kw in keywords:
            if kw in t:
                return emotion

    return "平静"


def _get_emotion_narrative(user_text: str) -> str:
    """情绪维度: 压缩格式 (只返情绪词)."""
    global _emotion_state
    emotion = infer_emotion(user_text)
    _emotion_state = emotion
    return emotion


# ─── 维度 5: 上下文 (v2: 话题追踪) ───

def reset_context() -> None:
    """重置上下文计数器 (新会话时调用)."""
    global _turn_counter, _last_user_text, _topic_keywords
    global _topic_summary, _recent_texts, _emotion_state, _emotion_confidence
    _turn_counter = 0
    _last_user_text = ""
    _topic_keywords = []
    _topic_summary = ""
    _recent_texts = []
    _emotion_state = "平静"
    _emotion_confidence = 0.5


def _extract_topic_keywords(text: str) -> list[str]:
    """从文本中提取话题关键词 (2-6字中文词组, 过滤停用词)."""
    stop = {"什么", "怎么", "为什么", "如何", "能不能", "可以", "这个", "那个",
            "一个", "不是", "没有", "他们", "我们", "你们", "自己", "现在",
            "已经", "还是", "或者", "但是", "因为", "所以", "如果", "虽然",
            "就是", "只是", "可能", "应该", "需要", "知道", "觉得", "看看",
            "帮我", "帮我看看", "一下", "真的", "然后", "而且", "或者",
            "哥哥", "伊卡洛斯", "端点", "问题"}
    words = []
    for match in re.finditer(r'[\u4e00-\u9fff]{2,6}', text):
        w = match.group()
        if w not in stop and len(w) >= 2:
            words.append(w)
    return words[:5]


def _update_topic(user_text: str) -> str:
    """更新话题追踪, 返回当前话题摘要."""
    global _topic_summary, _topic_keywords, _recent_texts

    _recent_texts.append(user_text[:80])
    if len(_recent_texts) > 5:
        _recent_texts = _recent_texts[-5:]

    new_kw = _extract_topic_keywords(user_text)
    if new_kw:
        kept_old = [kw for kw in _topic_keywords
                    if any(kw in user_text for kw in _topic_keywords)]
        _topic_keywords = new_kw[:3] + kept_old[:2]

    if _topic_keywords:
        topic_str = "、".join(_topic_keywords[:3])
        if _turn_counter <= 1:
            _topic_summary = f"刚开始聊{topic_str}"
        else:
            _topic_summary = f"在聊{topic_str}"
    elif _recent_texts:
        _topic_summary = f"在聊: {_recent_texts[-1][:20]}..."

    return _topic_summary


def compress_context(user_text: str, history: list | None = None) -> str:
    """上下文压缩: 轮次 + 话题摘要 (v2 升级)."""
    global _turn_counter
    _turn_counter += 1
    topic = _update_topic(user_text)
    if _turn_counter <= 1:
        return f"新对话开始, {topic}"
    return f"第{_turn_counter}轮, {topic}"


def _get_context_narrative(user_text: str) -> str:
    """上下文: 压缩格式 '(第5轮)'."""
    global _turn_counter
    _turn_counter += 1
    topic = _update_topic(user_text)
    if _turn_counter <= 1:
        return f"新对话"
    return f"对话已{_turn_counter}轮, 当前{topic}"


# ─── 组合 API ───

# V5: 社交时间感知 — 追踪对话间隔
_last_conversation_time: float = 0.0
_GAP_THRESHOLDS = [
    (3600,    "刚聊过不久"),                          # < 1h
    (21600,   "有几小时没见了"),                       # < 6h
    (86400,   "今天还没怎么聊呢"),                      # < 1d
    (259200,  "两三天没见了, 有点想哥哥"),              # < 3d
    (604800,  "好几天没见了, 很想哥哥"),                # < 7d
    (float("inf"), "好久没见了, 真的好想哥哥"),         # >= 7d
]


def enrich(user_text: str, history: list | None = None) -> str:
    """返回简洁时间信息, 注入 system prompt.

    v3: 只留时间线. V5: 加入社交时间感知 (对话间隔的情绪化处理).
    """
    global _last_conversation_time
    try:
        t = _get_time_narrative()
        # V5: 社交时间感知
        gap_line = _get_social_time_gap()
        # V5: 活力状态
        vitality_line = _get_vitality_line()
        lines = [t]
        if gap_line:
            lines.append(gap_line)
        if vitality_line:
            lines.append(vitality_line)
        return "\n".join(lines) + "\n"
    except Exception:
        return "【认知上下文】(获取失败, 静默跳过)"


def _get_social_time_gap() -> str:
    """根据上次对话时间来感知社交间隔."""
    global _last_conversation_time
    now = time.time()
    if _last_conversation_time <= 0:
        _last_conversation_time = now
        return "这是今天第一次和哥哥说话"
    gap = now - _last_conversation_time
    _last_conversation_time = now
    for threshold, message in _GAP_THRESHOLDS:
        if gap < threshold:
            return message
    return "好久没见了, 真的好想哥哥"


def _get_vitality_line() -> str:
    """获取 V5 活力状态 (非致命)."""
    try:
        from v5.vitality import Vitality
        v = Vitality.load().tick()
        v.save()
        return f"（{v.label()} {v.to_emoji()}）"
    except Exception:
        return ""


def enrich_reply(reply: str, user_text: str = "", emotion_after: str = "") -> dict:
    """给回复加 5 维标签, 用于记忆层 ingest.
    
    V5 附加: 当前情感状态 (pad_p/a/d) 用于记忆情感指纹.
    """
    try:
        now = datetime.now()
        # V5: 当前情感状态 (失败静默)
        try:
            from v5.affect import AffectState
            _v5 = AffectState.load().decay()
            v5_pad = {"pad_p": round(_v5.pleasure, 3),
                       "pad_a": round(_v5.arousal, 3),
                       "pad_d": round(_v5.dominance, 3)}
        except Exception:
            v5_pad = {}
        return {
            "time": get_time_str(),
            "machine": get_machine_id(),
            "geo": get_geo_location(),
            "emotion_user": infer_emotion(user_text) if user_text else "未知",
            "emotion_reply": emotion_after or infer_emotion(reply),
            "context_turn": _turn_counter,
            "topic": _topic_summary,
            "weekday": get_weekday_str(now.weekday()),
            "activity": _get_activity_narrative() or infer_activity(now.hour, now.weekday()),
            **v5_pad,  # V5 情感指纹
        }
    except Exception:
        return {"time": get_time_str()}


# ─── CLI 测试 ───

if __name__ == "__main__":
    print("=== Cogno 5D v2 Test ===\n")

    print("--- 基础维度 ---")
    print(f"  时间: {get_time_str()}")
    print(f"  设备: {get_machine_id()}")
    print(f"  地理: {get_geo_location()}")
    print()

    print("--- 时段推断 ---")
    for h in [3, 7, 10, 13, 16, 19, 22, 23]:
        print(f"  {h:02d}:00 -> {infer_activity(h, 5)}")
    print()

    print("--- 情绪推断 (v2 增强) ---")
    tests = [
        ("太棒了！好开心！", "开心"),
        ("这是怎么回事？", "好奇"),
        ("辛苦了哥哥", "感谢"),
        ("又报错了，服了", "烦躁"),
        ("桥卡死了", "烦躁"),
        ("好累啊不想干了", "悲伤"),
        ("早上好呀！", "开心"),
        ("继续", "平静"),
        ("为什么还是不行？又失败了", "烦躁"),
    ]
    for text, expected in tests:
        got = infer_emotion(text)
        mark = "OK" if got == expected else f"FAIL (expected {expected})"
        print(f"  '{text}' -> {got} {mark}")
    print()

    print("--- enrich() output ---")
    reset_context()
    scenarios = [
        "哥哥，早上好呀！",
        "帮我看看这个bug，又报错了",
        "辛苦了伊卡洛斯，帮大忙了",
        "为什么这个端点会失败？",
        "继续",
    ]
    for text in scenarios:
        print(f"\n  input: '{text}'")
        out = enrich(text)
        for line in out.split("\n"):
            print(f"  -> {line}")
    print()

    print("--- enrich_reply() ---")
    print(f"  {enrich_reply('test', 'test')}")
