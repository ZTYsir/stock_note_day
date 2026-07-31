# -*- coding: utf-8 -*-
"""
超跌反弹监控 agent
- 监控给定标的（银行/纳指/红利/证券等，可自定义），检测连续下跌
- 只对连续下跌的标的做 LLM 综合分析（超跌反弹角度），结合多源财经新闻
- 输出各时间窗口累积跌幅 + 连续下跌天数，交易日定时发邮件

数据源: akshare(免费)  |  LLM: 云端 OpenAI 兼容接口(千问/DeepSeek 通用)
仅需 2 个 GitHub Secret: LLM_API_KEY(必填) / MAIL_PASS(邮箱授权码, 必填)
其余全部在下方配置区修改。
"""
import os
import time
import datetime as dt

import akshare as ak
import pandas as pd
import requests

# ============================ 配置区（改这里）============================
# ---- 1. 监控标的：代码 -> (名称, 类型)  类型 "etf" 或 "stock" ----
WATCH_LIST = {
    "510880": ("红利ETF",     "etf"),
    "515080": ("中证红利ETF", "etf"),
    "512800": ("银行ETF",     "etf"),
    "512880": ("证券ETF",     "etf"),
    "513100": ("纳指ETF",     "etf"),
    "159941": ("纳指100ETF",  "etf"),
    "510300": ("沪深300ETF",  "etf"),
    # "600519": ("贵州茅台",   "stock"),
}

# ---- 2. 触发：连续下跌达到这些天数才分析（取命中的最大档）----
DOWN_THRESHOLDS = [3, 5, 7]

# ---- 3. 累积跌幅统计窗口（交易日根数，可自定义增删）----
PCT_WINDOWS = {
    "1天": 1, "3天": 3, "5天": 5, "7天": 7, "14天": 14,
    "1月": 21, "3月": 63, "半年": 126, "1年": 252,
}
LOOKBACK_DAYS = 400        # 拉取自然日历史长度（要 > 最长窗口对应的自然日）
RSI_PERIOD    = 14
ADJUST        = "qfq"      # "" 不复权 / "qfq" 前复权 / "hfq" 后复权

# ---- 4. 财经新闻源（冗余抓取，防止某些源在海外 runner 失效）----
#   func: akshare 接口名  params: 参数({today} 会替换成当天 yyyymmdd)
NEWS_SOURCES = [
    {"label": "财联社电报",  "func": "stock_info_global_cls",  "params": {"symbol": "全部"}},
    {"label": "财联社电报2", "func": "stock_telegraph_cls",    "params": {"symbol": "全部"}},
    {"label": "东财快讯",    "func": "stock_info_global_em",   "params": {}},
    {"label": "新浪财经",    "func": "stock_info_global_sina", "params": {}},
    {"label": "富途快讯",    "func": "stock_info_global_futu", "params": {}},
    {"label": "同花顺快讯",  "func": "stock_info_global_ths",  "params": {}},
    {"label": "财经早餐",    "func": "stock_info_cjzc_em",     "params": {}},
    {"label": "新闻联播",    "func": "news_cctv",              "params": {"date": "{today}"}},
    {"label": "宏观全球事件","func": "news_economic_baidu",    "params": {"date": "{today}"}},
    {"label": "百度停复牌",  "func": "news_trade_notify_suspend_baidu", "params": {"date": "{today}"}},
]
NEWS_PER_SOURCE   = 5       # 每个源最多取几条
NEWS_TOTAL_LIMIT  = 30      # 汇总给 LLM 的新闻总条数上限
ENABLE_STOCK_NEWS = True    # 是否对命中标的额外抓个股新闻(stock_news_em)
STOCK_NEWS_PER    = 3       # 每个命中标的抓几条个股新闻

# ---- 5. LLM（OpenAI 兼容接口，千问/DeepSeek 通用，只改这两行切换）----
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
LLM_MODEL    = "qwen-plus"
# DeepSeek: "https://api.deepseek.com/chat/completions"  +  "deepseek-chat"

# ---- 6. 邮件（发件邮箱与 SMTP，收件人可多个）----
MAIL_HOST = "smtp.qq.com"
MAIL_PORT = 465
MAIL_FROM = "974808867@qq.com"
MAIL_TO   = ["974808867@qq.com"]
MAIL_SUBJECT_PREFIX = "【超跌反弹监控】"

SKIP_NON_TRADE_DAY = True   # 非交易日直接退出(节假日靠交易日历挡)
REQUEST_SLEEP = 0.3         # 各接口调用间隔，降低被限流概率
# =====================================================================

# -------- 敏感信息从环境变量 / GitHub Secrets 读，不写死 --------
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
MAIL_PASS   = os.environ.get("MAIL_PASS", "")


def _today_str() -> str:
    return dt.date.today().strftime("%Y%m%d")


def is_trade_day(today: dt.date) -> bool:
    try:
        cal = ak.tool_trade_date_hist_sina()
        days = set(pd.to_datetime(cal["trade_date"]).dt.date)
        return today in days
    except Exception as e:
        print(f"[warn] 交易日历获取失败，默认按交易日处理: {e}")
        return True


def fetch_hist(code: str, asset_type: str) -> pd.DataFrame:
    """取日线历史，统一返回含 date/close、按日期升序的 DataFrame"""
    end = dt.date.today()
    start = end - dt.timedelta(days=LOOKBACK_DAYS)
    s, e = start.strftime("%Y%m%d"), end.strftime("%Y%m%d")
    if asset_type == "etf":
        df = ak.fund_etf_hist_em(symbol=code, period="daily",
                                 start_date=s, end_date=e, adjust=ADJUST)
    else:
        df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                start_date=s, end_date=e, adjust=ADJUST)
    df = df.rename(columns={"日期": "date", "收盘": "close"})[["date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna().sort_values("date").reset_index(drop=True)


def consecutive_down_days(closes: pd.Series) -> int:
    n = 0
    for i in range(len(closes) - 1, 0, -1):
        if closes.iloc[i] < closes.iloc[i - 1]:
            n += 1
        else:
            break
    return n


def window_pcts(closes: pd.Series) -> dict:
    """各窗口累积涨跌幅%。数据不足则用最早一根，键名加*标记"""
    last = closes.iloc[-1]
    out = {}
    for label, k in PCT_WINDOWS.items():
        if len(closes) > k:
            base = closes.iloc[-(k + 1)]
            out[label] = round((last - base) / base * 100, 2)
        elif len(closes) >= 2:
            base = closes.iloc[0]
            out[label + "*"] = round((last - base) / base * 100, 2)
        else:
            out[label] = None
    return out


def calc_rsi(closes: pd.Series, period: int = 14) -> float:
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1 / period, min_periods=period).mean()
    al = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = ag / al.replace(0, 1e-9)
    rsi = 100 - 100 / (1 + rs)
    return round(float(rsi.iloc[-1]), 1) if not rsi.dropna().empty else float("nan")


def scan() -> list:
    hits = []
    for code, (name, atype) in WATCH_LIST.items():
        try:
            df = fetch_hist(code, atype)
            time.sleep(REQUEST_SLEEP)
            if len(df) < RSI_PERIOD + 2:
                continue
            closes = df["close"]
            down = consecutive_down_days(closes)
            threshold = max([d for d in DOWN_THRESHOLDS if down >= d], default=0)
            if threshold == 0:
                continue
            hits.append({
                "code": code, "name": name,
                "down_days": down, "threshold": threshold,
                "last_close": round(float(closes.iloc[-1]), 3),
                "rsi": calc_rsi(closes, RSI_PERIOD),
                "pcts": window_pcts(closes),
            })
            print(f"[hit] {name}({code}) 连跌{down}天 RSI={hits[-1]['rsi']}")
        except Exception as ex:
            print(f"[warn] {name}({code}) 处理失败: {ex}")
    return hits


def _extract_titles(df: pd.DataFrame, n: int) -> list:
    if df is None or len(df) == 0:
        return []
    text_cols = [c for c in df.columns
                 if any(k in str(c) for k in ["标题", "内容", "title", "摘要", "简介"])]
    col = text_cols[0] if text_cols else df.columns[0]
    return df[col].dropna().astype(str).str.strip().head(n).tolist()


def fetch_market_news() -> list:
    """多源冗余抓取全局财经快讯"""
    today = _today_str()
    collected = []
    for src in NEWS_SOURCES:
        if len(collected) >= NEWS_TOTAL_LIMIT:
            break
        try:
            params = {k: (v.replace("{today}", today) if isinstance(v, str) else v)
                      for k, v in src.get("params", {}).items()}
            fn = getattr(ak, src["func"])
            df = fn(**params)
            for t in _extract_titles(df, NEWS_PER_SOURCE):
                collected.append(f"[{src['label']}] {t}")
            time.sleep(REQUEST_SLEEP)
        except Exception as ex:
            print(f"[news warn] {src['func']} 失败: {ex}")
    return collected[:NEWS_TOTAL_LIMIT]


def fetch_stock_news(hits: list) -> list:
    if not ENABLE_STOCK_NEWS:
        return []
    out = []
    for h in hits:
        try:
            df = ak.stock_news_em(symbol=h["code"])
            for t in _extract_titles(df, STOCK_NEWS_PER):
                out.append(f"[{h['name']}] {t}")
            time.sleep(REQUEST_SLEEP)
        except Exception as ex:
            print(f"[news warn] 个股新闻 {h['code']} 失败: {ex}")
    return out


def analyze_with_llm(hits: list, market_news: list, stock_news: list) -> str:
    if not LLM_API_KEY:
        return "（未配置 LLM_API_KEY，跳过 AI 分析）"
    lines = []
    for h in hits:
        pct_str = " ".join(f"{k}:{v}%" for k, v in h["pcts"].items() if v is not None)
        lines.append(f"- {h['name']}({h['code']})：连跌{h['down_days']}天，RSI(14)={h['rsi']}，"
                     f"最新价{h['last_close']}；各窗口累计涨跌幅 {pct_str}")
    news_block = "\n".join(market_news) if market_news else "（无）"
    stock_block = "\n".join(stock_news) if stock_news else "（无）"
    prompt = (
        "你是一名量化投资助手。以下是今日筛选出的、已连续下跌的标的（银行/红利/证券/纳指等，"
        "长期看具均值回归特征）及其多窗口累计涨跌幅：\n" + "\n".join(lines) +
        "\n\n【今日市场财经快讯】\n" + news_block +
        "\n\n【相关个股新闻】\n" + stock_block +
        "\n\n请结合上述新闻，从『超跌反弹』角度逐个分析：(1)超跌程度（连跌天数+累计跌幅+RSI是否超卖）；"
        "(2)结合新闻判断此次下跌是普跌情绪还是有针对性重大利空(利空会削弱反弹逻辑)；"
        "(3)反弹概率定性评估与操作建议(观望/分批试探/暂不介入)。最后给整体风险提示。中文，简洁，可用短列表。"
    )
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.3}
    try:
        r = requests.post(LLM_BASE_URL, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as ex:
        return f"（LLM 调用失败：{ex}）"


def build_html(hits: list, ai_text: str, news_count: int) -> str:
    win_headers = "".join(f"<th>{k}</th>" for k in PCT_WINDOWS.keys())

    def cell(v):
        if v is None:
            return "<td style='text-align:right;color:#bbb'>-</td>"
        color = "#c0392b" if v < 0 else "#27ae60"
        return f"<td style='text-align:right;color:{color}'>{v}%</td>"

    rows = ""
    for h in hits:
        cells = ""
        for k in PCT_WINDOWS.keys():
            v = h["pcts"].get(k, h["pcts"].get(k + "*"))
            cells += cell(v)
        rows += (f"<tr><td>{h['name']}</td><td>{h['code']}</td>"
                 f"<td style='text-align:center'><b>{h['down_days']}</b></td>"
                 f"<td style='text-align:right'>{h['last_close']}</td>"
                 f"<td style='text-align:center'>{h['rsi']}</td>{cells}</tr>")
    ai_html = ai_text.replace("\n", "<br>")
    return f"""
    <div style="font-family:sans-serif;max-width:960px">
      <h2>超跌反弹监控报告 · {dt.date.today()}</h2>
      <p>命中连续下跌标的 <b>{len(hits)}</b> 个 ｜ 参考财经快讯 {news_count} 条</p>
      <div style="overflow-x:auto">
      <table border="1" cellspacing="0" cellpadding="6"
             style="border-collapse:collapse;font-size:13px;white-space:nowrap">
        <tr style="background:#f2f2f2">
          <th>名称</th><th>代码</th><th>连跌<br>天数</th><th>最新价</th><th>RSI</th>{win_headers}
        </tr>{rows}
      </table></div>
      <p style="font-size:12px;color:#999">带 * 的窗口表示历史数据不足，按可得最早数据计算</p>
      <h3>AI 综合分析（结合新闻）</h3>
      <div style="background:#fafafa;padding:12px;border-radius:6px;line-height:1.7">{ai_html}</div>
      <p style="color:#999;font-size:12px;margin-top:16px">
        本报告由自动化脚本生成，仅供研究参考，不构成投资建议。连续下跌不必然反弹，请自行核实并控制风险。
      </p>
    </div>"""


def send_mail(subject: str, html: str):
    import smtplib
    from email.mime.text import MIMEText
    from email.header import Header
    msg = MIMEText(html, "html", "utf-8")
    msg["From"] = MAIL_FROM
    msg["To"] = ",".join(MAIL_TO)
    msg["Subject"] = Header(subject, "utf-8")
    with smtplib.SMTP_SSL(MAIL_HOST, MAIL_PORT) as s:
        s.login(MAIL_FROM, MAIL_PASS)
        s.sendmail(MAIL_FROM, MAIL_TO, msg.as_string())
    print("[ok] 邮件已发送")


def main():
    today = dt.date.today()
    if SKIP_NON_TRADE_DAY and not is_trade_day(today):
        print(f"[skip] {today} 非交易日，退出")
        return
    hits = scan()
    if not hits:
        print("[info] 今日无连续下跌标的，不发报告")
        return
    market_news = fetch_market_news()
    stock_news = fetch_stock_news(hits)
    print(f"[info] 抓取到市场新闻 {len(market_news)} 条，个股新闻 {len(stock_news)} 条")
    ai_text = analyze_with_llm(hits, market_news, stock_news)
    html = build_html(hits, ai_text, len(market_news))
    subject = f"{MAIL_SUBJECT_PREFIX}{today} 命中{len(hits)}个超跌标的"
    send_mail(subject, html)


if __name__ == "__main__":
    main()
