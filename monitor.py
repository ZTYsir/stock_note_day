# -*- coding: utf-8 -*-
"""
超跌反弹监控 agent（增强版）
- 监控给定标的，检测连续下跌/上涨
- 每标的输出：连涨跌天数、多空排列、均线穿越、多窗口累计涨跌幅、RSI、最新价
- 只对连续下跌达阈值的标的做 LLM 超跌反弹分析
- 额外：自抓指数+板块+新闻，交给 LLM 做「今日大盘复盘 + 次日开盘提醒」
- 交易日定时发邮件

数据源: 5 源容错(东财/新浪/腾讯/雅虎/Tushare)，海外 runner 上自动让能通的源优先
LLM: DeepSeek(OpenAI 兼容接口，纯离线，联网信息靠本脚本自抓后喂入)
Secret: LLM_API_KEY(必填) / MAIL_PASS(必填) / TUSHARE_TOKEN(可选，配了才启用 Tushare 源)
"""
import os
import time
import datetime as dt

import pandas as pd
import requests

# ============================ 配置区（改这里）============================
# ---- 1. 监控标的：代码 -> (名称, 类型)  类型 "etf" 或 "stock" ----
WATCH_LIST = {
    "518800": ("黄金板块",    "stock"),
    "510880": ("红利ETF",     "etf"),
    "512800": ("银行ETF",     "etf"),
    "512880": ("证券ETF",     "etf"),
    "513100": ("纳指ETF",     "etf"),
    "560580": ("电力ETF",     "etf"),
    "159559": ("机器人etf",   "etf"),
    "159558": ("半导体etf",   "etf"),
    "159583": ("通信etf",     "etf"),
    "601127": ("赛力斯",      "stock"),
    "600900": ("长江电力",    "stock"),
    "600036": ("招商银行",    "stock")
}

# ---- 2. 数据源（从左到右依次尝试，前面失败自动切后面）----
DATA_SOURCES = ["akshare_em", "akshare_sina", "akshare_tx", "yfinance", "tushare"]

# ---- 3. 触发：连续下跌达到这些天数才做超跌分析（取命中的最大档）----
DOWN_THRESHOLDS = [3, 5, 7]

# ---- 4. 累计涨跌幅统计窗口（交易日根数）----
#   当日 = 最新一根相对前一根
PCT_WINDOWS = {
    "当日": 1, "5天": 5, "15天": 15, "30天": 30,
    "3月": 63, "6月": 126, "1年": 252, "3年": 756,
}
LOOKBACK_DAYS = 800          # 覆盖 3 年(约756交易日)，日历天需放大
RSI_PERIOD    = 14
ADJUST        = "qfq"        # akshare 复权: "" / "qfq" / "hfq"

# ---- 均线周期（多空排列 / 穿越判定）----
MA_PERIODS = [5, 10, 20]

# ---- 整行加粗条件 ----
BOLD_DOWN_DAYS   = 5         # 连跌 >= 5 个交易日
BOLD_3D_DROP_PCT = -20.0     # 近3日累计跌幅 <= -20%

# ---- 5. 财经新闻源 ----
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
]
NEWS_PER_SOURCE   = 5
NEWS_TOTAL_LIMIT  = 40       # 大盘复盘要更多料，放大
ENABLE_STOCK_NEWS = True
STOCK_NEWS_PER    = 3

# ---- 大盘指数（复盘用）代码 -> 名称 ----
INDEX_LIST = {
    "sh000001": "上证指数",
    "sz399001": "深证成指",
    "sz399006": "创业板指",
    "sh000688": "科创50",
    "sh000300": "沪深300",
}
BOARD_TOP_N = 5              # 板块涨/跌各取前 N

# ---- 6. LLM（DeepSeek，OpenAI 兼容）----
LLM_BASE_URL = "https://api.deepseek.com/chat/completions"
LLM_MODEL    = "deepseek-chat"
LLM_TIMEOUT  = 180

# ---- 7. 邮件 ----
MAIL_HOST = "smtp.qq.com"
MAIL_PORT = 465
MAIL_FROM = "974808867@qq.com"
MAIL_TO   = ["974808867@qq.com"]
MAIL_SUBJECT_PREFIX = "【超跌反弹监控】"

SKIP_NON_TRADE_DAY = True
REQUEST_SLEEP = 0.5
FETCH_RETRY   = 2
# =====================================================================

LLM_API_KEY   = os.environ.get("LLM_API_KEY", "")
MAIL_PASS     = os.environ.get("MAIL_PASS", "")
TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "")

# GitHub Actions(海外 runner)上国内源基本被拒，自动让海外源优先
if os.environ.get("GITHUB_ACTIONS") == "true":
    DATA_SOURCES = ["yfinance", "tushare", "akshare_em", "akshare_sina", "akshare_tx"]
    print("[info] 检测到 GitHub Actions 环境，数据源优先级切为海外源优先")


def _today_str() -> str:
    return dt.date.today().strftime("%Y%m%d")


def _is_sh(code: str) -> bool:
    return code.startswith(("5", "6", "9", "11", "68"))


def is_trade_day(today: dt.date) -> bool:
    try:
        import akshare as ak
        cal = ak.tool_trade_date_hist_sina()
        return today in set(pd.to_datetime(cal["trade_date"]).dt.date)
    except Exception as e:
        print(f"[warn] 交易日历获取失败，默认按交易日处理: {e}")
        return True


# ==================== 数据源（统一返回 date/close 升序）====================
def _std(df: pd.DataFrame, date_col: str, close_col: str) -> pd.DataFrame:
    df = df[[date_col, close_col]].rename(columns={date_col: "date", close_col: "close"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna().sort_values("date").reset_index(drop=True)


def _src_akshare_em(code, atype):
    import akshare as ak
    s = (dt.date.today() - dt.timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    e = _today_str()
    if atype == "etf":
        df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=s, end_date=e, adjust=ADJUST)
    else:
        df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=s, end_date=e, adjust=ADJUST)
    return _std(df, "日期", "收盘")


def _src_akshare_sina(code, atype):
    import akshare as ak
    sym = ("sh" if _is_sh(code) else "sz") + code
    df = ak.stock_zh_a_daily(symbol=sym, adjust="qfq")
    return _std(df, "date", "close").tail(760).reset_index(drop=True)


def _src_akshare_tx(code, atype):
    import akshare as ak
    sym = ("sh" if _is_sh(code) else "sz") + code
    s = (dt.date.today() - dt.timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    df = ak.stock_zh_a_hist_tx(symbol=sym, start_date=s, end_date=_today_str(), adjust="qfq")
    return _std(df, "date", "close")


def _src_yfinance(code, atype):
    import yfinance as yf
    sym = code + (".SS" if _is_sh(code) else ".SZ")
    hist = yf.Ticker(sym).history(period="3y", auto_adjust=True)
    if hist is None or hist.empty:
        raise ValueError(f"雅虎无数据({sym})")
    return _std(hist.reset_index(), "Date", "Close")


def _src_tushare(code, atype):
    if not TUSHARE_TOKEN:
        raise ValueError("未配置 TUSHARE_TOKEN")
    import tushare as ts
    pro = ts.pro_api(TUSHARE_TOKEN)
    ts_code = code + (".SH" if _is_sh(code) else ".SZ")
    s = (dt.date.today() - dt.timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    e = _today_str()
    raw = (pro.fund_daily(ts_code=ts_code, start_date=s, end_date=e) if atype == "etf"
           else pro.daily(ts_code=ts_code, start_date=s, end_date=e))
    return _std(raw, "trade_date", "close")


SOURCE_FUNCS = {
    "akshare_em": _src_akshare_em,
    "akshare_sina": _src_akshare_sina,
    "akshare_tx": _src_akshare_tx,
    "yfinance": _src_yfinance,
    "tushare": _src_tushare,
}


def fetch_hist(code, asset_type):
    errors = []
    for source in DATA_SOURCES:
        fn = SOURCE_FUNCS.get(source)
        if fn is None:
            continue
        for attempt in range(FETCH_RETRY):
            try:
                df = fn(code, asset_type)
                if len(df) >= RSI_PERIOD + 2:
                    if source != DATA_SOURCES[0]:
                        print(f"[info] {code} 由备用源 {source} 取得")
                    return df
                errors.append(f"{source}:数据过短({len(df)})")
                break
            except Exception as ex:
                errors.append(f"{source}#{attempt+1}:{str(ex)[:60]}")
                time.sleep(REQUEST_SLEEP)
    raise RuntimeError(" | ".join(errors))


# ==================== 计算 ====================
def consecutive_down_days(closes):
    n = 0
    for i in range(len(closes) - 1, 0, -1):
        if closes.iloc[i] < closes.iloc[i - 1]:
            n += 1
        else:
            break
    return n


def consecutive_up_days(closes):
    n = 0
    for i in range(len(closes) - 1, 0, -1):
        if closes.iloc[i] > closes.iloc[i - 1]:
            n += 1
        else:
            break
    return n


def trend_streak(closes):
    """返回 (方向, 天数)：方向 'down'/'up'/'flat'。连续涨跌互斥，只有一个非零。"""
    down = consecutive_down_days(closes)
    if down > 0:
        return "down", down
    up = consecutive_up_days(closes)
    if up > 0:
        return "up", up
    return "flat", 0


def window_pcts(closes):
    last = closes.iloc[-1]
    out = {}
    for label, k in PCT_WINDOWS.items():
        if len(closes) > k:
            base = closes.iloc[-(k + 1)]
            out[label] = round((last - base) / base * 100, 2)
        elif len(closes) >= 2:
            out[label + "*"] = round((last - closes.iloc[0]) / closes.iloc[0] * 100, 2)
        else:
            out[label] = None
    return out


def calc_rsi(closes, period=14):
    delta = closes.diff()
    ag = delta.clip(lower=0).ewm(alpha=1 / period, min_periods=period).mean()
    al = (-delta.clip(upper=0)).ewm(alpha=1 / period, min_periods=period).mean()
    rsi = 100 - 100 / (1 + ag / al.replace(0, 1e-9))
    return round(float(rsi.iloc[-1]), 1) if not rsi.dropna().empty else float("nan")


def ma_alignment(closes):
    """多空排列：多头(MA5>MA10>MA20) / 空头(MA5<MA10<MA20) / 纠缠(其余)"""
    if len(closes) < max(MA_PERIODS):
        return "数据不足"
    mas = [float(closes.rolling(p).mean().iloc[-1]) for p in MA_PERIODS]  # [MA5, MA10, MA20]
    if mas[0] > mas[1] > mas[2]:
        return "多头"
    if mas[0] < mas[1] < mas[2]:
        return "空头"
    return "纠缠"


def ma_cross(closes):
    """
    均线穿越：比较昨收/今收相对各均线的位置。
    同一天可能穿多条，取周期最长的那条显示（突破更强）。
    返回如 '上穿20日线' / '跌破5日线' / '—'
    """
    if len(closes) < max(MA_PERIODS) + 1:
        return "—"
    today = float(closes.iloc[-1])
    yest = float(closes.iloc[-2])
    hit = None
    for p in MA_PERIODS:                       # 5,10,20 顺序遍历，后命中的覆盖前者→最终留最长周期
        ma_series = closes.rolling(p).mean()
        ma_today = float(ma_series.iloc[-1])
        ma_yest = float(ma_series.iloc[-2])
        if pd.isna(ma_today) or pd.isna(ma_yest):
            continue
        if yest <= ma_yest and today > ma_today:
            hit = f"上穿{p}日线"
        elif yest >= ma_yest and today < ma_today:
            hit = f"跌破{p}日线"
    return hit if hit else "—"


def scan():
    """扫描所有标的。is_hit 标记连跌达阈值；is_bold 标记整行加粗条件。"""
    results, failed = [], []
    for code, (name, atype) in WATCH_LIST.items():
        try:
            df = fetch_hist(code, atype)
            closes = df["close"]
            direction, streak = trend_streak(closes)
            threshold = max([d for d in DOWN_THRESHOLDS if direction == "down" and streak >= d], default=0)
            pcts = window_pcts(closes)

            # 整行加粗：连跌>=5 或 近3日累计跌幅<=-20%
            drop_3d = pcts.get("15天")  # 占位，真正3日在下面单独算
            pct_3d = None
            if len(closes) > 3:
                pct_3d = round((closes.iloc[-1] - closes.iloc[-4]) / closes.iloc[-4] * 100, 2)
            is_bold = (direction == "down" and streak >= BOLD_DOWN_DAYS) or \
                      (pct_3d is not None and pct_3d <= BOLD_3D_DROP_PCT)

            results.append({
                "code": code, "name": name,
                "direction": direction, "streak": streak,
                "is_hit": threshold > 0, "threshold": threshold,
                "is_bold": is_bold, "pct_3d": pct_3d,
                "align": ma_alignment(closes),
                "cross": ma_cross(closes),
                "last_close": round(float(closes.iloc[-1]), 3),
                "rsi": calc_rsi(closes, RSI_PERIOD),
                "pcts": pcts,
            })
            tag = "hit" if threshold > 0 else "ok"
            print(f"[{tag}] {name}({code}) {direction}{streak} 排列:{results[-1]['align']} "
                  f"穿越:{results[-1]['cross']}" + ("  ★达阈值" if threshold > 0 else ""))
        except Exception as ex:
            failed.append((code, name, str(ex)))
            print(f"[fail] {name}({code}) 所有数据源失败: {ex}")
    return results, failed


# ==================== 大盘指数 & 板块（复盘原料）====================
def fetch_indices():
    """抓主要指数今日涨跌，返回文本列表。海外 runner 上可能失败。"""
    import akshare as ak
    out = []
    try:
        spot = ak.stock_zh_index_spot_em(symbol="指数成份")
    except Exception:
        spot = None
    for code, name in INDEX_LIST.items():
        try:
            df = ak.stock_zh_index_daily(symbol=code)
            df = df.tail(2).reset_index(drop=True)
            if len(df) >= 2:
                chg = (df["close"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2] * 100
                out.append(f"{name}: {df['close'].iloc[-1]:.2f}（{chg:+.2f}%）")
        except Exception as ex:
            print(f"[index warn] {name} 失败: {ex}")
    return out


def fetch_board_ranking():
    """抓行业板块涨跌榜，返回 (涨前N, 跌前N) 两个文本列表。海外 runner 上多半失败。"""
    import akshare as ak
    try:
        df = ak.stock_board_industry_name_em()
        # 该接口含「板块名称」「涨跌幅」等列
        col_name = next((c for c in df.columns if "名称" in str(c)), None)
        col_chg = next((c for c in df.columns if "涨跌幅" in str(c)), None)
        if not col_name or not col_chg:
            return [], []
        df = df[[col_name, col_chg]].copy()
        df[col_chg] = pd.to_numeric(df[col_chg], errors="coerce")
        df = df.dropna()
        top_up = df.sort_values(col_chg, ascending=False).head(BOARD_TOP_N)
        top_down = df.sort_values(col_chg, ascending=True).head(BOARD_TOP_N)
        up = [f"{r[col_name]} {r[col_chg]:+.2f}%" for _, r in top_up.iterrows()]
        down = [f"{r[col_name]} {r[col_chg]:+.2f}%" for _, r in top_down.iterrows()]
        return up, down
    except Exception as ex:
        print(f"[board warn] 板块榜获取失败: {ex}")
        return [], []


# ==================== 新闻 ====================
def _extract_titles(df, n):
    if df is None or len(df) == 0:
        return []
    tc = [c for c in df.columns if any(k in str(c) for k in ["标题", "内容", "title", "摘要", "简介"])]
    col = tc[0] if tc else df.columns[0]
    return df[col].dropna().astype(str).str.strip().head(n).tolist()


def fetch_market_news():
    import akshare as ak
    today = _today_str()
    collected = []
    for src in NEWS_SOURCES:
        if len(collected) >= NEWS_TOTAL_LIMIT:
            break
        try:
            params = {k: (v.replace("{today}", today) if isinstance(v, str) else v)
                      for k, v in src.get("params", {}).items()}
            df = getattr(ak, src["func"])(**params)
            collected += [f"[{src['label']}] {t}" for t in _extract_titles(df, NEWS_PER_SOURCE)]
            time.sleep(REQUEST_SLEEP)
        except Exception as ex:
            print(f"[news warn] {src['func']} 失败: {ex}")
    return collected[:NEWS_TOTAL_LIMIT]


def fetch_stock_news(hits):
    if not ENABLE_STOCK_NEWS:
        return []
    import akshare as ak
    out = []
    for h in hits:
        try:
            df = ak.stock_news_em(symbol=h["code"])
            out += [f"[{h['name']}] {t}" for t in _extract_titles(df, STOCK_NEWS_PER)]
            time.sleep(REQUEST_SLEEP)
        except Exception as ex:
            print(f"[news warn] 个股新闻 {h['code']} 失败: {ex}")
    return out


# ==================== LLM ====================
def _llm_call(prompt):
    if not LLM_API_KEY:
        return "（未配置 LLM_API_KEY，跳过 AI 分析）"
    try:
        r = requests.post(LLM_BASE_URL,
                          headers={"Authorization": f"Bearer {LLM_API_KEY}",
                                   "Content-Type": "application/json"},
                          json={"model": LLM_MODEL, "temperature": 0.3,
                                "messages": [{"role": "user", "content": prompt}]},
                          timeout=LLM_TIMEOUT)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as ex:
        return f"（LLM 调用失败：{ex}）"


def analyze_oversold(hits, market_news, stock_news):
    """对连跌标的做超跌反弹分析"""
    lines = []
    for h in hits:
        pct = " ".join(f"{k}:{v}%" for k, v in h["pcts"].items() if v is not None)
        lines.append(f"- {h['name']}({h['code']})：连跌{h['streak']}天，RSI(14)={h['rsi']}，"
                     f"排列{h['align']}，{h['cross']}，最新价{h['last_close']}；累计涨跌幅 {pct}")
    prompt = (
        "你是一名量化投资助手。以下是今日筛选出的、已连续下跌的标的（银行/红利/证券/纳指等，"
        "长期看具均值回归特征）及其技术指标：\n" + "\n".join(lines) +
        "\n\n【今日市场财经快讯】\n" + ("\n".join(market_news) or "（无）") +
        "\n\n【相关个股新闻】\n" + ("\n".join(stock_news) or "（无）") +
        "\n\n请结合新闻，从『超跌反弹』角度逐个分析：(1)超跌程度(连跌天数+累计跌幅+RSI是否超卖+均线排列)；"
        "(2)结合新闻判断是普跌情绪还是针对性重大利空(利空会削弱反弹逻辑)；"
        "(3)反弹概率定性评估与操作建议(观望/分批试探/暂不介入)。最后给整体风险提示。中文，简洁，可用短列表。"
    )
    return _llm_call(prompt)


def analyze_market_review(indices, board_up, board_down, market_news):
    """今日大盘复盘 + 次日开盘提醒"""
    prompt = (
        "你是一名资深A股策略分析师。请基于以下今日客观数据，撰写一份专业、翔实的"
        "『今日大盘复盘 + 次日开盘提醒』。要求内容详实、有逻辑、有观点，可用小标题和列表。\n\n"
        "【今日主要指数】\n" + ("\n".join(indices) or "（未获取到指数数据，请基于新闻与常识审慎推断，并注明数据缺失）") +
        "\n\n【今日行业板块涨幅前五】\n" + ("\n".join(board_up) or "（未获取到板块数据）") +
        "\n\n【今日行业板块跌幅前五】\n" + ("\n".join(board_down) or "（未获取到板块数据）") +
        "\n\n【今日国内外财经资讯】\n" + ("\n".join(market_news) or "（无）") +
        "\n\n请严格按以下七个部分组织，每部分都要展开、给出具体判断而非空话：\n"
        "1. 盘面总览：今日整体表现、涨跌家数氛围、量能变化的定性判断。\n"
        "2. 指数结构分析：上证/深成/创业板/科创/沪深300 之间的强弱分化说明了什么。\n"
        "3. 板块主线分析：必须明确列出今日涨幅前五与跌幅前五板块，并分析资金主线与轮动方向。\n"
        "4. 资金情绪分析：结合板块与量能，判断市场情绪(亢奋/谨慎/恐慌)与主力资金取向。\n"
        "5. 消息催化分析：从今日财经资讯中提炼真正影响市场的催化剂(政策/产业/海外)。\n"
        "6. 明日操作建议：给出次日开盘的具体应对思路(仓位、方向、关注点)。\n"
        "7. 风险识别：列出次日需警惕的主要风险点。\n"
        "中文撰写，专业但不堆砌术语，结论要落地。"
    )
    return _llm_call(prompt)


# ==================== 邮件 ====================
def _send(subject, html):
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


def _fmt_streak(direction, streak):
    """连涨跌列：红涨绿跌。跌→绿，涨→红，平→灰"""
    if direction == "down" and streak > 0:
        return f"<span style='color:#27ae60'>跌{streak}</span>"
    if direction == "up" and streak > 0:
        return f"<span style='color:#c0392b'>涨{streak}</span>"
    return "<span style='color:#999'>—</span>"


def send_report(results, hit_items, oversold_text, market_text, news_count, failed):
    win_headers = "".join(f"<th>{k}</th>" for k in PCT_WINDOWS.keys())

    def cell(v):
        # 红涨绿跌
        if v is None:
            return "<td style='text-align:right;color:#bbb'>-</td>"
        color = "#27ae60" if v < 0 else "#c0392b"
        return f"<td style='text-align:right;color:{color}'>{v}%</td>"

    rows = ""
    for h in results:
        cells = "".join(cell(h["pcts"].get(k, h["pcts"].get(k + "*"))) for k in PCT_WINDOWS)
        row_style = "font-weight:bold" if h["is_bold"] else ""   # 只加粗，不加底色
        rows += (
            f"<tr style='{row_style}'>"
            f"<td>{h['name']}</td>"
            f"<td>{h['code']}</td>"
            f"<td style='text-align:center'>{h['align']}</td>"
            f"<td style='text-align:center'>{h['cross']}</td>"
            f"<td style='text-align:center'>{_fmt_streak(h['direction'], h['streak'])}</td>"
            f"{cells}"
            f"<td style='text-align:center'>{h['rsi']}</td>"
            f"<td style='text-align:right'>{h['last_close']}</td>"
            f"</tr>"
        )

    fail_note = ""
    if failed:
        items = "".join(f"<li>{n}({c})：{e[:80]}</li>" for c, n, e in failed)
        fail_note = (f"<p style='color:#e67e22'>⚠ 有 {len(failed)} 个标的未取到数据：</p>"
                     f"<ul style='color:#e67e22;font-size:12px'>{items}</ul>")

    if hit_items:
        oversold_section = (
            f"<h3>AI 超跌反弹分析（{len(hit_items)} 个连跌标的，结合 {news_count} 条快讯）</h3>"
            f"<div style='background:#fafafa;padding:12px;border-radius:6px;line-height:1.7'>"
            f"{oversold_text.replace(chr(10),'<br>')}</div>")
    else:
        oversold_section = ("<p style='color:#888'>今日无标的达到连续下跌阈值"
                            f"（≥{min(DOWN_THRESHOLDS)}天），暂无超跌反弹分析。</p>")

    market_section = (
        "<h3>📊 今日大盘复盘 &amp; 次日开盘提醒</h3>"
        f"<div style='background:#f5f7fa;padding:14px;border-radius:6px;line-height:1.8'>"
        f"{market_text.replace(chr(10),'<br>')}</div>")

    html = f"""
    <div style="font-family:sans-serif;max-width:1080px">
      <h2>超跌反弹监控 · 全景报告 · {dt.date.today()}</h2>
      <p>监控标的 <b>{len(results)}</b> 个 ｜ 连续下跌达阈值 <b style="color:#27ae60">{len(hit_items)}</b> 个（加粗行）</p>
      {fail_note}
      {market_section}
      <div style="overflow-x:auto;margin-top:16px">
      <table border="1" cellspacing="0" cellpadding="6"
             style="border-collapse:collapse;font-size:13px;white-space:nowrap">
        <tr style="background:#f2f2f2">
          <th>名称</th><th>代码</th><th>多空<br>排列</th><th>均线<br>穿越</th><th>连涨跌</th>
          {win_headers}<th>RSI</th><th>最新价</th>
        </tr>{rows}
      </table></div>
      <p style="font-size:12px;color:#999">
        颜色：<span style="color:#c0392b">红=涨</span> / <span style="color:#27ae60">绿=跌</span>；
        加粗行 = 连跌≥{BOLD_DOWN_DAYS}天 或 近3日累计跌幅≤{BOLD_3D_DROP_PCT}%；
        带 * 的窗口表示历史数据不足，按可得最早数据计算
      </p>
      {oversold_section}
      <p style="color:#999;font-size:12px;margin-top:16px">
        本报告由自动化脚本生成，仅供研究参考，不构成投资建议。连续下跌不必然反弹，请自行核实并控制风险。
      </p>
    </div>"""
    _send(f"{MAIL_SUBJECT_PREFIX}{dt.date.today()} 全景({len(results)}标的/{len(hit_items)}连跌)", html)


def send_data_alert(failed):
    items = "".join(f"<li>{n}({c})：{e[:120]}</li>" for c, n, e in failed)
    html = f"""
    <div style="font-family:sans-serif;max-width:720px">
      <h2 style="color:#c0392b">⚠ 数据源异常，今日未能获取任何行情</h2>
      <p>全部 {len(failed)} 个标的的所有数据源都失败了，<b>这不代表今日无信号，而是没抓到数据</b>。
         常见原因：GitHub 海外 runner 被国内源拒连、雅虎限流。可稍后手动重跑，或调整配置区 DATA_SOURCES / 配置 TUSHARE_TOKEN。</p>
      <ul style="font-size:12px;color:#c0392b">{items}</ul>
    </div>"""
    _send(f"{MAIL_SUBJECT_PREFIX}⚠ 数据源异常 {dt.date.today()}", html)


def main():
    today = dt.date.today()
    if SKIP_NON_TRADE_DAY and not is_trade_day(today):
        print(f"[skip] {today} 非交易日，退出")
        return

    results, failed = scan()
    total = len(WATCH_LIST)

    if len(failed) == total:
        print(f"[error] 全部 {total} 个标的数据获取失败，发告警邮件")
        send_data_alert(failed)
        return
    if failed:
        print(f"[warn] {len(failed)}/{total} 个标的数据失败，其余照常入报告")

    hit_items = [r for r in results if r["is_hit"]]

    # 大盘复盘原料：指数 + 板块 + 新闻（无论有无连跌标的都做复盘）
    print("[info] 抓取大盘复盘原料 ...")
    indices = fetch_indices()
    board_up, board_down = fetch_board_ranking()
    market_news = fetch_market_news()
    print(f"[info] 指数{len(indices)}条，板块涨/跌各{len(board_up)}/{len(board_down)}，新闻{len(market_news)}条")

    # 超跌分析（仅连跌标的）
    if hit_items:
        stock_news = fetch_stock_news(hit_items)
        print(f"[info] {len(hit_items)}个连跌标的，个股新闻{len(stock_news)}条")
        oversold_text = analyze_oversold(hit_items, market_news, stock_news)
    else:
        print("[info] 无标的达连跌阈值，跳过超跌分析")
        oversold_text = ""

    # 大盘复盘
    market_text = analyze_market_review(indices, board_up, board_down, market_news)

    send_report(results, hit_items, oversold_text, market_text, len(market_news), failed)


if __name__ == "__main__":
    main()
