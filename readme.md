# 超跌反弹监控 agent

监控给定标的（银行 / 红利 / 证券 / 纳指 / 宽基 ETF 等，可自定义），
检测连续下跌，只对连跌标的用 LLM 结合财经新闻做超跌反弹分析，交易日定时发邮件。

## 功能
1. 每交易日收盘后运行，用 akshare 拉取各标的日线。
2. 计算连续下跌天数 + 多窗口累积跌幅（**1/3/5/7/14 天、1月、3月、半年、1年**）+ RSI(14)。
3. 连跌达到 3/5/7 天阈值才触发分析。
4. 多源冗余抓取财经新闻（财联社/东财/新浪/富途/同花顺/财经早餐/新闻联播等 10 个源，
   防止单一源在海外 runner 失效），连同个股新闻一起交给 LLM。
5. LLM 结合新闻判断是普跌情绪还是针对性利空，给出反弹概率与操作建议。
6. 生成带涨跌幅红绿着色的 HTML 表格报告，发到邮箱。非交易日自动跳过。

## 配置（都在 monitor.py 顶部配置区）
- `WATCH_LIST` 标的列表
- `PCT_WINDOWS` 累积跌幅统计窗口
- `NEWS_SOURCES` 新闻源（10 个，参数可增删）
- `LLM_BASE_URL` / `LLM_MODEL` 切换千问 / DeepSeek（只改两行）
- `MAIL_HOST/PORT/FROM/TO` 邮件收发与 SMTP

## 两个必需的 Secret（不能写死在代码里）
| Secret | 说明 |
|--------|------|
| `LLM_API_KEY` | 千问 DashScope 或 DeepSeek 的 key（OpenAI 兼容接口通用）|
| `MAIL_PASS`   | 发件邮箱 SMTP 授权码（QQ 邮箱在"设置-账户"里开启获取）|

> 邮箱授权码也必须放 Secret：明文提交到公开仓库会被盗用发垃圾邮件。

## 本地跑
```bash
pip install -r requirements.txt
export LLM_API_KEY=sk-xxx
export MAIL_PASS=xxxx
python monitor.py
```

## 发布到 GitHub Actions
1. 新建仓库，推送本项目（含 `.github/workflows/monitor.yml`）。
2. Settings → Secrets and variables → Actions，添加 `LLM_API_KEY`、`MAIL_PASS`。
3. 默认工作日北京时间 15:05 触发（cron `5 7 * * 1-5`，UTC）。
   也可在 Actions 页面 `Run workflow` 手动测试。

## 注意
- Actions cron 高峰期可能延迟十几分钟，对收盘后运行无影响。
- akshare 走国内数据源，GitHub 海外 runner 偶发限流；新闻多源冗余就是为此，
  若行情接口也不稳，可改放国内小服务器跑 crontab，代码不用改。
- 连续下跌不必然反弹，趋势性下跌中超跌后可能继续跌，本项目仅供研究参考。
