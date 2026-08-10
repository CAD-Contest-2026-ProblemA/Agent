# Routing 準確率:改動與實驗結果

分支 `feat/routing-accuracy`,20 個 commit,17 個檔案,+3664 / −20 行。

**一句話:自然語言請求分派到正確 op 的成功率,從 86.9% 提升到 95.3%(gpt-4o-mini)
/ 97.8%(claude-haiku-4-5)。出貨路徑零新依賴,binary 維持 45MB。**

---

## 1. 對照三份既有 excel 的基準

| 測資 | excel 記錄的基準 | 改動後 | 提升 |
|---|---|---|---|
| `routing_test_101_171`(1420 句)| 純 LLM **86.9%** | **95.3%**(gpt-4o-mini)| **+8.4pt** |
| | | **97.8%**(claude-haiku-4-5)| **+10.9pt** |
| `routing_test_91_100`(710 句)| 純 LLM **92.4%** / 混合 91.5% | **98.9%** | **+6.5pt** |
| `routing_test_30`(30 句)| 純 LLM 96.7% / 混合 100% | 未測 | — |

失敗的組成變化(1420 句):

| | 放棄(noop/無輸出)| 選錯 op | 合計 |
|---|---|---|---|
| 基準 | 107 | 79 | 186 |
| gpt-4o-mini | 12 | 55 | 67 |
| claude-haiku-4-5 | 3 | 28 | 31 |

**「放棄」降 97%,「選錯」降 65%。** 兩種失敗成因不同,解法也不同 —— 例句檢索治
「沒認出這種問法」,消歧規則與模型能力治「分不開兩個近義 op」。

### 三個必須說清楚的限制

1. **`routing_test_30` 那 30 句沒測。** 它們來自 test01,不在例句庫裡;而且混合路徑
   本來就是 100%,沒有提升空間。
2. **`test91-100` 的 98.9% 偏樂觀。** 該組每個 op 橫跨 10 個 testcase,
   leave-one-testcase-out 排掉自己之後,姊妹 testcase 仍供得出同 op 的例句。
   `test101-171` 是一個 op 只在一個 testcase 裡 —— **95.3% 才是誠實的數字**。
   而且 98.9% 是在 k=25、第二輪消歧前、去重前量的,現在只會更好,未重測。
3. **excel 基準是哪個模型跑的沒有記錄。** 我的數字全部是 gpt-4o-mini(config 預設)。
   若基準當初用更強的模型,+8.4pt 會被高估。要消除這個疑慮,需用 `main` 的 catalog
   在同一模型上重跑一次基準(約 $0.15 / 15 分鐘),尚未做。

---

## 2. 改了什麼

### 2.1 量測基礎設施(先決條件)

改動前 repo 裡**沒有辦法重跑基準** —— `route_check.py` 只驗 regex 層,純 LLM 的結果
是另外產的,腳本沒進版控,ground truth 只存在 xlsx 裡。

| 檔案 | 內容 |
|---|---|
| `cada/llm/examples.jsonl` | 2130 筆標好的 (句子 → op),從兩份 xlsx 匯出 |
| `scripts/export_examples.py` | 用 stdlib 解 xlsx(zip + XML),零第三方套件;驗證每個標籤都在 `ALLOWED_INTENTS` |
| `scripts/route_eval.py` | 純 LLM routing 評估。`--retriever` / `--top-k` / `--sample` / `--provider` / leave-one-testcase-out;失敗拆成「放棄」與「選錯」兩類 |

### 2.2 Catalog 文字修補(`cada/llm/allowed_intents.py`)

**四個 op 連一則 few-shot 都沒有**,而它們正是「放棄」最多的:`symmetric`(13)、
`is_cut`(9)、`count_ports`(6)、`same_clock`(5)。各補 3 則,取自 test91-100 ——
**不碰 test101-171**,否則量到的是背答案而不是路由。

**20 組消歧區塊**(兩輪):第一輪 15 組,涵蓋 xlsx 裡重複出現的混淆配對;第二輪 5 組,
針對檢索之後仍存活的配對。消歧區塊從 17 組增加到 30 組。

其中兩組第一版寫錯了方向,是測出來才修正的:

- `dominator` vs `path_exists` —— 兩者都是「兩個端點 + 第三個要繞過的東西」,
  措辭完全重疊。真正的判別依據是**第三個名字是 gate 還是 net**。
- `connected_to_output` vs `successors` —— 兩者都吃 gate、回傳同一個 walk,
  只能靠詞彙分(鄰接/順序 vs 接腳/負載)。

**還修掉一條有害的既有規則**:catalog 原本寫「clock net 的 loads 就是它 clock 的
flip-flops,算 fanout」。2130 筆標註裡**零句**支撐這個說法,而它正是
`ffs_on_clock → fanout` 的肇因。

### 2.3 例句檢索(`cada/llm/retrieval.py`、`fallback.py`、`client.py`)

catalog 只放得下 79 則手寫範例,但例句庫有 2130 筆。改成**每次依請求檢索最相似的
50 則**貼進 prompt。

**是加法不是減法** —— 72 個 op 全清單與所有消歧規則都留在靜態段,檢索再差也只是那
50 則沒幫上忙,不存在「正確答案被丟掉」的懸崖。這正是「檢索例句而非檢索工具清單」
的理由(見 §3.1)。

Prompt 結構(快取關鍵):

```python
system=[
    {"type": "text", "text": INTENT_CATALOG,          # 固定
     "cache_control": {"type": "ephemeral"}},          # ← 斷點
    {"type": "text", "text": retrieved_examples},      # 每次不同
]
```

實測:靜態段 **10,280 tokens**,第一次 write、第二次起 **cache_read 10,274**,
每次只有 ~630 tokens 付全額。OpenAI 那條路把動態段接在 system 訊息末尾,同樣保住
自動前綴快取。

**與原計畫不同**:原本打算把 catalog 的 79 則 few-shot 縮成 30 則核心範例。實作時
全部保留 —— 它們帶 params、示範輸出格式,而檢索回來的只有「句子 → op」。純加法,
快取地基也更穩。

四種後端(`--retriever`):`bm25`(預設,零依賴)/ `onnx` / `union` / `union-rrf`,
加上比較用的 `qdrant` / `qdrant-hybrid`。任何一步失敗都退回 catalog-only,不影響請求。

### 2.4 收窄四條誤攔的 regex(`cada/agent/agent.py`)

xlsx 顯示 regex 攔到的部分正確率 97.2%,但**它的 8 個誤攔,純 LLM 全部答對** ——
這就是「看起來比較準的那層反而拉低總分」(混合 91.5% vs 純 LLM 92.4%)。

| 規則 | 問題 |
|---|---|
| `h_enum_paths` | 裸的 `enumerat` 分支吃掉任何 "Enumerate X … Y",連 `successors` / `connected_to_net` / `transitive_fanin/fanout` / `reg_to_reg_paths` 一起攔走 |
| `h_rename2` | "After that rename, what connects to renamed_sig?" 把 "connects to renamed_sig" 讀成新舊名配對 |
| `h_cone_gate_count` | 在只用 cone 限定範圍的 transform 裡誤觸 |
| `h_total` | "minimize total gate count" 裡的 "total gate count" 被當成查詢 |

全 2130 句量測:覆蓋率 574 → 554,**路由改變的 22 句原本全部都是誤攔**。2 句現在直接
命中正確 handler,20 句落到 LLM。**沒有損失任何正確路由。**

這層留著 —— 零成本、零延遲、可離線。修的是讓它在不確定時**放手**:漏接的代價是一次
LLM 呼叫,誤攔的代價是答錯。

### 2.5 打包(`scripts/spec_assets.py`、`.spec`、`requirements-retrieval.txt`)

- `data_path()` 透過 `sys._MEIPASS` 解析資料檔 —— one-file 解壓目錄與套件 `__file__`
  不同,不處理的話每次凍結執行都會靜默退回無檢索
- 打包邏輯放進版控(`scripts/spec_assets.py`),因為 `*.spec` 是 gitignore 的產物;
  spec 只留一行呼叫
- 編碼器與向量是**條件式**打包 —— 它們是 gitignore 的 build artifact,無條件寫入會讓
  新 clone 的建置直接失敗
- `requirements-retrieval.txt` 把 numpy/onnxruntime/tokenizers 排除在基本安裝之外,
  `pyproject.toml` 的「核心零第三方套件」保證仍然成立

---

## 3. 實驗結果

### 3.1 為什麼檢索例句而不是檢索工具清單

原本的構想是「為每個 op 寫 metadata,用 embedding 篩 top-k,再交給 LLM 選」。
BM25 對 1420 句量「正確 op 是否留在候選集內」:

| 檢索文件 | recall@5 | recall@10 | recall@20 |
|---|---|---|---|
| op 名 + 一行規格 | 43.1% | 53.0% | 65.4% |
| 加 few-shot + 消歧文字 | 58.1% | 66.1% | **75.1%** |

就算之後 LLM 百分之百選對,上限 75.1%,**比不做還低 12pt**。而且:

- **選錯那 42% 結構性免疫** —— 會被搞混的兩個 op 一定長得像,長得像就一定一起被撈進
  top-k,模型面對的還是同一個二選一
- **放棄那 58% 只會更糟** —— 候選集變小,「沒有一個符合」更容易成立

換成檢索**例句**(句子對句子的字面重疊遠高於句子對 op 名稱),label-recall@25 是 97.5%。

### 3.2 檢索後端六方比較(1420 句,leave-one-testcase-out,已去重)

| 後端 | 查詢 | recall@50 | 刁鑽 |
|---|---|---|---|
| `bm25`(stdlib)| **0.5ms** | 97.0% | 96.1% |
| `onnx`(精確餘弦,in-process)| 14.5ms | 98.0% | 97.3% |
| `qdrant`(dense, HNSW)| 25.5ms | **98.0%** | **97.3%** |
| `union`(交錯)| 15.4ms | **98.8%** | **98.4%** |
| `union-rrf` | 15.7ms | 98.7% | 98.3% |
| `qdrant-hybrid`(sparse+dense, engine RRF)| 39.4ms | **98.7%** | **98.3%** |

兩組對照到小數點吻合:

- **`qdrant` = `onnx`** —— 同一份向量,2130 筆下 HNSW 就等於精確搜尋,只是慢 1.8 倍
- **`qdrant-hybrid` = 自己實作的 `union-rrf`** —— 同樣的 RRF,慢 2.5 倍

Qdrant 是**向量搜尋引擎,不是編碼器**;它真正值錢的東西(百萬級 ANN、payload 過濾、
持久化、分散式)在 2130 筆上全都用不到。**RRF 也沒贏過天真交錯**(98.7 vs 98.8)。

### 3.3 端到端:召回轉化不成成功率

| 模型 | 後端 / k | 召回 | 端到端 |
|---|---|---|---|
| gpt-4o-mini | bm25 k=25 | 93.9% | 94.3% |
| gpt-4o-mini | onnx k=25 | 95.0% | 94.2% |
| gpt-4o-mini | bm25 k=50 | 96.4% | **95.3%** |
| gpt-4o-mini | onnx k=50 | 97.7% | 94.9% |
| gpt-4o-mini | union k=50 | 98.8% | 95.3% |
| gpt-4o-mini | union k=100 | **99.5%** | **94.9%** |
| haiku-4.5 | bm25 k=50 | 96.4% | 97.8% |
| haiku-4.5 | onnx k=50 | 97.7% | **98.0%** |

**召回跨度 5.6pt,端到端跨度 0.4pt,而且在 k=100 是反向的** —— 召回最高的組合
(union k=100,99.5%)端到端最差。多出來的例句是干擾項。

原因量得出來。「失敗的句子裡,正確 op 是否已經在撈回來的例句中」:

| 檢索設定 | 已經給對了卻仍答錯 |
|---|---|
| bm25 k=25 | 59/81(**73%**)|
| bm25 k=50 | 66/81(81%)|
| union k=50 | 78/81(**96%**)|
| union k=100 | 80/81(99%)|

**檢索已不是瓶頸。** 模型是分不開,不是沒看到。同理,去重修復讓召回 +0.6pt,
端到端**一句不差**(1353/1420 兩次相同)。

### 3.4 真正的槓桿是模型

同 catalog、同 bm25 k=50、同 leave-one-testcase-out:

| 模型 | 成功率 | 刁鑽 | 放棄 | 選錯 |
|---|---|---|---|---|
| gpt-4o-mini | 95.3% | 93.7% | 8 | 59 |
| **claude-haiku-4-5** | **97.8%** | **97.1%** | **3** | **28** |

**換模型 +2.5pt,換檢索後端 ±0.4pt —— 差一個數量級。**

haiku 剩下的 31 個失敗裡,**16 個(52%)是 `reachable_from ↔ transitive_fanout` 的
標註歧義** —— catalog 自己就寫明兩者回傳同一集合、只差輸出格式,句子看不出差別。
扣掉那組實際上限是 **98.9%**。

> ⚠️ 97.8% 是**去重修復之前**量的。去重後召回上升,端到端理論上不會更差,但
> Anthropic 額度在重測前用罄,**去重後的 haiku 數字沒有實測**。

---

## 4. 過程中修掉的三個 bug

**1. 429 只重試 3 次就降級成 no-op。** rate limit 不是失敗的請求,是還沒發生的請求,
但它跟一般 transient 共用 3 次、2s/4s 的退避預算。持續超限時三次全部在同一個補充視窗
內用完,`complete()` 回 None,那一行就變成 no-op ack —— **一個看起來跑完了、其實沒有的
執行**。已改成 rate limit 專用 7 次、指數退避上限 60s。

**2. 降級的跑分看起來就像成績。** 我兩次踩到同一個坑(兩個評估 job 搶同一個 provider
的額度),第二次產出一個很合理的 **67.2%** —— 同設定單獨跑是 97.8%,差別只在 458 個
放棄 vs 3 個。已加 `LLMClient.degraded` 計數,`route_eval` 只要 `degraded > 0` 就印
**INVALID RUN** 橫幅。這個保護在正式比賽跑分時同樣重要 —— 同樣的靜默會把請求變成
「已確認但沒做」。

**3. 例句庫 17% 重複,而檢索排序的是「列」不是「唯一例句」。** 2130 筆裡只有 1776 種
唯一 (句子, op);`count_gates` 那句出現 10 次。結果 top-50 平均只帶 **42.5 則**唯一
例句(最少 28)—— 約 15% 的 prompt 預算浪費在讓模型重看同一句。已在基底 `top_k`
依 `(text, op)` 去重。

另外 `union` 檢索器原本各取 k/2 再去重,要 50 則只拿到平均 39.2 則(最少 29)——
比單一後端還少,當時的比較是在量一個殘障版本。已修。

---

## 5. 回歸驗證

| 檢查 | 結果 |
|---|---|
| `scripts/verify_basic_golden.py` | **211 PASS / 4 FAIL / 244 SKIP** |
| 同樣 4 個 FAIL 在 `main` 上 | **一模一樣** —— 既有的 golden/實作差異,非本次回歸 |
| `scripts/route_check.py` | 通過 |
| 真實 Agent 跑 `testcase/test01` | 正常(載入 1794 gate、寫出、stderr 乾淨)|
| 離線管線檢查(9 項)| 全過 |
| 預設路徑載入的重量級模組 | 無 —— 純 stdlib |

---

## 6. 出貨設定與可選項

**預設(零新依賴,binary 維持 45MB):**

```
retriever = bm25   (stdlib,倒排索引,建置 0.01s / 查詢 0.5ms)
top_k     = 50     (量出來的最佳點,見 §3.3)
```

**可選(需 `requirements-retrieval.txt`):** `--retriever onnx`,約 +90MB 模型 +
onnxruntime/numpy/tokenizers,並引入 glibc / libstdc++ / SIGILL 的可攜性風險。
**實測不值得** —— 見 §3.3。

**純比較用(不出貨):** `--retriever qdrant` / `qdrant-hybrid` / `union-rrf`。

## 7. 重現方式

```bash
# 不打 API、不花錢
.venv/bin/python scripts/route_check.py
.venv/bin/python scripts/verify_basic_golden.py

# 純 LLM routing 評估
.venv/bin/python scripts/route_eval.py --sample 200 --retriever bm25   # 試水溫
.venv/bin/python scripts/route_eval.py --retriever bm25                # 全量 1420 句
.venv/bin/python scripts/route_eval.py --retriever bm25 --provider anthropic
.venv/bin/python scripts/route_eval.py --retriever union --top-k 100   # 換後端 / 調 k

# 重建檢索資產(改了 testcase 或 xlsx 之後)
.venv/bin/python scripts/export_examples.py
.venv/bin/python scripts/fetch_embed_model.py && .venv/bin/python scripts/build_vectors.py  # 只有走 onnx 才需要
```

> `route_eval` 一次只跑一個,**不要並行** —— 兩個 job 搶同一個 provider 的額度會把
> 跑分變成無效資料(現在會印 INVALID RUN,但還是浪費一輪)。

## 8. 還沒做的事

- 用 `main` 的 catalog 在 gpt-4o-mini 上重跑基準,消除「excel 基準是哪個模型」的疑慮
- 去重修復後的 haiku 端到端重測(Anthropic 額度用罄)
- `routing_test_30` 那 30 句沒納入評估
- `reachable_from ↔ transitive_fanout`(haiku 剩餘失敗的 52%)—— 是標註歧義而非語言
  歧義,寫規則只會擬合標註;要解得先回頭釐清題目定義
