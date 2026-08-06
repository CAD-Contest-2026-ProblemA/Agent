# Routing 準確率改善計畫

目標:把自然語言請求分派到正確 op 的成功率往上推。
資料來源:`routing_test_101_171.xlsx`、`routing_test_91_100.xlsx`(已在 repo 內)。

> **狀態:Phase 0–3 已實作完成**,分支 `feat/routing-accuracy`。
>
> **結論一句話:86.9% → 97.8%。** 出貨設定是 **stdlib BM25 + k=50**,零新依賴、
> binary 維持 45MB。ONNX / Qdrant / union / RRF 全部實作並量測過,端到端差距都在
> ±0.4pt(雜訊量級);**真正的槓桿是模型**(gpt-4o-mini → haiku-4.5 = +2.5pt)。
> 完整數據見 §-1,與原計畫不同的結論已就地標註。

---

## -1. 實測結果(全量 1420 句,leave-one-testcase-out)

| 階段 | 成功率 | 安全 | 刁鑽 | 放棄 | 選錯 |
|---|---|---|---|---|---|
| 原始基準 | 86.9% | 99.4% | 82.7% | 107 | 79 |
| + Phase 1 第一輪(4 個 few-shot + 15 組消歧)| ~89.5%※ | 98.1% | 86.5% | 8※ | 13※ |
| + Phase 2 檢索 BM25 k=25 | 94.3% | 100% | 92.4% | 10 | 71 |
| + Phase 2 檢索 ONNX k=25 | 94.2% | 99.7% | 92.4% | 8 | 74 |
| + k=50 | 94.6% | 100% | 92.8% | 7 | 70 |
| **+ 第二輪消歧(5 組)、BM25 k=50** | **95.3%** | **100%** | **93.7%** | **8** | **59** |

※ Phase 1 那列是 200 句等距取樣,其餘為全量 1420 句。

### 換模型才是剩下最大的槓桿

同樣的 catalog、同樣 BM25 k=50、同樣 leave-one-testcase-out,只換模型:

| 模型 | 成功率 | 安全 | 刁鑽 | 放棄 | 選錯 |
|---|---|---|---|---|---|
| gpt-4o-mini | 95.3% | 100% | 93.7% | 8 | 59 |
| **claude-haiku-4-5** | **97.8%** | **100%** | **97.1%** | **3** | **28** |

**86.9% → 97.8%,累積 +10.9 個百分點。**

> ⚠️ **這個 97.8% 是去重修復之前量的。** 去重後 bm25 的召回從 96.4% 升到 97.0%,
> 理論上端到端只會更好,但 Anthropic 額度在重測前用罄,**去重後的 haiku 數字沒有
> 實測**。gpt-4o-mini 的去重前後對照有量(見下),可作為方向參考。

而且剩下的 31 個失敗裡,**16 個(52%)就是 `reachable_from ↔ transitive_fanout`
那組標註歧義** —— 扣掉它就是 98.9%。這與「檢索已不是瓶頸、消歧才是」的診斷完全一致:
gpt-4o-mini 分不開的近義 op,haiku 分得開。

**兩個推翻原計畫的發現:**

1. **ONNX 在 k=25 和 k=50 都沒有贏。** 召回確實較好,端到端就是轉化不出來:

   | 模型 | 後端 | 召回@k | 端到端 |
   |---|---|---|---|
   | gpt-4o-mini | bm25 k=25 | 93.9% | 94.3% |
   | gpt-4o-mini | onnx k=25 | 95.0% | 94.2% |
   | gpt-4o-mini | bm25 k=50 | 96.4% | **95.3%** |
   | gpt-4o-mini | onnx k=50 | 97.7% | 94.9% |
   | gpt-4o-mini | union k=50 | **98.9%** | **95.3%** |
   | haiku-4.5 | bm25 k=50 | 96.4% | 97.8% |
   | haiku-4.5 | onnx k=50 | 97.7% | **98.0%** |

   (此表的召回是**去重前**的數字;去重後的六方比較見下方 Qdrant 段落 ——
   結論不變,只是整體上移約 0.5pt。)

   在 gpt-4o-mini 上,**召回跨度 2.5pt,端到端跨度 0.4pt 且非單調** —— 召回最高的
   union 只跟召回最低的 BM25 打平。換到 haiku 上 onnx 反超 BM25,但只差 **2 句
   (0.14pt)**,`temperature=0.2` 的重跑變異就有這個量級,**不能算贏**。

   對比之下,**換模型是 +2.5pt** —— 檢索後端的選擇比模型選擇小一個數量級。
   原計畫訂的門檻是「贏 ≥2pt 才值得那些依賴」,無論哪個模型上都沒達到。
   **預設為 BM25**(零依賴、零體積、可離線),ONNX / union 保留但需明確指定。
   連帶:§4 的 glibc / libstdc++ / SIGILL 風險全部**不必承擔**,binary 維持 45MB。

   > 若只追求最高成功率、完全不計代價:**haiku-4.5 + onnx k=50 = 98.0%**,
   > 但比 haiku + bm25 只多 2 句,而要多背 90MB 模型與整組原生依賴。

2. **瓶頸確認不在檢索器。** 檢索把「放棄」從 107 降到 10(-91%),但「選錯」只從
   79 降到 71。這與 §5.2 的預測一致:例句能教會沒認出的問法,但分不開的近義 op
   還是分不開。下一步的槓桿在消歧規則,不在換更強的模型。

---

### 檢索器 label-recall 掃描(離線,免費,leave-one-testcase-out)

| k | bm25 | onnx | union(各 k/2) |
|---|---|---|---|
| 25 | 93.9% / 91.9% | 95.0% / 93.3% | 96.4% / 95.2% |
| 50 | 96.4% / 95.2% | 97.7% / 97.0% | **98.7% / 98.2%** |
| 75 | 97.5% / 96.7% | 98.5% / 98.0% | 98.9% / 98.6% |
| 100 | 98.2% / 97.6% | 99.3% / 99.1% | 99.3% / 99.1% |

(格式:全體 / 刁鑽。union = 兩邊各取 k/2 後交錯去重)

### 為什麼召回贏了卻轉化不成成功率

把「失敗的句子裡,正確 op 是否已經出現在撈回來的例句中」量出來(k=25 BM25 的 81 個失敗):

| 檢索設定 | 正確 op 已在例句中卻仍答錯 |
|---|---|
| bm25 k=25 | 59/81(**73%**)|
| bm25 k=50 | 66/81(81%)|
| bm25 k=100 | 74/81(91%)|
| union k=50 | 78/81(**96%**)|
| union k=100 | 80/81(99%)|

**檢索已經不是瓶頸。** 即使召回做到近乎完美,那些句子仍然錯 —— 模型是**分不開**
兩個近義 op,不是**沒看到**正確答案。所以:

- 加大 k、換更強的 embedding、做 union —— 這些提升的召回,落在「本來就已經給對了
  卻仍答錯」的區間裡,轉化不成成功率
- 唯一還有效的槓桿是**消歧規則**(已做兩輪:15 組 + 5 組)
- 剩下最大的一塊 `reachable_from ↔ transitive_fanout`(12 句)是標註本身的歧義,
  catalog 已寫明兩者回傳同一集合、只差輸出格式,句子看不出差別

### 過程中找到的資料問題:例句庫 17% 重複

例句庫 2130 筆裡只有 **1776 種唯一的 (句子, op)** —— 354 筆(17%)是重複的,
因為 test91-100 與 test101-171 有相同句子,而 "Please count all the gates in this
design..." 出現 10 次。

檢索原本排序的是**列**而不是**唯一例句**,所以 top-50 平均只帶 **42.5 則**唯一
例句(最少 28)—— 約 15% 的 prompt 預算浪費在讓模型重看已經看過的句子。

在基底 `top_k` 依 `(text, op)` 去重後,六個後端都確實回傳 50 則唯一例句,
召回也跟著上升(bm25 96.4% → 97.0%,onnx 97.7% → 98.0%)。

**但端到端一句不差:** gpt-4o-mini + bm25 k=50 去重前後都是 **1353/1420 = 95.3%**。
第三次驗證同一件事 —— 檢索品質已經不是瓶頸,把 15% 的 prompt 預算從重複例句換成
新例句,對答案沒有影響。

**k=100 反而更差。** 拿召回最高的組合(union k=100,召回 99.5%)實測端到端:

| 設定 | 召回 | 端到端 |
|---|---|---|
| bm25 k=50 | 97.0% | **95.3%** |
| union k=50 | 98.8% | 95.3% |
| union k=100 | **99.5%** | 94.9% |

召回 +2.5pt,端到端 **−0.4pt** —— 兩者在這個區間是**反向**的。多出來的 50 則
例句是干擾項。`DEFAULT_TOP_K = 50` 是量出來的最佳點,不是「越多越好」。

去重後的 k 掃描(召回持續上升,但同理不會轉化):

| k | bm25 | union |
|---|---|---|
| 25 | 94.5% / 92.7% | 97.2% / 96.2% |
| 50 | 97.0% / 96.1% | 98.8% / 98.4% |
| 75 | 98.0% / 97.3% | 99.3% / 99.1% |
| 100 | 98.5% / 98.0% | 99.5% / 99.3% |
| 150 | 99.2% / 98.9% | 99.9% / 99.8% |

### Qdrant 測了,沒有加分

事前預測:**Qdrant 是向量搜尋引擎,不是編碼器** —— 它存放並搜尋同一個編碼器產生的
向量。2130 筆的規模下,精確餘弦就是一次 2130×384 matmul,而 HNSW 是近似最近鄰,
**近似的上限就是精確**。所以「同一模型 + Qdrant」不可能贏過已經在做精確搜尋的
`OnnxRetriever`。

實測(1420 句,leave-one-testcase-out,已去重):

| 後端 | 查詢 | recall@50 | 刁鑽 |
|---|---|---|---|
| bm25(stdlib) | **0.5ms** | 97.0% | 96.1% |
| onnx(精確餘弦,in-process)| 14.5ms | 98.0% | 97.3% |
| **qdrant(dense,HNSW)** | 25.5ms | **98.0%** | **97.3%** |
| union-interleave | 15.4ms | **98.8%** | **98.4%** |
| union-rrf(自己實作)| 15.7ms | 98.7% | 98.3% |
| **qdrant-hybrid(sparse+dense,engine RRF)** | 39.4ms | **98.7%** | **98.3%** |

兩組對照都到小數點吻合:

- **`qdrant` = `onnx`(98.0 / 97.3)** —— 同一份向量,2130 筆下 HNSW 就等於精確搜尋,
  只是慢 1.8 倍(client 序列化開銷)
- **`qdrant-hybrid` = 自己實作的 `union-rrf`(98.7 / 98.3)** —— 同樣的 RRF,慢 2.5 倍

而且 **RRF 沒有贏過天真交錯**(98.7 vs 98.8)。融合演算法在兩個來源、k=50 的情境下
不是瓶頸。

Qdrant 真正有價值的地方(百萬級向量的 ANN、payload 過濾、持久化、分散式)在這個
2130 筆的問題上全都用不到。

### 量測紀律:兩次踩到同一個坑

第一次:兩個評估 job 同時打 OpenAI,超過 4M TPM,重試耗盡後每個失敗的呼叫都變成
no-op ack,被計為「模型放棄」。第二次:同樣的錯誤打在 Anthropic 上,產出一個看起來
很合理的 **67.2%** —— 對照同設定單獨跑的 97.8%,差別只在有沒有跟別人搶額度
(458 個放棄 vs 3 個)。

**降級的跑分看起來就像成績,這是最危險的地方。** 已修:

1. `LLMClient` 對 rate limit 給 7 次重試、指數退避上限 60s(原本 3 次 2s/4s)
2. `LLMClient.degraded` 計數所有重試耗盡的呼叫
3. `route_eval.py` 只要 `degraded > 0` 就印 **INVALID RUN** 橫幅,說明成功率無意義

第 3 點立刻就派上用場 —— 下一次執行時抓到 Anthropic 額度用罄,而不是默默產出一個
低分。這個計數器在正式比賽跑分時同樣重要:同樣的靜默會把請求變成「已確認但沒做」。

### 回歸驗證

`scripts/verify_basic_golden.py`(驅動真實 Agent + 獨立重算,無 API key → 只走 regex 層):
**211 PASS / 4 FAIL / 244 SKIP**。同樣 4 個 FAIL 在 `main` 上一模一樣出現,
是既有的 golden/實作差異(type_count 的 cone 範圍、const1 的 A21.1 functional
constants、deepest vs largest_fanin_cone),**不是這次改動的回歸**。

`scripts/route_check.py` 通過;真實 Agent 跑 testcase/test01 正常(載入 1794 gate、
寫出、stderr 乾淨)。

---

## 0. 現況基準

| 測資 | 路徑 | 成功率 |
|---|---|---|
| test101-171(1420 句)| 純 LLM | **86.9%**(安全 99.4% / 刁鑽 82.7%)|
| test91-100(710 句)| 混合 regex+LLM | 91.5% |
| test91-100(710 句)| 純 LLM | 92.4% |

test101-171 的 186 個失敗:

| 類型 | 數量 | 佔比 | 說明 |
|---|---|---|---|
| **放棄** | 107 | 58% | 回 `noop` 或完全沒輸出 |
| **選錯** | 79 | 42% | 選成別的 op,集中在 38 種配對 |

兩種成因不同:放棄 = 模型**沒認出**這種問法(正確 op 就在清單裡);選錯 = 模型**分不開**兩個相近的 op。

---

## 1. 目標架構

做完之後,送給 LLM 的 prompt 長這樣:

```python
system=[
    # 固定不變、上快取斷點。包含規則、EDA 詞彙、消歧規則、72 個 op 清單,
    # 以及約 30 則核心範例(同時擔任輸出格式的示範)
    {"type": "text", "text": CATALOG_STATIC,
     "cache_control": {"type": "ephemeral"}},

    # 每次不同:依這句話從 2130 筆例句庫檢索回來的 25 則
    {"type": "text", "text": retrieved_examples},
],
messages=[{"role": "user", "content": line}],
```

Runtime flow —— **單一檢索步驟**(實作後預設是 BM25,不是 ONNX,理由見 §-1):

```
使用者句子
  → BM25 對 2130 筆例句計分（倒排索引，0.5ms，零依賴）
  → top-50 貼進 system[1]
  → 打 API
```

`--retriever onnx` 會改走「編碼查詢句(14.7ms)→ 對預先算好的向量做 cosine」;
`union` 則是兩者交錯。三種都實作了,實測見 §-1。

**退路:檢索失敗就不組 `system[1]`**,退回 catalog-only 的行為。零額外程式碼。

三個關鍵設計決定,理由見 §5:

1. 檢索的是**例句**,不是工具清單 —— 72 個 op 永遠全部給,一個都不刪
2. 動態內容一律放在快取斷點**之後**(實測:靜態 10,280 tokens 全部命中快取,
   每次只有 ~630 tokens 付全額)
3. **與原計畫不同**:catalog 既有的 79 則手寫 few-shot 全部留在靜態段,沒有縮成
   30 則 —— 它們帶 params、示範輸出格式,而檢索回來的只有「句子 → op」。純加法,
   快取地基也更穩

---

## 2. 為什麼是這個順序

1. **投報率**:Phase 1 不寫程式、不加依賴,覆蓋約 42% 的失敗。
2. **快取地基**:Phase 2 會把大部分 few-shot 移出固定 prompt。靜態前綴若掉到 **4096 tokens** 以下,Haiku 4.5 的快取會**靜默失效** —— 不報錯,`cache_creation_input_tokens` 一直是 0。

   | 靜態前綴內容 | 估計 tokens |
   |---|---|
   | 規則 + 詞彙 + 消歧 + op 清單 | ~4221 ⚠️ 只差 125 |
   | + Phase 1 補的 15 組消歧 | ~6433 |
   | + 保留 30 則核心範例 | **~7600** ✅ |

   Phase 1 補的東西剛好加在靜態那一半。**順序反過來做,會得到一個看起來有開快取、實際上一次都沒命中的系統。**
3. **可歸因**:分階段跑同一批測資,才知道進步是哪來的。

---

## Phase 0 — 先把量測工具補起來 ✅ 已完成

**所有後續階段的前提。** repo 裡只有 `scripts/route_check.py`,它只驗 regex 層(印 `NO_REGEX` 就結束,不打 API)。那三份 xlsx 的純 LLM 結果是另外產的,腳本沒進版控,ground truth 也只存在 xlsx 裡 —— **現在沒有辦法重跑基準**。

需要一支 `scripts/route_eval.py`:

- 吃 `testcase/test101-171` 的 prompt + ground truth
- 每句只跑 `Fallback.translate()`,不執行 handler(test91-171 本來就沒附 `.v` 設計檔)
- 輸出:總成功率、安全/刁鑽分開、放棄 vs 選錯比例、混淆配對表
- `--limit N` 先跑小批試水溫(全量 1420 句要真的花錢)

**驗收**

- [x] ground truth 從 xlsx 匯成純文字檔進版控 → `cada/llm/examples.jsonl`(2130 筆,`scripts/export_examples.py` 用 stdlib 解 xlsx)
- [x] `scripts/route_eval.py` —— 支援 `--retriever` / `--top-k` / `--sample` / `--provider` / leave-one-testcase-out

---

## Phase 1 — 補 catalog 的文字缺口 ✅ 已完成

**不需要任何新依賴。只改 `cada/llm/allowed_intents.py`。**

### 1.1 補四個完全沒有範例的 op

這四個是「放棄」次數最多的,而且在 `FEW-SHOT EXAMPLES` 區塊裡連一則都沒有:

| op | 放棄次數 |
|---|---|
| `symmetric` | 13 |
| `is_cut` | 9 |
| `count_ports` | 6 |
| `same_clock` | 5 |

合計 33 筆(全部失敗的 18%)。每個至少補 2–3 則涵蓋不同問法的範例。

### 1.2 補 15 組沒被消歧區涵蓋的混淆配對

19 組重複出現(n≥2)的配對裡,15 組在 `DISAMBIGUATION` 完全沒寫到:

| 次數 | 正確 | 被誤選成 |
|---|---|---|
| 6 | `cone_depth` | `max_depth_between` |
| 6 | `connected_to_output` | `fanout` |
| 6 | `dominator` | `path_exists` |
| 4 | `floating_count` | `delta_count` |
| 3 | `depends_on` | `path_exists` |
| 3 | `ffs_on_clock` | `reachable_from` |
| 3 | `ffs_on_clock` | `fanout` |
| 3 | `same_clock` | `signals_equivalent` |
| 3 | `successors` | `fanout` |
| 2 | `cone_depth` | `cone_gate_count` |
| 2 | `ffs_on_clock` | `gates_driven_by` |
| 2 | `is_cut` | `articulation` |
| 2 | `is_cut` | `dominator` |
| 2 | `minimize_area` | `remove_dangling` |
| 2 | `transitive_fanin` | `reachable_from` |

合計 46 筆(25%)。沿用既有 `▸ A vs B` + `KEY RULE:` 的寫法 —— 那格式在已涵蓋的配對上有效。

**驗收**

- [x] 四個缺 few-shot 的 op 各補 3 則(取自 test91-100,不碰評估集)
- [x] 9 個新消歧區塊涵蓋全部 15 組配對,並修正 catalog 裡「clock net 的 loads 算 fanout」這條沒有案例支撐、且正是 `ffs_on_clock → fanout` 肇因的錯誤宣稱
- [x] 針對性回歸:因這 15 組而失敗的 49 句 → 43 句答對(88%)
- [x] 靜態前綴實測 **10,280 tokens**(`count_tokens`,遠高於 4096 下限)

---

## Phase 2 — 例句檢索 ✅ 已完成(預設改為 BM25,非 embedding-only)

### 2.1 建例句庫

`testcase/test91-171` 已經有 **2130 筆標好的 (句子 → op)**,全部塞不進 prompt。

- [x] 匯出 `cada/llm/examples.jsonl`(2130 筆,312KB)
- [x] **與原計畫不同:catalog 既有的 79 則手寫 few-shot 全部保留在靜態段**,沒有縮成 30 則。
      它們帶 params、示範輸出格式,而檢索回來的例句只有「句子 → op」。純加法、
      快取地基更穩(10,280 tokens),沒有理由拆掉。

### 2.2 拆 prompt

- [x] `client.complete()` 加 `dynamic` 參數,一律落在快取斷點之後(Anthropic 第二個 system block;OpenAI 附在 system 訊息末尾)
- [x] `Fallback._examples()` 組動態區塊,任何例外都只是跳過,不影響請求
- [x] 快取實測:第 1 次 write 10,274 tokens,第 2 次起 **cache_read 10,274**,每次只有 ~630 tokens 付全額

### 2.3 檢索器

介面單一,實作只有一個:

```python
class Retriever:
    def top_k(self, query: str, k: int) -> list[Example]: ...
```

- [x] `scripts/fetch_embed_model.py`(MiniLM-L6-v2 fp32 ONNX,90.4MB)+ `scripts/build_vectors.py`(2130×384 float32 → 3.3MB,附自我檢索驗證)
- [x] runtime 只編碼查詢句(14.7ms),對預算向量做 cosine
- [x] 四種後端:`bm25`(預設,零依賴)/ `onnx` / `union`(兩者交錯)/ `none`

### 2.4 一次性離線基準(不進 binary、不進 flow)

BM25 版本跑**一次**就好,純粹當對照:

- [x] 純 Python BM25(倒排索引,建置 0.01s、查詢 0.5ms),離線 label-recall 與端到端全量都量了
- [x] **結論:embedding 在 k=25 端到端輸 1 句(94.2% vs 94.3%),沒有達到門檻**。詳見 §-1

理由見 §5.2:BM25 在這個任務上召回已經 97.5%,天花板只剩 2.5 個百分點。

**Phase 2 驗收**

- [x] `cache_read_input_tokens` 第二次呼叫起 = 10,274 ✅
- [x] test101-171 成功率 **94.3%**(k=25)/ **94.6%**(k=50)✅ 超過 94% 門檻
- [x] 評估已實作 **leave-one-testcase-out** —— 撈例句時排除 query 自己所屬的 testcase

> ⚠️ 最容易搞砸的一點。例句庫和評估集是同一批資料,不排除就是拿答案抄答案,數字是假的。
> 實務上 test101-171 的每個 testcase 只含一個 op,排除自己之後同標籤例句只能來自 test91-100,這個切分天然乾淨。

---

## Phase 3 — 檢查 regex 層是否還划算

在 test91-100 上:**混合 91.5% vs 純 LLM 92.4%** —— 108 條 regex 在那組測資上是**淨負分**。

- [ ] 逐條比對混合與純 LLM 的差異,找出誤攔的規則
- [ ] 評估「保留但收窄」或「只留高信心規則」

獨立的一條線,不擋 Phase 0–2。注意 regex 有它的價值(零延遲、零成本、離線可用),不要只看準確率就整層拆掉。

---

## 4. 打包

> ⚠️ **本節多數內容現在是選用的。** 實測後預設檢索器改為零依賴的 BM25(見 §-1),
> binary 維持 45MB,下面的 glibc / libstdc++ / 指令集風險都不必承擔。只有在
> 明確要走 ONNX 路線時才適用。已實作的部分:`scripts/spec_assets.py`(打包
> 邏輯進版控,因為 `*.spec` 是 gitignore 的產物)、`cada/llm/retrieval.py` 的
> `data_path()`(one-file 解壓目錄解析)、`requirements-retrieval.txt`(把
> numpy/onnxruntime/tokenizers 排除在基本安裝之外)。

現況:PyInstaller one-file,`dist/cada1125_beta` **45MB**,已內嵌 abc(23MB)+ yosys(35MB)。依賴只有 `PyYAML` / `openai` / `anthropic`。

| 項目 | 體積 | 備註 |
|---|---|---|
| onnxruntime + numpy | ~15–25MB | numpy 跟著 onnxruntime 一起來 |
| int8 量化的 MiniLM 級模型 | ~25–35MB | fp32 約 90MB,量化後品質差異可忽略 |
| 2130 筆預算向量 `.npy` | 3.3MB(fp32)/ 0.8MB(int8)| build 時算好 |
| tokenizer(`tokenizers` wheel)| ~3–5MB | |
| `examples.jsonl` | ~210KB | |

**45MB → 約 90–110MB。**

`.spec` 要加:

```python
datas += [('cada/llm/examples.jsonl', 'cada/llm'),
          ('cada/llm/example_vecs.npy', 'cada/llm'),
          ('<模型目錄>', 'embed_model')]
hiddenimports += ['numpy', 'onnxruntime']
```

路徑解析沿用 `cada/toolpaths.py` 的 `${BUNDLE}` 機制(one-file 解壓目錄),不可寫死絕對路徑。

### 4.1 環境前提(已確認,不再是風險)

比賽機器的 **Python 版本無關** —— PyInstaller 會把直譯器本身包進執行檔,目標機器不需要裝 Python。

| 項目 | 狀態 |
|---|---|
| **glibc 版本** | ✅ 已確認相容。建置機 Ubuntu glibc **2.35** |
| **CPU 架構** | ✅ 已確認 **x86_64** |
| **CPU 指令集(AVX2/AVX-512)** | ⚠️ **未確認 —— 先做,之後再修正**,見 §4.3 |

### 4.2 靜態連結原則:能靜態的就靜態

| 元件 | 能否靜態 | 做法 |
|---|---|---|
| **abc / yosys** | ✅ 已經是 | 維持現況。它們是獨立子行程,不 dlopen、不解 DNS,全靜態沒有副作用 |
| **libstdc++ / libgcc** | ✅ 可以 | 見下方兩條路線 |
| **glibc** | ❌ 不可能 | 與 PyInstaller 衝突,見下方說明。已由 §4.1 的版本確認迴避 |
| **CPU 指令集** | ❌ 與連結無關 | 是編譯目標的問題,不是連結方式的問題 |

**libstdc++ / libgcc 兩條路線,先走 A:**

- **A(預設,零成本)** —— 讓 PyInstaller 把 `.so` 收進 bundle。**現有 binary 裡已經有 `libgcc_s.so.1`**,證明它確實會自動收擴充模組相依的系統庫。加上 onnxruntime 之後只要確認 `libstdc++.so.6` 也被收進去就行。嚴格說這是「隨附」不是「靜態」,但可攜性效果相同。
  - 驗證:`strings -a dist/<binary> | grep -o 'libstdc++[^ ]*'`
  - 沒被收進去的話,在 `.spec` 手動加:
    ```python
    binaries += [('/usr/lib/x86_64-linux-gnu/libstdc++.so.6', '.')]
    ```
- **B(只在 A 失敗時才做)** —— 從源碼建 onnxruntime,加 `-static-libstdc++ -static-libgcc`。真正的靜態,但要跑 CMake、耗時且選項繁多。**成本遠高於 A,不要一開始就走。**

**為什麼 glibc 不可能靜態:** one-file 的核心機制就是執行時把 `.so` 解壓出來 `dlopen`,而 glibc 一旦靜態連結,`dlopen` 就不保證能用;另外 NSS(`getaddrinfo`)也需要動態載入,而打 API 必須解 DNS。這是機制衝突,不是參數問題。abc/yosys 能全靜態,正是因為它們兩件事都不做。

### 4.3 CPU 指令集:先做,之後修正(僅 ONNX 路線適用;預設 BM25 不涉及)

現在**先用 onnxruntime 官方預建 wheel**,它對多數 kernel 有執行期分派,在大部分 x86_64 上沒問題。等目標機器確定後再回來處理。

確定之後要做的事:

- [ ] 在目標機器上 `lscpu | grep -o 'avx[^ ]*'` 看支援哪些指令集
- [ ] 若不支援 AVX2:改用保守建置,或用 `ORT_DISABLE_ALL` 之類的最佳化開關降級(需實測)
- [ ] 加一個啟動時的 smoke test —— 編碼一句固定字串、比對向量,失敗就走 §1 的退路而不是 crash

> 指令集不相容的症狀是 **SIGILL(Illegal instruction)**,而且通常在第一次真的跑推論時才炸,不是載入時。所以 smoke test 要真的做一次編碼,不能只檢查模型載得起來。

### 4.4 其餘檢查

- [ ] one-file 打包後模型權重、vocab、`.npy` 都讀得到
- [ ] 打包後的 binary 在**乾淨機器**上跑得起來(不能只在開發機驗)
- [ ] 模型載入或編碼失敗時確實走退路,而不是整個 crash
- [ ] 確認建置環境固定 —— 現有 binary 內含 `libpython3.13`,但開發 venv 是 3.12,兩者不是同一個環境。動打包前先把建置流程定下來,否則出問題難歸因

### 4.5 選用:拿掉 `tokenizers` 依賴(已無必要 —— 預設 BM25 完全不用 tokenizer)

BERT 系列的 WordPiece 用純 Python 實作約 100 行 + 230KB 詞表就能取代 Rust wheel。但**必須跟原版逐 token 一致**,否則向量整個錯掉而且不會報錯。要做的話得先寫逐 token 比對測試;不確定就用官方 wheel。

### 4.6 明確排除

**torch / sentence-transformers。** 推論階段模型固定,匯出 ONNX 用 `onnxruntime` 跑是**同一份權重、同一個模型、輸出幾乎逐位元相同**,體積卻少一個數量級。torch 買到的不是準確率,是幾百 MB 和一組更難打包的依賴。決定品質的是模型,不是 runtime。

---

## 5. 已量測的數字(決策依據)

### 5.1 為什麼不檢索工具清單

BM25 對 1420 句量「正確 op 是否留在候選集內」:

| 檢索文件內容 | recall@5 | recall@10 | recall@20 |
|---|---|---|---|
| op 名 + 一行規格 | 43.1% | 53.0% | 65.4% |
| 加 few-shot + 消歧文字 | 58.1% | 66.1% | **75.1%** |

就算之後 LLM 百分之百選對,上限 75.1%,比現況 86.9% **低 12 個百分點**。而且:

- **選錯那 42% 完全免疫** —— 會被搞混的兩個 op 一定長得像,長得像就一定一起被撈進 top-k,模型面對的還是同一個二選一
- **放棄那 58% 只會更糟** —— 候選集變小,「沒有一個符合」更容易成立

### 5.2 為什麼檢索例句可行

例句庫 = test91-100(710 句 / 71 ops),query = test101-171(1420 句),天然 leave-out:

| | 全體 | 安全 | 刁鑽 |
|---|---|---|---|
| label-recall@5 | 90.8% | 100% | 87.7% |
| label-recall@10 | 94.6% | 100% | 92.9% |
| **label-recall@25** | **97.5%** | 100% | 96.7% |

**目前 LLM 答錯的 186 句裡,BM25 top-25 有 179 句(96%)撈得到正確標籤。**

差異來自「句子對句子」的字面重疊遠高於「句子對 op 名稱」。同時這也說明:如果接上例句後準確率沒明顯跳,**瓶頸不在檢索器**,而在 k 值、例句擺放位置與提示語 —— 換更強的模型救不了。

### 5.3 成本

每則範例平均 39 tokens,檢索 25 則 ≈ 985 tokens:

| 情境 | 等效 input tokens |
|---|---|
| 現在(全靜態 + 快取)| ~690 |
| 目標架構 + 快取 | ~1795 |
| 目標架構、沒快取 | ~8635 |

快取仍砍掉約 **79%**;單次成本比現在高約 2.6 倍,那是準確率的價錢。以 1600 次呼叫、Haiku 4.5 計,差額是幾毛美金。

---

## 5.4 下一步(如果還要往上推)

剩下 31 個失敗(haiku + BM25 k=50)的組成:

| 類型 | 數量 | 可行性 |
|---|---|---|
| `reachable_from ↔ transitive_fanout` | 16 | ❌ 標註歧義,catalog 已寫明兩者回傳同一集合,句子看不出差別 |
| `dominator ↔ path_exists` | 4 | 🟡 規則已寫(gate vs net),仍有殘留 |
| `successors ↔ connected_to_output` | 2 | 🟡 規則已寫(鄰接詞彙 vs 接腳詞彙),仍有殘留 |
| 其餘單發 | 6 | 🟢 各自獨立,補規則的邊際效益低 |
| 放棄(noop/無輸出) | 3 | 🟢 已幾乎消滅 |

**扣掉標註歧義那 16 句,實際上限已經是 98.9%。** 要再往上,順序是:
(1) 釐清 `reachable_from` vs `transitive_fanout` 的判準(可能要回頭看題目定義),
(2) 換更強的模型,(3) 擴充例句庫。加大 k、換 embedding 都已量測為無效。

---

## 6. 明確不做的事

- **不做「檢索工具清單、只給 LLM top-k 個 op」** —— 見 §5.1(recall@20 只有 75.1%,
  比不做還低 12pt)
- **不引入 torch / sentence-transformers** —— 見 §4.6
- ~~**runtime 不放 BM25**~~ —— **這條被實測推翻:BM25 是預設,ONNX 才是選用的。**
  原本假設 embedding 會贏所以只當離線基準,實際上 BM25 在 k=25 和 k=50 都略勝
  (95.3% vs 94.9%),而且零依賴、零體積、可離線。
- **不再追 `reachable_from ↔ transitive_fanout`** —— 16 句(佔 haiku 剩餘失敗的
  52%)是標註歧義而非語言歧義,寫規則只會擬合標註。要解得先回頭釐清題目定義。
- **不再加大 k 或換更強的檢索器** —— 已量測:失敗句裡 73–96% 本來就已經看到正確
  答案的例句了(§5「為什麼召回贏了卻轉化不成成功率」)。

---

## 7. 量測方式

```bash
# 只驗 regex 路由,不打 API、不花錢
.venv/bin/python scripts/route_check.py

# 純 LLM routing 評估
.venv/bin/python scripts/route_eval.py --sample 200 --retriever bm25   # 試水溫(等距取樣,涵蓋全部 op)
.venv/bin/python scripts/route_eval.py --retriever bm25                # 全量 1420 句
.venv/bin/python scripts/route_eval.py --retriever bm25 --provider anthropic   # 換模型比較
.venv/bin/python scripts/route_eval.py --retriever onnx --top-k 100    # 換後端 / 調 k

# 重建檢索資產(改了 testcase 或 xlsx 之後)
.venv/bin/python scripts/export_examples.py     # xlsx -> examples.jsonl
.venv/bin/python scripts/build_vectors.py       # 只有走 onnx 才需要

# 回歸:regex 路由 + golden 三方驗證
.venv/bin/python scripts/route_check.py
.venv/bin/python scripts/verify_basic_golden.py

# 靜態前綴的真實 token 數(確認遠高於 4096)
# client.messages.count_tokens(model=..., system=CATALOG_STATIC, messages=[...])
```

每個 Phase 結束後全量重跑,更新 §0 的基準表,並記錄:總成功率、安全/刁鑽分開、放棄 vs 選錯比例。**光看總成功率會看不出是哪一種失敗被修好了。**
