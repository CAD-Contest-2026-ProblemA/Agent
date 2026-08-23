# 這個系統怎麼運作(agent + evaluator 白話說明)

這份文件用白話把兩件事講清楚:
1. **agent**:一個自然語言請求,怎麼從 stdin 進來、被處理、再從 stdout 出去。
2. **evaluator**:本機怎麼檢查 agent 跑得對不對。

---

## 一、核心觀念(先記住這句)

> **「大腦」是 deterministic 的 EDA 程式,LLM 只負責把一句話翻成一個指令。**

- 真正在做電路分析/轉換/驗證的,是我們自己寫的 Python(在自寫的網表 IR 上跑圖演算法)。
- ABC / yosys 只當兩種「裁判」:**判斷兩個電路等不等價**、**幫忙做成本最小化的最佳化**。它們不負責命名、不當答案來源。
- 打包的 binary 預設讓 LLM(gpt-4o-mini / claude-haiku)處理每一句;加 `--rules` 才會先走 regex、沒命中再交給 LLM。LLM 只把句子翻成 `{intent, params}`,不做電路推理。

---

## 二、一個請求的生命週期(agent 怎麼跑)

```
./cada1125_alpha -config cfg.yaml < prompt.txt
        │
        ▼
cada1125_alpha (bash shim)        ── 找 .venv/python、設 PYTHONPATH、PYTHONHASHSEED=0
        │                            不改工作目錄(讓相對路徑照常解析)
        ▼
cada/main.py                       ── 讀 config(LLM 設定 + 工具路徑)→ 設定 abc/yosys 路徑 → 建 Agent
        │                            → 跑 Protocol REPL
        ▼
cada/io_/protocol.py (REPL)        ── 從 stdin 一行一行讀
        │   第1行:抽出 case name → 開 <case>.log
        │   每一行:呼叫 agent.handle(line, id) 拿到回應
        │           印出   #RESPONSE <id> / 回應內容 / #END <id>
        │           stdout 和 .log 都 flush(harness 看到 #END 才送下一行)
        │   EOF:直接結束,不發 frame
        ▼
cada/agent/agent.py : handle(line)
        │
        │  (1) rules-on 時:先由上到下比對 regex 表,命中就呼叫對應 handler
        │  (2) rules-off 或 regex 沒命中:由 LLM 翻成 {intent, params}(有 cache)→ dispatch
        │  (3) 還是不行 → 安全 no-op(回「已收到、設計不變」)
        ▼
   對應的 handler 做事(分三類,見下)
        ▼
   回傳一段文字 → protocol 包成 #RESPONSE/#END 印出去
```

### handler 三大類

| 類型 | 例子請求 | 做什麼 | 會不會改設計 |
|------|----------|--------|--------------|
| **分析** | 「count gates」「max depth A→B」「path 存不存在」「cone 有幾個 gate」 | 在 IR 上跑圖演算法算出答案,格式化成文字 | 否 |
| **轉換** | 「remap 成 NAND+NOT」「插 buffer 讓 fanout≤4」「移除 dangling」「合併重複」「改名」 | 先 snapshot → 改 IR → **過 guards** → 過了 commit、沒過 rollback | 是 |
| **最佳化** | 「minimize depth」「optimize cone of X」「resynthesize」 | 進 **resynth 引擎**(見下節):多 seed × 多 recipe 的組合最佳化 → 依真實成本排名 → cec 驗證 → 有改善才採用,否則回原圖 | 是(只在更好時) |

### 「轉換」為什麼安全(防 0 分的關鍵)

每個轉換 **commit 前**都會過 `cada/guards/validators.py` 的三關:
1. **功能等價**:用 register-cut 把時序電路投影成組合電路,丟 ABC `cec`。
2. **結構界限**:該行有要求(例如 fanout ≤ 4)就重算一次確認。
3. **basis 純度**:該行要求只用某些閘(例如只有 NAND/NOT)就檢查 gate 型別。

任何一關沒過 → **rollback** 回轉換前的 snapshot(寧可不做,也不交出功能被改壞的設計)。
另外,每個轉換會把「改了幾個」記成 delta,後面那種「How many X were added?」的問題就直接讀這個 delta。

### 「最佳化」怎麼跑(resynth 引擎,ALS_Final_Project 方法論移植)

`cada/optimize/resynth.py` 是所有 depth / cone-depth / gate-count 最佳化的引擎,
流程是「多個 seed × 多個 recipe → 依真實成本排名 → cec 把關」:

1. **切 register boundary**:DFF 的 Q 當虛擬 PI、D 當虛擬 PO,把組合核心投影成 BLIF
   (`abc_opt._opt_blif`,DFF 本體完全不動)。
2. **產生 seeds**:
   - `base`:目前的設計本身。
   - `tpl`(`templates.py`,**逆向工程模板**):對整個組合核心做 512 筆隨機模擬,
     把 PI bus / 暫存器組(Q bus 對應的 D 向量)當「字」,猜每個目標字是不是已知函數。
     函數庫涵蓋:純接線置換(const shift/rotate/byteswap/bitreverse/shift-register,
     逐 bit 欄位雜湊比對,深度 ≤1)、二元算術(add/sub/±1/neg/帶 carry-in 加法)、
     乘法與 MAC(a*b+c)、多運算元加法樹(a+b+c、a+b+c+d)、平均((a+b)>>1 含
     floor/ceil)、bitwise 全家、比較器(eq/ne/lt/le/gt/ge)、max/min/absdiff、
     字級 mux(sel ? a : b)、變動 shift(barrel)。猜中的先對該 cone 做 ABC `cec`
     **證明**,再用深度最優結構重建(Sklansky prefix adder、3:2 壓縮 + Wallace tree、
     prefix borrow 比較、log 層 mux barrel)。沒猜中就沒有這個 seed——純加分項,
     錯誤的猜測到不了下一步。
   - `ys`(`yosys_synth.py`):把 BLIF 丟給 yosys `opt -full; techmap; aigmap` 重新合成,
     當作結構不同的第三個起點(yosys 不在就自動略過)。
3. **ABC recipe portfolio**:每個 seed 各跑數條 recipe。深度用 `dch -f; if -g -K 6` 的
   choice-mapping 迴圈(實測比 `resyn2` 淺 2~4 倍),面積用 `compress2rs` 家族;
   最後都 `map` 到**單位延遲 genlib**(可依 basis 限制只給 NAND+NOT 等 cell),
   所以 ABC 最小化的就是比賽計的成本(1 gate = 1 level,含 inverter)。
4. **單調選擇**:所有候選解析回 IR、重算**真實成本**(不信 ABC 的數字)、由小到大排序,
   第一個「嚴格更好 + 全設計 `cec` 等價 + basis 純度成立」的候選才會取代目前設計;
   一個都沒有就回報 already optimal,原圖不動。

這套設計的重點:**模板逆向是通用的**(靠抽樣猜 + cec 證,不靠特定電路長相),
沒看過的電路只要含有已知函數家族的子結構就接得住;接不住也還有 portfolio 保底。

---

## 三、用到的關鍵零件

| 檔案 | 角色 |
|------|------|
| `cada/netlist/ir.py` | 網表 IR(唯一真相):gate / dff / net、driver/loads、snapshot |
| `cada/netlist/reader.py` `writer.py` | 自寫 Verilog parser / canonical 結構 writer(可完整 round-trip) |
| `cada/analysis/*` | 計數、cone、深度、連通性、path(DP 計數不列舉)、functional、sequential |
| `cada/transform/*` | basis 轉換、XOR/XNOR 分解、常數傳遞、dangling 移除、重複合併、buffer 樹、改名 |
| `cada/optimize/resynth.py` | 最佳化引擎:多 seed × recipe portfolio → 真實成本排名 → cec 把關 |
| `cada/optimize/templates.py` | 逆向工程模板:抽樣猜 word-level 函數 → cec 證明 → 深度最優重建 |
| `cada/optimize/yosys_synth.py` | yosys 重合成 seed(`opt -full; techmap; aigmap`,沒裝就略過) |
| `cada/optimize/abc_opt.py` | ABC 底層:register-cut BLIF 匯出、單位延遲 genlib 映射、解析回 IR |
| `cada/equiv/*` | 等價閘:IR→BLIF→ABC cec(主);yosys 當備援 |
| `cada/agent/*` | 規則 router、狀態(snapshot / delta / case name / frame id) |

---

## 四、具體例子:trace test21(5 行)

`testcase/test21/prompt.txt`:

| # | 請求(節錄) | 走哪個 handler | 做了什麼 | 回應(節錄) |
|---|--------------|----------------|----------|-------------|
| 1 | This is the beginning ... case name is test21. | `h_begin` | 設 case name、開 test21.log | Acknowledged. Initialized testcase "test21"... |
| 2 | load the design from file test21.v ... | `h_load` | 解析網表,存成 original 快照 | Loaded ... 343 combinational gates, 55 flip-flops. |
| 3 | count all the gates ... broken down by type | `h_count_all` | 掃 IR 算各型別數量 | Total gate count: 398 / AND: 0 / OR: 8 / ... / DFF: 55 |
| 4 | Insert buffers ... no gate drives more than 4 loads. ... | `h_buffers_fanout` | snapshot → 插平衡 buffer 樹(每 driver ≤4)→ guard 重算 max-fanout≤4 + cec → commit | Inserted 30 buffer(s) so that no driver exceeds 4 loads; ... verified. |
| 5 | write the current design to test21_out.v | `h_write` | 自寫 writer 輸出結構網表 | Wrote the current netlist to "test21_out.v" successfully. |

每一行都會被 protocol 包成:
```
#RESPONSE 4
Inserted 30 buffer(s) so that no driver exceeds 4 loads; max-fanout bound and equivalence verified.
#END 4
```

---

## 五、evaluator 怎麼檢查(`evaluator/`)

執行:`python evaluator/evaluate.py`(**預設在拋棄式 sandbox 裡跑**,不會留下 `*_out.v`)。用 `--exe` 時,evaluator 會明確傳 `--rules` 與 `--bm25`,不依賴 binary 的預設;加 evaluator 的 `--no-rules` 或 `--no-bm25` 才切換對應模式。

流程(對每個 case):
```
harness:一行一行餵給 agent,記下 (request, response)
        並在「有結構性需求的那一步」拍下 netlist snapshot
        保留 original(載入時)和 final(最後)
        │
        ▼
requirements.derive(每一行)  ── 從文字推導「這行該檢查什麼」
        │
        ▼
四類檢查 + 報告
```

### `derive` 會從句子推出這些需求
- **equiv**:出現「preserve functional equivalence / nothing changes functionally」→ 最終要與原始等價。
- **basis**(全設計):「remap the entire design to only NAND and NOT」之類。
- **basis**(某 cone):「restructure the cone of n8 using only NAND and NOT」之類。
- **gate_absent**(指定閘消失):「convert every XNOR ... to NOR-only」→ 之後不該再有 XNOR(不是整個設計變 NOR/NOT!這是我特地區分的)。
- **max_fanout**:「no gate/signal drives more than K」。
- **optimize cost**:成本是 depth / cone depth / gate count。

### 四類檢查
| 群組 | 檢查什麼 | 怎麼驗(關鍵:用「另一種方法」獨立重算) |
|------|----------|------------------------------------------|
| **HARD** | 0 分與否的硬門檻 | final 與 original 等價(ABC cec)、結構界限重算、basis 純度 grep、指定閘是否歸零、輸出網表能 round-trip |
| **DERIVED** | 有客觀值的答案 | gate 型別數、PI/PO 數,**直接掃 `.v` 原檔**重算後比對回應 |
| **GOLDEN** | 回歸偵測 | 每個回應跟 `evaluator/golden/<case>.txt` 基準逐行比 |
| **OPT** | 最佳化成本 | 報 final 的 max depth / gate count(越小排名越好) |

報告長這樣:
```
test21   HARD 3/3  DERIVED 1/1  GOLDEN ✓
test28   HARD 4/4  DERIVED 1/1  GOLDEN ✓  OPT 130
```
**只要有任何 HARD 沒過,exit code 就非 0**(可以拿來當 CI gate)。

### 重要分際
- **HARD / DERIVED 是真檢查**:等價交給 ABC(外部裁判)、計數掃原檔、界限/basis 從結果網表重算——都不是再問一次 agent 自己。
- **GOLDEN 是回歸,不是對錯**:basis 沒有官方答案,所以 golden 存的是「目前引擎的輸出」。它能抓「輸出有沒有變動」,但要等你把**官方答案**放進 `evaluator/golden/<case>.txt` 才會變成「答案對不對」的真評分。
- `--update-golden`:把目前輸出釘成新基準(只在確定輸出正確時用)。
- `--no-sandbox`:把 `*_out.v` 留在當前目錄(預設是 sandbox、不留檔)。

---

## 六、一句話總結

- **agent**:stdin 進來一句話 → 預設由 LLM 翻成指令(可用 `--rules` 切成 regex 優先)→ deterministic EDA 引擎做事(轉換一定過等價/界限/basis 三關才 commit)→ stdout 出去一段答案 + `#RESPONSE/#END`。
- **evaluator**:重跑一遍,**獨立**驗硬性要求(等價/界限/basis/計數),其餘用 golden 做回歸;預設 sandbox、不污染 repo。
