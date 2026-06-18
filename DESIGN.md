# ICCAD 2026 Problem A — 系統設計文件（修訂版）

> LLM-Assisted Netlist Exploration and Transformation
> 本文件取代先前的 `ai_agent/` 通用 agent 規劃。所有「實測」結論皆在真實 40 個 testcase + `/home/as6325400/abc/abc` + yosys 0.66 上驗證過。

---

## 0. 結論摘要（為什麼原規劃要大改）

原本的 `ai_agent/` 是一個「通用 LLM chatbot 腳手架」（planner / memory / vector_store / web_search / research-coding-debugging workflows）。但本題的本質是：

> **一個 deterministic 的 gate-level netlist EDA 引擎，外面包一層極薄的自然語言前端。**

關鍵約束決定了架構方向：

1. **評分是正確性二元制**：transform / optimize 只要違反任一 hard requirement（功能等價、結構界限、gate basis）整個 testcase 0 分；analysis 要「完全正確」才得分。
2. **LLM 是小模型**（claude-haiku-4-5 / gpt-4o-mini，temp 0.2，4096 tokens）。**LLM 絕不能做電路推理**，只能把一句話翻成一個結構化工具呼叫。
3. **prompt 高度模板化** → 規則式 parser 就能處理絕大多數，LLM 只當 fallback。
4. **重運算交給 ABC**（等價驗證、深度/面積最佳化），輕量圖演算法留 Python。

因此：**「大腦」是 deterministic EDA code，不是 LLM。** 原規劃把優先順序顛倒了。

---

## 1. 題目與評分（精簡版）

- 執行檔 `cadaXXXX_alpha`，呼叫方式 `./cadaXXXX_alpha -config <cfg>`。
- 從 **stdin 逐行**讀自然語言請求；每個請求把答案寫到 **stdout**，框在 `#RESPONSE <id>` … `#END <id>`，並**鏡像**到 `<case_name>.log`。
- harness 看到 `#END <id>` 才送下一行 → **必須每幀 flush，否則死鎖**。
- 設計：單一 `top` module，flat gate-level。Gate：`and/or/nand/nor/not/buf/xor/xnor`（除 not/buf 為單輸入，其餘 2 輸入）；`dff`；wire；常數 `1'b0/1'b1`；scalar/bus port。
- 限時：basic 操作 60s，其他 300s。
- 計分：每 testcase 1 分；optimize 以 `cost_min / cost` 排名（cost = max depth 或 total gate count 或某 cone 的 depth，依該行指定）。

---

## 2. ⚠️ 實測關鍵結論（這些推翻了 PDF 的假設，是設計的地基）

| # | 實測結論 | 對設計的影響 |
|---|----------|--------------|
| F1 | **ABC `read_verilog` 讀不了 DFF testcase**。PDF 寫 `dff(clk,rst_n,d,q)`，但 **40 個 testcase 全部用 named-port `dff g(.RN,.SN,.CK,.D,.Q)`**（async active-low reset/set）。test01–20 無 dff（可讀），test21–40 全部讀失敗（`Cannot parse a standard gate`）。 | **必須自寫 Verilog parser/IR**。ABC 只能吃我們做的 comb-frame 投影。 |
| F2 | **純 Python 夠快**。最大 test39（112,300 instances / 5.6MB）：parse 0.18s、建 driver/loads 0.05s、histogram 0.007s、完整 levelize 0.12s、cone BFS <0.01s，峰值 RSS ~90MB。Python 深度 == ABC pre-strash `lev`（test12: 86==86）。 | **全部 analysis 留 Python**，不需 C extension / numpy / 圖庫。 |
| F3 | **path 列舉會指數爆炸**。test01 已有 2,728,648 條 PI→PO path；test12 有 4.7e11；單一 A→B 也可能 ~2e5（test14）。但 **DP path-count 在 0.05s 內給精確值**。 | path 一律用 **DP 計數 + reachability**，**絕不 materialize**。 |
| F4 | **等價閘可行且快**。`cec a.aig b.aig` 在 112k gate 上 0.02–2s；能抓到壞 transform 並給反例。sequential 用 yosys `equiv_induct`（test31 10393 cells / 8.9s）。 | 兩層等價閘（comb→ABC cec / seq→yosys equiv）。 |
| F5 | **cec 成功字串是 `Networks are equivalent after structural hashing.`**，不是裸的 `Networks are equivalent.`。 | 等價閘用 **substring 比對**，否則把正確 transform 誤判失敗→無謂 rollback→丟分。 |
| F6 | **ABC `write_verilog` 不能當答案檔**。AIG 輸出是 behavioral `assign ~/&/\|`，且 bus 被打散成 escaped scalar（`\n0[0]`），破壞 `input [7:0] n0;` 格式。 | **必須自寫 canonical structural writer**。 |
| F7 | **gate basis 用 `map` 會洩漏 `zero`(CONST0) cell**（只要有 undriven net；test01、test12 實測都有）。`zero` 不在 `{NAND,NOT}` → 違反 basis → 0 分。 | basis 輸出前先 **sweep undriven net**，再 **grep 驗純度**（gate type 必須是請求 basis 的子集）。 |
| F8 | **DFF register-cut 會撞 multi-driver assertion**：(a) 多個 dff 共用同一 Q net（test39:96, test40:160），(b) Q net 同時是 module output port（test39:51, test40:96），(c) D net 同時是宣告的 PO（test40 `Repeated CO names: n10[9]`）。 | cut 必須走 **driver-tracking IR（非 regex）**，處理三種別名；cut 後自動 `read_verilog` 檢查，失敗就 fallback。 |
| F9 | **ABC byte 不可重現但 cost 穩定**（resyn2 跑 5 次 `and=99499 lev=64` 全同，但 md5 不同）。`resyn2/dc2/...` 是 `abc.rc` 的 alias，cwd 不對就 `unknown command`。 | 讀回一律 **by name/topology**；cost 讀 `print_stats`；ABC 一律 `source abc.rc` 或指定 cwd + 絕對路徑。 |
| F10 | **原始設計沒有任何 comb gate 帶常數輸入**（常數只在 dff `.SN/.RN`）。 | 「report gates with const input / const-prop」在初始設計上答案是 0；必須對 **當前狀態**（前面 transform 後）重算。 |

---

## 3. 系統架構

```
stdin ─► io/protocol ─► agent/router (規則優先, LLM fallback)
                             │
                             ▼
                     agent/intents  ──dispatch──►  analysis / transform / optimize 引擎
                             │                              │
                       agent/state                    netlist/ir (唯一真相)
                  (current / original / pre 快照,           │
                   per-transform delta)            netlist/reader  netlist/writer
                             │                     netlist/aig_export (register-cut)
                             ▼                              │
                     guards/validators ◄──── equiv/gate ──► ABC (cec) / yosys (equiv_induct)
                             │                              optimize/abc_opt (resyn2/if -g/dc2)
                             ▼
        #RESPONSE/#END ─► stdout  +  <case>.log   (每幀 flush)
```

**四條鐵則**
1. LLM 只翻譯不推理；rule parser 先行、LLM 當 fallback，且 LLM 只能回 `{intent, params}` JSON。
2. **所有 analysis（counts/depth/cone/path/fanout）只在自寫 IR 上算**，ABC/yosys 只負責「等價判定」與「cost-ranked 合成」，永不當作命名事實來源。
3. **每個 structural transform 後，commit 前一律過等價閘 + 結構界限 + basis 純度**；任一不過就 rollback 到快照（回傳原設計，spec 允許 "Report original if already optimal"）。
4. 一切輸出走自寫 canonical writer（保留 bus 宣告、output-pin-first、inserted gate 名稱用 monotonic counter 依 canonical 順序），確保可重現。

---

## 4. 修訂版目錄結構

```
cada/                         # 套件根（執行檔由 main.py 打包）
  main.py                     # 進入點：解析 -config、跑 REPL
  io/
    protocol.py               # stdin REPL、#RESPONSE/#END framing、flush、.log 鏡像、case-name 抽取
    config.py                 # -config YAML：provider / openai.model / anthropic.model / generation.*
  netlist/
    ir.py                     # Netlist IR：gate/dff/net/port/bus/const、driver[] & loads[]、deepcopy 快照
    reader.py                 # 自寫 gate-level Verilog parser（含 named-port dff、bus 展開、dangling PI/PO）
    writer.py                 # 自寫 canonical structural writer（保留 bus 宣告、output-pin-first、決定性命名）
    aig_export.py             # IR→AIG/BLIF；DFF register-cut（Q→pseudo-PI, D→pseudo-PO）driver-tracking 版
  analysis/
    counts.py                 # 各型別計數、cone 計數、PI/PO 位寬、型別清單、const-input 報告
    depth.py                  # levelize、max depth(A,B)、cone depth、depth>K、最深 output、gate-on-max-path
    paths.py                  # 存在性(避點)、DP path-count、reachability、dominator、articulation、cut、len-0
    cones.py                  # transitive fanin/fanout、shared cone、fanout+loads、successors、reachable
    functional.py             # 訊號等價、output 恆 0、depends-on、Boolean eqn、symmetry、NAND-pair（ABC 輔助）
    sequential.py             # 某 clock 下 FF、reg-to-reg path、enable/hold 偵測、DFF-Q→D 重導
  transform/
    rewrite.py                # XNOR→NOR、XOR→4NAND、XOR→AOI、cone→basis、remap 全設計、NAND-const1→INV
    constprop.py              # 常數傳遞（AND0/OR1/NAND/NOR…）對「當前狀態」迭代到 fixpoint
    cleanup.py                # 移除 dangling/floating、收合 NOT-NOT、合併重複(topo+union-find 到 fixpoint)
    buffering.py              # fanout≤K buffer 樹、每 load 一 buf、reset fanout 降載
    naming.py                 # rename gate/wire/signal + 更新所有 reference
  optimize/
    abc_opt.py                # cost-ranked 深度/面積最佳化（ABC）+ basis remap + cone 最佳化 + 已最佳則回原圖
  equiv/
    abc_bridge.py             # IR→AIG、cec（tier-1 comb）、成功字串 substring 比對
    yosys_bridge.py           # equiv_make/equiv_induct/equiv_status -assert（tier-2 sequential）
    gate.py                   # 等價閘：選 tier、PASS/FAIL、rollback
  llm/
    client.py                 # openai + anthropic provider 切換（讀 config）
    fallback.py               # tool catalog system prompt、{intent,params} JSON schema、驗證、retry、cache
  agent/
    router.py                 # 規則優先 intent parser（regex 表）→ dispatch
    intents.py                # intent 定義 + 參數抽取 + tool 簽章（= 我們對 LLM 描述的「EDA interface」）
    state.py                  # 設計狀態：current/original/pre 快照、per-transform delta、case_name、id 計數
  guards/
    validators.py             # 等價 + 結構界限(max-fanout/max-depth≤K) + basis 純度；失敗 → rollback
  utils/
    canonical.py              # 決定性排序/命名
    json_utils.py  retry.py
configs/{default.yaml, intents.yaml}
tests/{test_reader, test_writer, test_analysis, test_transform, test_equiv, test_protocol}
scripts/{run_local.py, eval_local.py}   # 本機把 prompt.txt 餵 stdin、比對 .log
```

### 4.1 原 `ai_agent/` 樹的 KEEP / REPURPOSE / DROP

| 原模組 | 處置 | 說明 |
|--------|------|------|
| `main.py` | **REPURPOSE** | 改成解析 `-config` + 跑 REPL |
| `app.py` | REPURPOSE | 併入 `io/protocol` + agent loop |
| `core/agent.py` | REPURPOSE | → `agent/router.py`（規則優先） |
| `core/planner.py` | **DROP** | 不做 LLM 規劃，流程是 deterministic |
| `core/executor.py` | REPURPOSE | → tool dispatch |
| `core/router.py` | REPURPOSE | → `agent/router.py` |
| `core/state.py` | REPURPOSE | → `agent/state.py`（裝 netlist 狀態 + 快照 + delta） |
| `core/types.py` | REPURPOSE | → `netlist/ir.py` 型別 |
| `llm/client.py` | **KEEP** | 雙 provider，必要 |
| `llm/prompt_builder.py` | REPURPOSE | → `llm/fallback.py`（tool catalog 提示） |
| `llm/output_parser.py` | REPURPOSE | → `llm/fallback.py`（JSON 驗證） |
| `llm/schemas.py` | KEEP | intent schema |
| `tools/base.py`,`registry.py` | REPURPOSE | → `agent/intents.py`（intent/tool 註冊表） |
| `tools/filesystem.py` | REPURPOSE | 只留 load/write design 檔案 I/O |
| `tools/shell.py` | REPURPOSE | → `equiv/*_bridge` 的 ABC/yosys 子行程呼叫 |
| `tools/web_search.py`,`code_runner.py`,`calculator.py` | **DROP** | 與題無關 |
| `memory/*`（5 個檔） | **DROP** | 「狀態」是 netlist 圖，不是對話記憶；由 `agent/state.py` 取代 |
| `workflows/research|coding|debugging.py` | **DROP** | 通用 agent flow，無關 |
| `workflows/base.py` | DROP（可選保留為 EDA flow base） | |
| `guards/policy.py`,`permissions.py` | **DROP** | 不需權限/政策 |
| `guards/validators.py` | **KEEP** | → hard-requirement 驗證器 |
| `observability/logger.py` | KEEP | 兼 `.log` 鏡像 |
| `observability/tracer.py`,`events.py` | DROP（debug 可選） | |
| `utils/config.py` | REPURPOSE | → `io/config.py` |
| `utils/json_utils.py`,`retry.py` | KEEP | |
| `configs/*`,`tests/*`,`scripts/*` | KEEP/REPURPOSE | |

**ADD（原樹完全沒有、但決定勝負的核心）**：`netlist/{ir,reader,writer,aig_export}`、`analysis/*`、`transform/*`、`optimize/abc_opt`、`equiv/{abc_bridge,yosys_bridge,gate}`、`io/protocol`、`agent/{router,intents,state}`、**DFF register-cut**。

---

## 5. 模組重點規格

### 5.1 io/protocol（協定 harness）
- `id` 從 1 起，**每收到一行非 EOF 就 +1**；line1 = begin-testcase = id1。
- 每幀格式：`#RESPONSE <id>\n` + body + `\n#END <id>\n`，**寫完 stdout 與 `.log` 都 flush**。
- case name：`case name is\s+(\S+?)\.`（non-greedy，避免吃到句點），**在寫 id1 之前**就開 `<case_name>.log`，id1 自己也要鏡像。
- EOF 紀律：用能分辨 `''`(EOF) 與 `'\n'`(空行) 的讀法；EOF 直接 break **不發幀**，避免 id drift。

### 5.2 netlist（自寫 IR + reader + writer）
- IR：struct-of-arrays（`g_type/g_out/g_in/g_name`）、獨立 dff 表（`clk=CK, rst_n=RN, set_n=SN, d=D, q=Q, name`，**逐 instance 讀，不假設極性**）、interned net 表、`driver[net]`、`loads[net]`（首次 fanout 查詢時建）。
- **PI/PO 從 port 宣告 + bus 展開取得**（`input [7:0] n6` → `n6[0..7]`），不要從 gate 用法推；要能表示 dangling 宣告 PI/PO（test09 `n2`、test15 `n3`）。
- gate 是 **output-pin-first**：`not g(out,in)`、`nand g(out,a,b)`。
- writer：保留 bus port 宣告原樣、output-pin-first、dff 還原 named-port、inserted gate 名稱決定性。

### 5.3 equiv/gate（防 0 分核心，兩層）
- **Tier-1 comb**（test01–20、所有 cone 級檢查）：IR→AIG，`cec before.aig after.aig`。
- **Tier-2 sequential**（test21–40）：yosys `equiv_make; equiv_simple; equiv_induct; equiv_status -assert`（權威）；或 driver-tracking register-cut 後 ABC cec（快但需處理 F8 的別名）。
- PASS 判定：body 含 `Networks are equivalent`（substring）**或** yosys `Equivalence successfully proven!`，**且**不含 `NOT EQUIVALENT`/`Verification failed`/`Error`。
- 任一 transform commit 前必過：等價閘 **AND** 結構界限（重算 max-fanout/max-depth ≤ K）**AND** basis 純度（gate type set ⊆ 請求 basis，且 grep 無 `zero/one/and/...` 雜質）。失敗 → rollback 到 pre 快照。

### 5.4 agent（規則優先 parser + LLM fallback）
- `agent/intents.py` 定義完整 intent 表（見 §6），每個 intent = trigger regex + 參數抽取 + tool 簽章；這份表同時就是「**我們描述給 LLM 的 EDA interface**」。
- LLM fallback 合約：system prompt = tool catalog + 「你只能輸出一個 `{"intent":..., "params":{...}}` JSON，不准對電路推理」；回傳後嚴格驗證（intent ∈ enum、必填參數齊全且型別對、引用的 signal/gate 必須存在於 IR）；失敗重試一次，仍失敗則安全 no-op（transform 回原圖、analysis 回 best-effort）。
- 以 `request string → intent` 持久化 cache，消除 temp 0.2 的不可重現性。

---

## 6. 任務 → 引擎 → 演算法/ABC → 驗證 對照表

> 全部 analysis 在自寫 IR 上算（決定性、可重現）；ABC/yosys 只做等價與 cost-ranked 合成。

### 6.1 Basic / Count / Report
| 請求類型 | 引擎/演算法 | 驗證 |
|---|---|---|
| begin / load / write design | protocol + reader + writer；load 時存 `original` 快照 | — |
| 各型別 gate 計數、total、某型別現量 | IR 線性掃描；transform 後對 current 算 | 與 grep 對照（已實測一致） |
| fanin/logic cone 的 gate 數、cone 型別分佈 | 反向 BFS（`cones.py`） | — |
| 列出某型別 gate 及其 I/O、PI/PO 位寬清單、PI/PO 數 | IR 查表 | — |
| 某 gate 型別 + pin 連接 | IR 查表 | — |
| 帶常數輸入的 gate（report→simplify→count 鏈） | **對當前狀態**算（F10）；delta 記在 state | 由前一個 simplify 記錄的 delta 回答 |

### 6.2 Path（**永不列舉**）
| 請求類型 | 演算法 | 備註 |
|---|---|---|
| A→B（避開 N）是否存在 | 移除 N 後 BFS/DFS 可達性 | yes/no |
| 列舉/完整列舉 A→B 路徑 | 先 **DP 精確計數**；count 小（≤ 數千）才 materialize，否則回 count + 有界樣本 | **格式需對齊參考答案**（見 §8） |
| len-0 路徑（PI 直連 PO） | 掃 driver=PI 且為 PO 的 net | |
| 每條 A→B 經過哪些 gate | 同上，受 count 限制 | |
| 是否每條 A→B 都過 gate G（dominator） | `!path_exists(A,B,avoid={G})` | |
| A,B 間 articulation points | dominator ∩ post-dominator（實測 91k gate 0.09s） | |
| wire W 是否為 cut | 移除 W 後是否仍有 PI→PO 可達 | 語意見 §8 |

### 6.3 Depth（**自寫圖、1 gate = 1 level，含 BUF/NOT，排除 cut 的 wrapper buf**）
| 請求類型 | 演算法 |
|---|---|
| max depth A→B / longest / critical | 前向 levelize 取 A 到 B 最長路 |
| cone 的 max depth / depth of cone of X | cone 內 levelize |
| 全域 max comb depth、PI→DFF-D、reg-to-reg | 對應 source/sink 邊界的 levelize |
| outputs depth > K 的數量、最深 output | 每 net 深度（實測 0.06s） |
| gate g0 是否在某 max-depth path 上 | `dep_from_src[g] + dep_to_sink[g] == global_max` |

### 6.4 Connectivity
| 請求類型 | 演算法 |
|---|---|
| PI fanout + 直接 load 清單、g0 驅動數、immediate successors | `loads[]` 查詢 |
| transitive fanin/fanout cone、reachable、shared cone | BFS |
| 連到 g0 output net 的 gate、cone gate 清單 | `loads[]` / 反向 BFS |
| 最高 fanout 的 PI、某訊號現在的 max fanout | `loads[]` 統計 |

### 6.5 Functional（ABC/SAT 輔助；目標多為小 support）
| 請求類型 | 演算法 | 驗證 |
|---|---|---|
| 兩內部訊號是否功能等價 | 各取 cone 建雙輸出網路，`&cec` | sub-second |
| output 是否恆 0、是否 depend on input Y | SAT / cone support 檢查 | |
| Boolean eqn / 以 PI 表示 | `cone X; collapse -B 1e6; write_eqn`，**包 10s timeout**；大 cone 退化回 factored/「過大」 | 實測目標 support 1–6，極快 |
| f 在 A,B 是否對稱 | cone 內 atomic swap 兩訊號 → `cec`（等價即對稱；變數不在 support 即 vacuously 對稱） | |
| 是否存在 (a,b) 使 NAND(a,b)≡target | 限縮 support⊆target 的候選 → 小 truth table 找 a&b==~T → **兩個獨立單輸出 AIG + `&cec` 確認**（**禁用 in-place XOR miter**，名稱碰撞會給假陽性） | test35 n25 實測找到 (n29459,n6359) |

### 6.6 Transform（rewrite / cleanup / buffer / naming）
| 請求類型 | 演算法 | 驗證 |
|---|---|---|
| XNOR→NOR、XOR→4NAND、XOR→AOI | **純 Python 結構模板**（by construction 正確） | 仍跑 cec 保險 |
| cone/全設計 remap 到 NAND+NOT / NOR+NOT / AND+NOT | ABC `strash; resyn2; read_genlib <basis>; map`，**自寫 writer 重序列化成 positional gate**；**先 sweep undriven net 去除 `zero` cell**；grep 驗純度 | cec + basis 純度 grep |
| NAND(const-1)→INV、常數傳遞 | 對當前狀態迭代 fixpoint | cec |
| 移除 dangling/floating/redundant | 反向可達性標記後刪未達 PO 者 | cec |
| 收合 NOT-NOT | 找 not→not 串接改接線 | cec |
| 合併重複 gate | **topo-order + union-find 到 fixpoint**（單趟 naive 會少算：test33 966 vs 正確 1970）；functional dup 用 sim-signature + cec | cec；count 由 delta |
| fanout≤K buffer | **平衡 buffer 樹**，每個 driver（含插入 buf）load ≤ K（inclusive），buf 數最小化 | 重算 max-fanout ≤ K + cec |
| 每 load 一 buf、reset fanout 降載 | 結構插入 | 同上 |
| rename gate/wire/signal | 更新所有 reference；companion「列出連到新名的 gate」 | by construction |

### 6.7 Optimize（cost-ranked，ABC）
| 請求類型 | ABC recipe | 驗證 |
|---|---|---|
| 最小化 max depth | `source abc.rc; strash; resyn2; if -g; st; if -g`（實測 test22 lev 39→14） | cec + 讀 `lev` 當 cost |
| 最小化 gate count | `strash; resyn2; dc2; resyn2`（最小 node） | cec + 讀 `nd/and` |
| 某 cone 深度最佳化 + 維持 basis | `cone X` 抽取 → 最佳化 → `map` 到 basis → 自寫 writer splice 回 | cec + basis 純度 |
| "Report original if already optimal" | 若 cost 未改善則回原圖（仍需過等價） | cec |

### 6.8 Sequential / Verify
| 請求類型 | 演算法 | 驗證 |
|---|---|---|
| 某 clock 下的 FF、reg-to-reg path | IR dff 表 + 圖走訪 | |
| D-input enable/hold 結構偵測 + 數量 | **精確結構樣式**（2:1 hold-mux / AND-hold，data 之一為自身 Q）；**定義敏感，見 §8** | |
| 證明 transform 後 == 變更前 / == 原始 / == 上次載入 | 對應快照跑等價閘（comb→cec / seq→yosys equiv） | F4/F5 |

---

## 7. 防 0 分檢查清單（critic 在真實 testcase 上重現的 blocker）

- [ ] **B1 basis 純度**：basis 輸出前 sweep undriven net；輸出後 grep gate type，出現 `zero/one` 或 basis 外 token 即 FAIL → rollback。
- [ ] **B2 答案檔格式**：永不 ship ABC `write_verilog`；用自寫 structural writer（保留 bus 宣告、output-pin-first、named-port dff），最後用自己的 parser 回讀 + cec 自檢。
- [ ] **B3 DFF cut 健壯性**：driver-tracking IR；處理 (a) 多 dff 共用 Q、(b) Q==PO、(c) D==宣告 PO（`Repeated CO names`）；cut 後自動 `read_verilog` 檢查，失敗 fallback yosys equiv / 純 Python。
- [ ] **B4 等價字串**：substring 比對 `Networks are equivalent`（含 `after structural hashing`）/ yosys proven，並排除失敗字串。
- [ ] **B5 flush / cwd**：每幀 flush stdout+log；ABC 一律絕對路徑 + `source abc.rc`（或 chdir），啟動時自測 `resyn2` 可用否。
- [ ] **B6 fanout bound**：`≤ K`（inclusive）、per-driver 平衡樹，commit 前重算 max-fanout。
- [ ] **B7 id / log**：每非-EOF 行一幀；EOF 不發幀；`.log` 名去尾句點、id1 也鏡像。
- [ ] **B8 計數正確性**：合併重複用 fixpoint union-find；delta 記在 state、由後續 "how many" query 讀取。

---

## 8. 必須先「對答案/對格式」鎖定的語意問題（exactly-correct-or-0，架構已預留 hook）

這些不是架構問題，而是**語意/格式約定**，建議用官方 sample 或主辦回覆鎖定後再凍結；在此之前用文件中的預設並集中在 `agent/intents.py` 一處可調：

1. **DFF-Q 輸出的語意（最高頻風險）**：當 `output X` 是 dff 的 Q net（test33 n8/n9、test39 n12、test40 n10[9] 都是），其 cone/equation/depth/path 應**重導到該 Q 的 next-state D net cone**（register-cut 會讓它變成空 cone/PI）。預設：Q=深度 0 source、D=sink，Q-output 查詢一律走 D-cone。
2. **path 列舉輸出政策**：test14 單一 A→B 可達 ~2e5–3e5 條。grader 要「全列」還是「計數」？格式（gate 序列 vs net 序列、箭頭樣式、排序）為何？預設：count 小才全列，否則 count + 有界樣本。
3. **enable/hold 定義**：test40 廣義 2373 vs 嚴格 2219（差 154）。預設採「透過 mux/AND 且 data 之一為自身 Q」的結構樣式（貼合題目字面）。
4. **depth 約定**：gate-level（每 gate +1，含 BUF/NOT，PI→PO 直連 = 0），用自寫圖、**排除 cut 的 wrapper buf**、不用 post-strash AIG；reg-to-reg 是否計入 launch/capture FF 待確認。
5. **duplicate 定義**：structural-hash（canonical 代表輸入）vs exact-input；影響 "how many merged"。
6. **`-config` YAML 實際 key 命名**：目前只依 PDF（provider / openai.model / anthropic.model / generation.temperature / max_output_tokens）；因 LLM 只是 fallback，風險較低。

---

## 9. 開發優先序

1. **地基**：`netlist/{ir,reader,writer}` + `io/protocol` + `io/config` → 能 load/write 並 round-trip 自檢（reader→writer→reader 一致）。
2. **等價閘**：`netlist/aig_export`（driver-tracking register-cut）+ `equiv/{abc_bridge,yosys_bridge,gate}` → 先把防 0 分骨架立起來。
3. **analysis 引擎**（量大但單純，且多數 testcase 前段都是 analysis）：counts→depth→cones→paths→functional→sequential。
4. **transform/optimize 引擎** + `guards/validators`（每步等價/界限/basis）。
5. **agent/router 規則表** + `llm/{client,fallback}`（fallback 最後接）。
6. `scripts/eval_local.py` 把每個 `prompt.txt` 餵 stdin、收 `<case>.log`，建立回歸；對 §8 的語意逐一鎖定。

---

*附：本文件結論由背景 grounding workflow（5 個實測 agent + 2 個對抗式 critic，共 325 次工具呼叫）在真實 testcase 上驗證得到，重點實測指令與數據見對話紀錄。*
