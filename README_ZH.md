# CADA — LLM 輔助的網表探索與轉換

**ICCAD 2026 競賽 Problem A** 的參賽實作。系統從 stdin 讀取自然語言請求,逐句解讀,在 gate-level Verilog 網表上執行對應的分析 / 轉換 / 最佳化,並把答案以 `#RESPONSE <id>` … `#END <id>` 的格式輸出(同時鏡像到 `<case_name>.log`)。

> English version: see [README.md](README.md).

## 一句話設計

**一個 deterministic 的 gate-level EDA 引擎,外面包一層極薄的自然語言前端。** 規則式 regex router 先把(高度模板化的)請求對應到一個結構化 intent;LLM 只當 fallback,負責把「沒對應到規則」的句子翻成一個 `{"intent", "params"}` JSON,**絕不對電路做推理**。所有攸關正確性的工作(parse、分析、轉換、等價驗證)都是 deterministic 的 Python;ABC / yosys 只當「等價判定」與「成本排名合成」的 oracle。

## 環境需求

* Python 直譯器 — **3.8 以上**(benchmark 路徑本身只用標準函式庫)。
* **`abc`(必要)** 與 **`yosys`(備援)** 在 `PATH` 上(或用環境變數 `ABC_BIN` 指向 ABC 執行檔)——當作等價 / 最佳化後端。注意:`setup.sh` / pip / uv **都不會幫你裝這兩個外部執行檔**,要自己確認機器上有。實測 ABC 被 24/40 個 testcase 使用(test01–16 純 Python 不碰 ABC);yosys 只在 ABC 無法判定時當 fallback,實測 0/40。
* 選用的 Python 套件:`openai` / `anthropic`(只有 LLM fallback 會用到;benchmark 不會呼叫 LLM)、`PyYAML`(設定檔解析,沒裝會自動退回內建 mini-parser)。

## 取得程式碼

要把 repo 內容**直接放進目前的資料夾**(不要多一層 `Agent/` 子資料夾),在結尾加上 `.` 當目標即可(目前資料夾必須是空的):

```bash
mkdir my-submission && cd my-submission
git clone https://github.com/CAD-Contest-2026-ProblemA/Agent.git .
```

或是 clone 進你指定名稱的資料夾:

```bash
git clone https://github.com/CAD-Contest-2026-ProblemA/Agent.git <資料夾名> && cd <資料夾名>
```

(如果目標資料夾不是空的,`git clone … .` 會拒絕;這時先 clone 到暫存資料夾再把內容搬過去。)

## 快速開始(建議用 uv)

`uv` 會抓一份獨立、現代版本的 Python(用舊 glibc 編譯,連舊的競賽機器都能跑),完全不依賴系統 Python:

```bash
bash setup.sh        # 一次性:uv 建立 .venv(內含 Python 3.12,並裝可選相依套件)

# 用跟競賽 harness 一模一樣的方式執行:
./cada0001_alpha -config configs/default.yaml < testcase/test01/prompt.txt
```

`cada0001_alpha` 啟動器會優先用 `.venv/bin/python`;若沒有 `.venv` 但機器上有 `uv`,會自動 bootstrap 一次;再不行就退回系統的 `python3`。

### 不用 uv 的情況

```bash
pip install -r requirements.txt     # 只有要用 LLM fallback 才需要
./cada0001_alpha -config configs/default.yaml < testcase/test01/prompt.txt
```

benchmark **完全可離線執行、不需任何第三方套件**——config parser 內建了一個 mini-parser fallback,而 regex router 已涵蓋全部 40 個 testcase(在這些測資上 LLM 完全不會被呼叫)。

## 設定檔(`-config`)

格式與競賽題目 Figure 6 相同。把你的 API key 填進 `configs/default.yaml`:

```yaml
provider: "openai"          # 或:"anthropic"
openai:
  api_key: <YOUR_API_KEY>
  model: "gpt-4o-mini"
anthropic:
  api_key: <YOUR_API_KEY>
  model: "claude-haiku-4-5"
generation:
  temperature: 0.2
  max_output_tokens: 4096
```

## 環境自檢(doctor)

評測前先檢查環境裝齊了沒:

```bash
./cada0001_alpha --doctor        # 或:python3 -m cada.doctor / python3 scripts/doctor.py
```

依序檢查:**Python**(先看有沒有裝 uv → 有的話看 `.venv/` 在不在 → 套件都查 `.venv` 裡的;只要 uv 或 `.venv` 不在,這段直接 fail,**不會去檢查 host 的 Python**);**外部工具**(`abc` 必要、`yosys` 備援,用跟 agent 一樣的順序解析路徑後**實際執行一次**,所以「檔案在但跑不起來」例如 glibc/架構不合也會被抓出來);**設定檔**與選用的 LLM key;以及 **agent 套件本身**(import + 解析一個 testcase)。有 hard failure 時 exit code 非 0。

## 指定外部工具位置(不靠 $PATH)

在 **`configs/tools.yaml`**(會自動載入)裡指定 `abc`/`yosys`(或其他工具)的絕對路徑,就不必把它們放進 `$PATH`:

```yaml
tools:
  abc: /home/me/abc/abc          # 指向真正的執行檔(abc.rc 要跟它放一起)
  yosys: /usr/local/bin/yosys
```

解析順序(後者覆蓋前者):`configs/tools.yaml` → 環境變數(`ABC_BIN`、`YOSYS_BIN`、或 `CADA_<NAME>_BIN`)→ `-tools <file>` → `-config` 檔裡的 `tools:` 區段 → 內建候選路徑 → `$PATH`(最後手段)。若設定的路徑不存在,會自動往下一個來源退,不會直接壞掉。

## 本機測試

```bash
python3 scripts/run_local.py testcase/test22          # 單一 case,輸出到 stdout
python3 scripts/run_local.py --all                    # 全部 case
```

## 架構

```
stdin ─► io_/protocol ─► agent/agent (regex router;LLM fallback)
                              │
            ┌─────────────────┼──────────────────────────┐
            ▼                 ▼                           ▼
       analysis/*        transform/*  ── guards ──►  optimize/abc_opt
   counts depth paths    rewrite constprop           (成本排名,ABC)
   cones connectivity    cleanup buffering naming           │
   functional sequential        │                           │
            │                    ▼                           ▼
            └────────►  netlist/ir  (唯一真相來源) ◄──────── equiv/gate
                         reader · writer · blif_export      (ABC cec / yosys)
                              │
                  #RESPONSE/#END ─► stdout + <case>.log  (每幀都 flush)
```

| 領域 | 模組 | 用途 |
|------|------|------|
| IR | `netlist/ir.py` | flat gate-level 網表;gate、named-port DFF、driver/loads、快照 |
| 解析/輸出 | `netlist/reader.py`, `netlist/writer.py` | 自寫 Verilog parser + canonical structural writer(可完全 round-trip) |
| 等價 | `netlist/blif_export.py`, `equiv/*` | register-cut BLIF → ABC `cec`;yosys 備援 |
| 分析 | `analysis/*` | 計數、cone、depth、連通性、path(DP 計數,永不列舉)、functional(ABC/SAT)、sequential |
| 轉換 | `transform/*` | basis remap、XOR/XNOR 分解、常數傳遞、dangling 移除、fixpoint 重複合併、buffer 樹、改名 |
| 最佳化 | `optimize/abc_opt.py` | 用 ABC + 單位延遲 genlib mapping 做深度/面積最小化,保 basis,過 cec |
| Agent | `agent/*` | 規則 router、請求狀態 + 快照 + transform delta |
| LLM | `llm/*` | 輕量雙 provider client + 帶 cache 的 fallback 翻譯器 |

## 正確性模型

* 每個 structural transform 都是 **by-construction 等價保證**,commit 前還會再過一次 register-cut combinational `cec`;一旦違反就 rollback 回轉換前的快照。
* basis 轉換用 **純 Python 模板**(不走 ABC technology mapping),所以經典的 ABC `zero`/`one` cell 洩漏永遠不會跑到輸出,basis 純度有保證。
* path 用 **拓樸 DP 精確計數**(big-integer 不溢位),只有在數量低於門檻時才實際列舉——絕不 materialize 整個 path 集合。
* depth 用題目的定義量測(1 個 gate = 1 層,含 inverter);最佳化時 map 到單位延遲 library,讓 ABC 最小化的就是同一個指標。

驗證:全部 40 個 testcase 都能端到端跑完、framing 正確、每一行 prompt 都對應到一個明確的 EDA 操作,而且每個輸出網表都經 ABC 驗證與輸入功能等價。

## 備註 / 可調語意

有幾個「答對才有分」的語意約定(path 列舉的輸出格式、「wire 是不是 cut」的定義、enable/hold 的寬鬆 vs 嚴格計數、Boolean equation 以暫存器狀態為葉節點)集中在一處並有註解;若主辦的判定字面不同,可對著官方 sample 調整。DFF 採通用的 named-port 形式 `dff(.RN,.SN,.CK,.D,.Q)`,逐 instance 處理非同步 active-low reset(RN)與 set(SN)(reset 優先於 set)。
