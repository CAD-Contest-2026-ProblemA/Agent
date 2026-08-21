# Targeted 轉換缺口：「Replace all OR gates in the cone」會多轉換無關 gate

> 狀態:**已決議採方案 B 並實作完成**(2026-08-22)。§6 的 bank test38:19 修正仍待處理。
> 相關:beta test25/test27 r4、題目書 §4.3 範例、QA A21.2。
> 驗證:targeted 路徑(規則/LLM-dispatch 雙路)、無 only_types 零回歸、beta 空 cone 回 0、
> evaluator 5 案 HARD 全過且本改動零 golden drift。

## TL;DR

「把 cone 裡的 **OR gate** 換成 NAND+NOT」是一個 **targeted(定點)轉換**:只動 OR、
其他 gate 保留。我們的 op 語彙沒有 `or_to_nand`,這句話目前映射到
`convert_basis {basis: nand+not, scope: cone}` —— 它會把 **scope 裡的每一個 gate**
(AND/XOR/NOR…)全部改寫成 NAND+NOT,不只 OR。等價不會壞、basis 檢查會過,
但「多做的改寫」會在後續分析題與結構性評分上失分。
建議在 final 前補上 targeted 能力(方案 B:`convert_basis` 加 `only_types` 參數)。

## 1. 觸發這個問題的題目(原文)

beta 0813 有兩題,同一句 wording(test25 r4、test27 r4):

```
Replace all 2-input OR gates in the cone of n11[0] with equivalent logic
built only from NAND and NOT gates. Ensure the design functionality does not change.
```

test27 後面還接著:

```
r5  Eliminate unused logic gates from the netlist. ...
r6  Perform depth optimization ... while ensuring the cone of n11[0] continues
    to use only NAND and NOT gates. ... cost = maximum logic depth ...
```

這句直接抄自**題目書 §4.3 的範例**("Replace all 2-input OR gates in the cone of
flag with equivalent logic built only from NAND and NOT gates."),hidden case
再出同型題(可能換成 AND/NOR/任意型別)的機率很高。

## 2. 題目要的 vs 我們做的

| | 題目字面語意(targeted) | convert_basis(現況) |
|---|---|---|
| cone 裡的 OR | 換成 `NAND(NOT a, NOT b)` | 換掉 ✓ |
| cone 裡的 AND/XOR/NOR… | **原封不動** | **一併被改寫成 NAND+NOT** |
| 功能等價 | ✓ | ✓ |
| cone 的 basis | 不保證變純(其他型別還在) | 變成純 NAND+NOT(比要求強) |

具體例:cone = { OR g1, AND g2, XOR g3 }。題目要求後 cone 應是
{ NAND/NOT×(g1 的展開), **AND g2**, **XOR g3** };convert_basis 後三個都變 NAND+NOT。

## 3. 會怎麼失分(cone 非空時)

1. **後續分析題**:「How many **AND** gates are in the cone now?」——
   最小解讀預期「不變」,我們會答 0。
2. **結構性評分**:若 grader 檢查「非 OR gate 應保留」→ fail。
3. **cost 劣化**:NAND+NOT 展開通常讓 cone 變深/變大,後面接 depth/area cost
   的題(如 test27 r6)對排名不利。

反過來說,它**不會**壞的:功能等價(cec 有護欄)、「cone 無 OR」、
「cone 為 NAND+NOT」這類檢查——都是超額滿足。

## 4. beta 0813 實測:目前兩題都無害(已驗證)

用獨立引擎(val-harness parser)量測 beta 網表:

- `n11[0]` 由 **DFF g229 的 Q 直接驅動**(registered output)。
- 依 **QA A21.2**,組合 cone 止於 DFF.Q 邊界 → **cone(n11[0]) = ∅(0 個 gate)**。
- 正確行為就是「沒有 OR 可換,設計不動」。
- 我們的 scope 解析(`_scope_from_param`)對「存在的 net、空 cone」回**空集合**
  (cd3229b 的 whole-design fallback 只在 scope 指到不存在的 net 時觸發)→
  `to_basis` 改寫 0 個 gate → 回覆 "0 gates rewritten; equivalence verified"。✓
- test27 r6 的「cone 維持 NAND+NOT」對空 cone 空虛滿足。✓

**所以 beta 這兩題不會出事;風險全部集中在「hidden case 出一題 cone 非空的同型題」。**

## 5. 解決方案

### 方案 A:新增 targeted op `or_to_nand {scope}`

比照現有 `xor_to_nand` 的模式(rewrite template + op + catalog 條目 + bank 例句)。
缺點:只解決 OR;hidden 若出「replace all **NOR/AND/XNOR** gates with …」又要再加。

### 方案 B(建議):`convert_basis` 加 `only_types` 參數

```
convert_basis {basis: [...], scope: X, only_types: ["or"]}
```

- `rewrite.to_basis()` 加一個型別過濾:只改寫 `only_types` 內的 gate,其餘保留。
- 一次覆蓋所有「replace all <型別> gates with <basis>」變體(含未見過的組合),
  也不增加 intent 數量(catalog 已經很擠,LLM routing 的混淆成本低)。
- 現有呼叫不帶 `only_types` → 行為不變,零回歸風險。

需要動的檔案(估計半天內含驗證):

1. `cada/transform/rewrite.py`:`to_basis(nl, basis, scope_gates, only_types=None)`
2. `cada/agent/agent.py`:`op_convert_basis(basis, scope, only_types=None)` 傳遞參數;
   規則路徑 `h_basis` 從句中抽型別
3. `cada/llm/allowed_intents.py`:catalog 條目說明 `only_types` 的觸發語
   ("Replace all <TYPE> gates in ..." → 帶 only_types;"use only <basis>"
   整體重建 → 不帶)
4. `cada/llm/public_examples.jsonl`:補 2-3 條帶 `only_types` 的例句
   (test25/27 r4 就是現成素材)
5. 驗證:beta test25/27 重放(應維持 0 rewritten)+ 造一個 cone 非空的
   local case 確認只動目標型別、等價通過

## 6. 附帶一:bank 有一筆標注與 catalog 矛盾(需修,與本缺口無關但同場討論)

`cada/llm/public_examples.jsonl` 的這一筆:

```json
{"case": "test38", "line": 19,
 "text": "Compute the fanin logic cone of output n14 and list all gates that contribute to this output.",
 "op": "cone_gate_count", "params": {"output": "n14"}}
```

問題:這句要求 **LIST** cone 裡的 gate,而 catalog 自己有兩處明確規定列名單要用
`transitive_fanin`:

- `allowed_intents.py:112`:「If it asks to **LIST** the gates in that cone,
  use **transitive_fanin** instead.」
- `allowed_intents.py:430`(ANSWER SHAPE is a parameter):
  「Applies to: **transitive_fanin**, transitive_fanout, successors」,
  且例句正是同句型:"List the instance names of all gates in the fanin cone of
  N426." → `transitive_fanin {net, form:"list"}`。

影響:bank 是 retrieval 的教材——這句 wording 在 beta 也出現(test82 r8),
檢索到這筆錯例會**直接教 LLM 選錯 op**,而且與 catalog 的規則互相矛盾,
正是我們這週報告的「system prompt 前後矛盾」同類問題,只是發生在 bank 與
catalog 之間。

建議修法:把該筆的 `op` 改為 `transitive_fanin`、params 改為
`{"net": "n14", "form": "list"}`(兩個 op 算的是同一個 gate 集合,
改的只是 routing 歸宿,不影響任何 golden 數字)。
`beta0813_question_function_map.xlsx` 中 test82 r8 已按 catalog 修正為
`transitive_fanin`。

## 7. 附帶二:另外兩個「近似覆蓋」項(嚴重度低,一併知悉)

1. **「any redundant gates … remove them if found」→ `remove_dangling`**:
   若 "redundant" 被解讀為含 structural duplicates,現在只清 dangling。
   低風險;可在 handler 對 "redundant" 字樣附帶跑 `merge_duplicates`(一行)。
2. **「insert buffers … cost = total gate count」→ `insert_buffers`**:
   覆蓋沒問題;得分取決於插 buffer 的省不省(品質項,已由 buffering 策略處理)。

## 附錄:本次驗證方法

- beta 0813 共 91 案、680 行 prompt;扣除 begin/load/write 273 行,
  407 個問題行去重為 **163 個 unique 模板**,逐一人工比對 op 語意與
  catalog 裁決規則;參數層(scope/basis/mode/kind/form)逐 op 檢查簽名。
- 過程中修正 2 個映射(test31 r5 → `minimize_depth`;test82 r8 → `transitive_fanin`),
  並發現 bank 一筆與 catalog 矛盾 → 詳見 §6。
- 結果檔:`beta0813_question_function_map.xlsx`(407 行,exact 384 /
  template 3 / manual 20,UNMATCHED 0)。
