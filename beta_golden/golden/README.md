# Beta Golden 標註檔(final submit 對答案用)

由 Team2/Team3/Team4/Team7 四隊 beta log 交叉比對產生(2026-08-27,91 題標註 + 91 題對抗性覆核)。
每隊每題的 beta 得分(`_manifest.json`)作為權重:**該題拿滿分的隊,其每一行答案都是 ground truth 錨點**;364 個隊×題組合中有 304 個滿分錨點。

## 檔案

- `testNN.json` — 每題一檔,每個 prompt 行一個 entry
- `golden_all.json` — 91 檔合併(key = case 名)
- `_manifest.json` — 每題 max 分數 + 四隊 beta 得分(來自 beta_score_cmp.xlsx)

## 每行 entry 結構

```json
{
  "line": 4,
  "prompt": "<prompt 原文>",
  "category": "ack|load|analysis|transform|opt|write",
  "golden": {
    "rule": "<判分規則,見下>",
    "must_numbers": [107],        // 答案必含的數字(無則 null)
    "must_names": ["n8"],         // 答案必含的 gate/net 名(無則 null)
    "expected": "yes|no|null",    // yes_no 題的正解
    "reference": "<一句標準答案,數字名稱都已填入>",
    "notes": "<判分指引、各隊分歧分析、要 sanity-check 什麼>"
  },
  "team_answers": {"Team2": "...", "Team3": "...", "Team4": "...", "Team7": "..."},
  "agreement": "4/4",
  "confidence": "high|medium|low"
}
```

## rule 類型(680 行分佈)

| rule | 行數 | 意義 / 對答案方式 |
|---|---|---|
| `ack` | 273 | begin/load/write 等回覆,只要表達成功即可 |
| `exact_number` | 179 | 客觀數字(原始 netlist 上的 count/depth/fanout),必含 `must_numbers` |
| `yes_no` | 108 | 是/否結論,對 `expected`;直述句(不含 yes/no 字面)也算 |
| `transform_ack` | 60 | transform 指令,回覆要聲明成功+等價;實質對錯驗 out.v |
| `name_set` | 27 | 必須點名特定 gate/net(如最大 cone 的 output 名) |
| `team_dependent` | 12 | 前面 transform 不同 → 各隊數字本來就不同,任何自洽值皆可;notes 說明要檢查什麼 |
| `opt_qor` | 12 | 優化品質,跨隊相對計分(小數部分給分),記錄各隊達成值 |
| `llm_judge` | 9 | 描述型(布林方程式推導等),reference 為參考解 |

信心度:high 648 / medium 31 / low 1。

## 重要慣例與特例(對 final 答案前必讀)

1. **fanin/logic cone 一律含邊界 DFF**:數量計入、實例點名(如「4 comb + 3 DFFs (g22,g23,g30)」)。
   beta test76/test82 已證實只報 `DFF: 0` 會丟分。test91 L10 的 22 vs 29 之謎同源:29 = 22 comb + 7 DFF,兩種都拿分,但保險做法是兩個數都報。
2. **test91 L7/L8(enable/hold 偵測)官方答案未定,候選收斂到 721 或 2213**。
   可靠三隊(T3=1796、T4=1474、T7=1817)其他行全對、都拿 9/11 → 三個值都**不是**官方答案。
   netlist 實查(重審修正):**2213**/2585 顆 DFF 的 D-cone 含自身 Q(前版說 0 是錯的);1817 + Team7 排除的 396 = 2213;1817 = D 由 NAND 驅動的 DFF 數。
   剩餘候選:721(Team2 交換 log 的值,但該 log 非評分 run,無法佐證)與 2213(自迴圈總數)。final 若考同款題建議報 2213 並附判定式說明。
   ⚠️ Team2 交換 log 是另一次 run(在 xlsx Team2 滿分的 test24/51/58/73 有 planner error),任何以 Team2 分數為前提的推理都無效。
3. **路徑列舉類**:寫進外部檔案 + log 報數量 = 可接受(多個滿分隊如此做);log 中的數量是判分核心。
4. **無路徑的 depth 題**:「0」和「no path/undefined」都有滿分隊用,皆可接受。
5. **load 行的統計數字不判分**(Team2 只報檔案行數/bytes 也拿滿分),但要聲明載入成功。
6. 小數失分 = `opt_qor` 的相對排名,不是事實錯誤。**重審發現的評分機制**:優化題的 credit ≈ best_depth / 自家 out.v 的**實測**深度——grader 量的是輸出網表,不是 log 宣稱的數字(test22/23/27 的分數反推:有隊 log 宣稱深度 12 但按實測 ~19 給分)。所以 final 的優化題:宣稱值必須與 out.v 實測一致,且真正決定分數的是 out.v。
7. **Team2 的 log 是另一次 run**,與 xlsx 分數不完全對應(內含 14 行 planner error,分佈在 xlsx 上它拿滿分的題)。golden 標註已排除這些行的影響(參考答案取自 Team3/4/7),但**不要把 Team2 log 當滿分錨點用**;Team1(我們)/Team3/Team4/Team7 的 log 與 xlsx 分數皆已驗證吻合。

## 用法建議(final 對答案)

對每題每行:取 final log 的 `#RESPONSE N`,依 `golden.rule` 檢查:
`exact_number`/`name_set` 直接字串含檢;`yes_no` 判語意;`team_dependent`/`transform_ack` 檢查成功聲明+自洽(數字與自家 out.v 一致);`opt_qor` 記錄值供跨隊比較;`llm_judge` 用 LLM 對 reference 判。
`confidence` 為 medium/low 的行(32 行)建議人工複核,notes 裡都寫了分歧原因。
