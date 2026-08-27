# Golden 使用指南(final submit 對答案 SOP)

> Schema 與判分慣例的完整說明在 [README.md](README.md);本文件只講「怎麼用」。

## TL;DR

```bash
cd beta_check/golden
python3 check_log.py <你的log資料夾>            # 全部 91 題
python3 check_log.py <你的log資料夾> --case test76   # 只看一題
python3 check_log.py <你的log資料夾> --verbose       # 列出所有 REVIEW 行
```

log 資料夾需含 `test01.log`~`test91.log`(或 `.txt`),內容為 `#RESPONSE N ... #END N` 格式。

## 輸出怎麼讀

| 結果 | 意義 | 該做什麼 |
|---|---|---|
| `PASS` | 規則自動判定通過 | 不用管 |
| `FAIL` | 必含數字/名稱缺失 | **逐條人工確認**,幾乎都是真丟分 |
| `REVIEW` | 無法自動判定 | 對照該行 golden 的 `reference`/`notes` 人工判 |

REVIEW 固定包含約 33 行 `team_dependent`/`opt_qor`/`llm_judge`(本質上就要人工/LLM 判),
加上少量語意模糊的 yes/no。健康的 log 大約 PASS 640+/FAIL 0/REVIEW 35(Team7、Team3 實測皆 0 FAIL)。

## 對答案流程建議(final 前)

1. **跑 checker**:先把 FAIL 清到 0。每個 FAIL 打開對應 `testNN.json` 該行,
   看 `golden.reference`(標準答案)和 `notes`(判分指引、可接受的替代表述)。
2. **掃 REVIEW**:
   - `team_dependent`:確認回答有「成功 + 等價驗證」聲明,且數字與自家 out.v 一致
     (可用 `final_vs_team7_report.md` 裡的 netlist 實測手法驗)。
   - `opt_qor`:跟 `reference` 裡的 best observed 比,差太多就是優化沒做好。
   - `llm_judge`:對照 `reference` 人工判,或丟給 LLM 判。
3. **醒目行專查**(beta 已知失分模式,final 高機率重考):
   - 所有 cone 類問題 → 回答必須含邊界 DFF(數量+實例名)
   - 所有「數量」問題 → 完整句子,不要裸數字
   - transform 類 → 回報數字必須跟 out.v 實際 diff 一致
4. **confidence=medium/low 的行**(33 行)不要盲信 golden,`notes` 都寫了為什麼不確定。

## 適用範圍與限制

- Golden 是照 **beta 題目**建的。final 題目有重用 beta 電路+同款問題(如 beta test89 原題重現),
  這部分可直接對;全新題目要先用 prompt 文字把 final 行對映到 golden 行,對不上的行沒有 golden。
- checker 是**文字層篩選器,不是官方評分器**:
  - 語意寬鬆(ack/宣告句自動過),所以 PASS ≠ 保證得分;FAIL 才是強訊號。
  - 看不到 netlist 層的失敗(out.v 等價驗證掛掉、fanout bound 沒達成),
    transform 題請務必另外驗 out.v(參考 `final_vs_team7_report.md` 的驗法)。
- 已知未解:test91 L7/L8 enable/hold 的官方數法(候選 721/1474/1796/1817),
  final 若再考同款題,建議把結構樣板(mux/AND enable)的判定式跟 721 對齊研究後再作答。

## 檔案索引

| 檔案 | 用途 |
|---|---|
| `check_log.py` | 自動對答案腳本(本文件主角) |
| `golden_all.json` | 91 題合併 golden(程式讀這個) |
| `testNN.json` | 單題 golden(人工查閱方便) |
| `README.md` | schema、rule 類型、判分慣例 |
| `_manifest.json` | 各隊每題 beta 得分(權重依據) |
