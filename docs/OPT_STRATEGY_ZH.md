# minimize\_depth 優化策略測試報告

## 背景

本次測試針對 test21–40 中所有需要 `minimize_depth` / `optimize_cone_depth` 操作的測資，
使用預先快取的 LLM intent 映射（`intent_cache_21_40.json`，共 192 筆）執行 replay 評估，
完全不需呼叫 GPT API。

---

## Baseline OPT 結果

| 測資 | OPT 指標 | 結果 | 備註 |
|------|----------|------|------|
| test22 | global depth | **41** | 原始深度即 41 |
| test23 | global depth | **33** | 原始深度即 33 |
| test24 | global depth | **25** | 原始深度即 25 |
| test26 | cone depth (n10) | **0** | n10 為 PI→PO 直通線，無邏輯 |
| test27 | cone depth (n15) | **0** | n15 為 PI→PO 直通線，無邏輯 |
| test28 | global depth (AND_NOT) | **130** | 原始 67，AND_NOT 轉換後理論下界 ≈134，優於下界 |
| test29 | global depth (AND_NOT) | **316** | 原始 200，AND_NOT 轉換後 498，折疊後 316 |
| test30 | global depth (AND_NOT) | **23** | 原始 24，AND_NOT 折疊後改善為 23 |
| test33 | cone depth (n8) | **0** | n8 為 PI→PO 直通線，無邏輯 |
| test40 | cone depth (n14, NAND_NOT) | **2** | |

---

## 各優化策略測試結果

### 策略一：前處理（const_propagate + remove_dangling + collapse）
**測試方法**：在 `minimize_depth` 呼叫 ABC 之前先執行 const_propagate + collapse_double_inverters。  
**結果**：depth 不變，僅減少 gate 數量（test29: 4318→3903 gates，depth 維持 316）。  
**結論**：❌ 無效果。

### 策略二：genlib inverter cost = 0
**測試方法**：修改 genlib 讓 NOT gate 無深度成本，引導 ABC 更自由插入 NOT。  
**結論**：❌ 理論有誤——NOT gate 在本題 cost function 中確實佔一個 level，設為 0 會使 ABC 的深度模型與實際評分不一致。未實作。

### 策略三：擴充 ABC recipe（加入 dc2）
**測試方法**：`DEPTH_RECIPE = ["resyn2", "dc2", "resyn2"]`。  
**結果**：所有測資結果與原 `["resyn2", "resyn2"]` 完全相同。  
**結論**：❌ 無效果，且增加 runtime。已還原。

### 策略四：先用完整 basis 優化，再轉 basis（暫時突破 basis 限制）
**測試方法**：`optimize_comb(basis=None)` → `_finalize(basis="AND_NOT")` vs `optimize_comb(basis="AND_NOT")`，取較佳結果。  
**結果**：所有測資結果完全相同。  
**原因**：ABC 以完整 basis 優化後，`to_basis()` 轉換把省下的深度全數加回。  
**結論**：❌ 無效果。已還原。

---

## 深度分析

### AND_NOT 轉換鏈路（test29 為例）

```
原始: depth=200（NAND/NOR 混合）
→ to_basis(AND_NOT): depth=498（約 2.5× 膨脹）
→ collapse_double_inverters: depth=316（核心：大量 NOT-NOT 鏈折疊）
→ merge_structural_duplicates: depth=316（無變化）
→ ABC minimize_depth: depth=316（ABC 無法改善）
```

`collapse_double_inverters` 是 AND_NOT 路徑的關鍵：NAND→NOT(AND) 後若緊接 NOT，
即可折疊回 AND，大幅消除多餘層次。多輪迭代測試顯示一輪即收斂。

### 為何 ABC 無法進一步改善 316？

test29 的 AND_NOT 電路深度 316 已在 ABC 280 秒 timeout 內達到局部最優。  
test28 的結果 130 < 134（2×67 理論下界）更優於理論值，說明 ABC 對此類電路效果已相當好。

---

## 結論

**目前所有 test21–40 的 minimize\_depth 結果均已接近最優。**  
四種策略均無法改善 OPT 分數。現有 `abc_opt.py` 實作維持不變（DEPTH_RECIPE = ["resyn2", "resyn2"]）。

---

## 相關檔案

- `intent_cache_21_40.json`：192 筆 prompt → intent 映射快取
- capture script: `/tmp/.../scratchpad/capture_cache.py`（建置 cache 用，已完成）
- replay benchmark: `/tmp/.../scratchpad/bench_opt.py`（無 API replay 用）
