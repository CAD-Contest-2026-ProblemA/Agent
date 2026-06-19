# 官方 Q&A 對照 — 待調整清單(TODO)

來源:`problem/A_QA_20260615.pdf`。這份記錄「官方怎麼說 vs 我們現在怎麼做 vs 要不要改」。
優先序:🔴 高(可能丟分)、🟡 中、🟢 低/邊界。

> **狀態(更新):下面 1~5 + doctor 卡死 已全部實作完成。** 變更摘要:
> 1. ✅ write 輸出到 design_dir(輸入同目錄);evaluator sandbox 改用 copytree 避免污染。
> 2. ✅ cone/depth 把 DFF.Q 當 PI(`redirect_q` 預設改 False)。
> 3. ✅ 路徑「完整列舉」與大型 gate 清單改寫檔案、回應給路徑(超過門檻)。
> 4. ✅ constant 改為「功能常數」(模擬 + cec 確認;test39 找到 3 個結構檢查會漏的)。
> 5. ✅ fanout 計數含 primary-output 連線(查詢用;buffer 守門員用 pin-only)。
> 6. ✅ doctor 在 frozen binary 不再卡死(改 in-process 檢查);evaluator 要用 `.venv` python。
> 下面保留原始分析作為紀錄。

> 大前提(A14):最終評分是 **LLM-as-judge 語意比對**,不是字串完全比對。
> 所以「格式/措辭」不重要,**「值對不對」才重要**。下面挑的都是會影響「值」或「檔案位置」的。

---

## 🔴 1. 輸出檔要寫到「輸入 testcase 的同一目錄」(A5.3)
- 官方:read 用 prompt 給的路徑;**write 到同一個 testcase 目錄**;輸入輸出都相對 working dir。
- 現在:`cada/agent/agent.py` 的 `h_write` 把 prompt 給的檔名(通常是裸檔名 `testNN_out.v`)寫到**當前目錄 cwd**。
- 要改:write 時若檔名沒帶目錄,就寫進載入設計的目錄 `state.design_dir`;若 prompt 帶了目錄就照用。
- 風險:若評測期待在 testcase 子目錄,**每個 write 都放錯 → 該題 0 分**(影響全部 40 題的最後一步)。
- 待確認:評測實際 cwd / write 路徑慣例;寫進 design_dir 是最保險的解。

## 🔴 2. cone / depth / 方程式要把 DFF.Q 當 primary input(A21.2、A30)
- 官方:fan-in cone **純組合**、終止在 DFF.Q(Q 視為 primary input),不要穿過暫存器。
- 現在:`cada/analysis/cones.py` 的 `effective_sinks(..., redirect_q=True)` 會把「輸出本身是 DFF.Q」的查詢**重導到該暫存器的 D-cone**(DESIGN.md §8.1 的猜測,方向相反)。被 cones / depth / deepest_output / boolean_equation 使用。
- 要改:把 `redirect_q` 改成 `False`(Q 當 PI,cone 到 Q 就停;暫存器輸出的組合 cone 為空/淺)。
- 影響:「which output has the deepest/largest fanin cone」「gates in cone of X」「depth of cone of X」「boolean equation of X」當 X 是暫存器輸出時,目前會算錯。

## 🟡 3. 「complete enumeration」要全列;大結果寫檔給路徑(A16、A21.3)
- 官方:complete enumeration = **literally 列出每一條路徑**;結果很大就**寫到檔案、回應給檔案路徑**。「list all <type> gates」(如 19682 個 NAND)同理。
- 現在:`cada/analysis/paths.py` `enumerate_paths` 超過 `ENUM_LIMIT=2000` 只回**數量**;`agent.py` 的 `h_enum_paths` / `h_list_type` 截斷到 ~200 並附「(N total)」。
- 要改:
  - 路徑列舉:小→全列;大→把全部路徑寫到檔(放 testcase 目錄)、回應給檔案路徑。
  - 大型 gate 清單:同樣寫檔給路徑,而非截斷。
- 風險:列舉/大清單類題目會被視為不完整 → 扣分。
- 注意:test14 的 n0[1]→n63[0] 約 289,366 條;要 streaming 寫檔、別全載入記憶體。

## 🟡 4. 「constant」是「功能上恆定」,不只結構綁定(A21.1)
- 官方:signal 只要「provably 0/1 for all inputs」就算 constant;「always 0」時 **DFF 初始狀態為 0、X 忽略**(組合)。
- 現在:`cada/transform/constprop.py` 的 `gates_with_const_input` 只看結構上綁 `1'b0`/`1'b1`;`cada/analysis/functional.py` 的 `output_always_constant` 用 cec 查組合 cone。
- 要改:「report gates with constant input」要納入**功能常數**(用 SAT/cec 證明某輸入恆 0/1),不只結構常數。
- 風險:漏報功能常數 → report/simplify/count 那串答案偏低。公開測資結構常數=0(我們報 0),但功能常數可能存在。
- 成本:較高(要對訊號做 SAT/const 檢查)。

## 🟢 5. fanout 計數要含 primary-output 連線(A29)
- 官方:fanout load 含 gate input、DFF 的 D/CK/RN/SN、**以及 primary output 連線**。
- 現在:`cada/analysis/connectivity.py` 的 `fanout_count` 用 `nl.loads(net)` 算 gate/dff pin(已含 DFF 各腳 ✓),但「net 本身是 PO」這種 PO 連線**沒當成 +1 load**。
- 要改:若 net 是 primary output,fanout 多算一個 load。
- 風險:邊界少算(影響 fanout 查詢與 max-fanout 界限的精確值)。

---

## ✅ 已經對上、不用改(備忘)
- 只做**組合等價**、DFF 當邊界(A30)→ 我們的 register-cut cec 正是如此。
- fan-in 在 DFF.Q 停(A21.2 前半)→ 圖把 Q 當 source ✓(但見第 2 點的 Q-輸出重導)。
- fanout 含 DFF D/CK/RN/SN(A29)→ `loads()` 已含 ✓。
- 單執行緒、一次一個 request、60s/300s(A18/A21.5-6)✓。
- 具名 instance、只有 built-in primitives、可能有 floating port(A1/A2/A5.6)→ parser 已處理 ✓。
- depth 不為空 net 加層(A33)✓。
- test01 文字不同但功能等價即可(A25)✓。
- DFF `.SN`=active-low set、`.RN`=active-low reset(A12)✓。

## 📦 部署備忘(非程式碼)
- **不收 Docker、直接在 RedHat 8 VM 跑(glibc 2.28)**(A39/A40):PyInstaller binary 要在目標機(或 glibc ≤ 2.28)build;本機有 uv 的話,**uv wrapper 最穩**。
- 最終評測有 **hidden 測資 + 不同措辭**,但任務類別不變(無 timing/placement/算術/FSM)(A6.2/A31)→ regex 會漏接,**LLM fallback 重要**(評測會換成官方 key)。
- 可隨附第三方 binary(static ABC)、無大小限制(A15)。
- 沒有官方答案 / 沒有官方 cost estimator(A26/A35/A19)→ 本機只能靠我們的 evaluator 做硬性檢查 + golden 回歸。
- cost 定義寫在各 testcase 的 prompt(更新版)(A8/A28/A34)。
