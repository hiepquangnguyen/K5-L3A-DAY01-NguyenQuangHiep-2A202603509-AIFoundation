# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603509
- **Họ và tên:** Nguyễn Quang Hiệp
- **SEED:** 4857

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output (15/15 checks pass) | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra (`seed = 4857`) | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước (`0.87`, `0.5`, `0.73`, `0.6`, `10.50`) | ☑ |

## 2. Kết quả chính

Bảng tổng hợp 8 phương pháp trên tập test đầy đủ ($N = 720$ tin: 624 ham, 96 spam) và tập con `SUB` ($N = 40$ tin: 20 ham, 20 spam; riêng các hàng `4a`/`4b`/`4c` đánh giá trực tiếp trên `SUB` 40 tin để tiết kiệm chi phí API):

| Phương pháp | Accuracy | Precision | Recall | F1 (test) | F1 (SUB 40) | Train (s) | Tin/giây | p50 (ms/tin) | p95 (ms/tin) | USD / 1k tin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1. BoW + LogReg (tự code)` | 0.972 | 1.000 | 0.771 | 0.871 | 0.919 | 8.610 | 17,284.4 | 0.053 | 0.710 | 0.000 |
| `2. BoW + MLP (deep learning)` | 0.982 | 1.000 | 0.854 | 0.921 | 1.000 | 11.715 | 34,247.1 | 0.390 | 0.696 | 0.000 |
| `3a. Encoder (MiniLM) + kNN` | 0.975 | 0.867 | 0.948 | 0.905 | 1.000 | 2.025 | 87.5 | 39.990 | 67.607 | 0.000 |
| `3b. Encoder (MiniLM) + LogReg` | 0.991 | 0.989 | 0.938 | **0.963** | 1.000 | 3.601 | 1,227.3 | 6.786 | 10.876 | 0.000 |
| `3c. Encoder (MiniLM) + MLP head` | 0.991 | 0.989 | 0.938 | **0.963** | 1.000 | 3.165 | 1,661.4 | 6.781 | 10.261 | 0.000 |
| `4a. LLM nhỏ zero-shot` (`gpt-oss-20b`) | 0.975 | 0.952 | 1.000 | 0.976 | 0.976 | 0.000 | 0.8 | 860.573 | 2,659.919 | 0.004 |
| `4b. LLM lớn zero-shot` (`gpt-oss-120b`) | 0.950 | 0.909 | 1.000 | 0.952 | 0.952 | 0.000 | 1.0 | 515.520 | 2,565.100 | 0.008 |
| `4c. LLM nhỏ few-shot` (`gpt-oss-20b`) | 0.950 | 0.909 | 1.000 | 0.952 | 0.952 | 0.000 | 0.6 | 1,786.728 | 3,030.614 | 0.016 |

- **Trên đường Pareto (tập SUB 40):** `2. BoW + MLP (deep learning)` (đạt lỗi $1 - \text{F1}_{\text{SUB}} = 0\%$ và `p95 = 0.696 ms`, lấn át cả các mô hình khác trên tập `SUB`). Nếu xét trên tập test đầy đủ ($N = 720$), đường biên Pareto đi qua `2. BoW + MLP` (`F1 = 0.921`, `p95 = 0.696 ms`) và `3c. Encoder + MLP head` (`F1 = 0.963`, `p95 = 10.261 ms`).
- **Đạt SLA (`p95 < 50 ms/tin`):** `1. BoW + LogReg` (`0.710 ms`), `2. BoW + MLP` (`0.696 ms`), `3b. Encoder + LogReg` (`10.876 ms`), và `3c. Encoder + MLP head` (`10.261 ms`). Riêng `3a. Encoder + kNN` (`67.607 ms`) và toàn bộ nhóm LLM (`2,565–3,031 ms`) vượt ngưỡng SLA.

## 3. Dự đoán và thực tế

Ba dự đoán `predict(...)` lệch nhiều nhất hoặc bất ngờ nhất so với kết quả thực nghiệm:

1. **`P2_mlp_vs_bow` (Phần 2 — BoW + MLP so với BoW + LogReg):**
   - **Dự đoán:** `"Gần như bằng nhau (chênh < 0.01) vì dữ liệu nhỏ"` (nghĩ rằng tập train 2.880 tin nhắn SMS ngắn là quá nhỏ nên thêm hidden layer trên BoW sẽ dễ overfit và không tạo ra khác biệt lớn).
   - **Thực tế:** `2. BoW + MLP` đạt `F1 (test) = 0.921` (`Recall = 0.854`, `F1 SUB = 1.000`), tăng vọt **+0.050 điểm F1** (+5.0 điểm %) so với `1. BoW + LogReg` (`F1 (test) = 0.871`, `Recall = 0.771`, `F1 SUB = 0.919`).
   - **Vì sao lệch:** BoW + LogReg là bộ phân loại tuyến tính nên chỉ cộng độc lập trọng số của từng từ đơn lẻ; khi tin nhắn spam chứa ít từ khoá mạnh nhưng xuất hiện tổ hợp từ gợi ý (*"reply" + "text" + số điện thoại*), tổng tuyến tính chưa vượt ngưỡng $z = 0$ nên bị bỏ sót (`Recall` chỉ đạt `0.771`). Lớp ẩn 64 chiều kèm kích hoạt phi tuyến `ReLU` của MLP học được các tổ hợp đồng xuất hiện giữa các token, giúp kéo `Recall` lên `0.854` trong khi vẫn giữ `Precision = 1.000` (nhờ dừng ở epoch hợp lý trước khi overfit nặng).

2. **`P3_bow_vs_emb` (Phần 3 — Thí nghiệm 3.B trên 6 câu tricky tự chọn):**
   - **Dự đoán:** `"Embedding thắng rõ (≥ 2 câu hơn) vì hiểu đồng nghĩa"` (kỳ vọng `MiniLM + kNN` sẽ vượt xa `BoW + LogReg` khi gặp các câu biến tấu không dùng từ khoá kinh điển).
   - **Thực tế:** Cả hai mô hình **hoà nhau 5/6 câu đúng** và **cùng đoán sai câu thứ 6** (`"Hi Sarah, your package delivery failed; confirm your details at the link below"` — nhãn thật là `spam (1)`, nhưng cả hai đều đoán `ham (0)`).
   - **Vì sao lệch:** 5 câu đầu trong bộ 6 câu nhỏ vẫn chứa một số token đủ phân biệt trong từ điển 2.000 từ (`won`, `cash`, `reward`, `meeting`, `lunch`) nên `BoW + LogReg` vẫn xử lý tốt. Ngược lại, ở câu thứ 6 (phishing giả mạo thông báo giao hàng), văn phong và từ vựng (`Hi Sarah`, `package delivery`, `confirm details`) giống hệt một tin nhắn giao dịch bình thường và trong tập huấn luyện SMS năm 2011 hầu như không có dạng lừa đảo bưu kiện hiện đại này, khiến cả vector ngữ nghĩa của `MiniLM` lẫn túi từ `BoW` đều đặt nó vào vùng `ham`.

3. **`P4_big_vs_small` & `P4_fewshot_delta` (Phần 4 — LLM lớn và Few-shot so với LLM nhỏ Zero-shot):**
   - **Dự đoán:** Dự đoán model lớn `4b` (`gpt-oss-120b`) sẽ tương đương hoặc nhỉnh hơn `4a` (`gpt-oss-20b`), và thêm 4 ví dụ mẫu trong `4c` (few-shot) sẽ giúp F1 tăng thêm `+1% đến +3%`.
   - **Thực tế:** `4a. LLM nhỏ zero-shot` đạt `F1 = 0.976` (chỉ sai 1/40 tin trên `SUB`), cao hơn cả `4b. LLM lớn zero-shot` (`F1 = 0.952`, sai 2/40 tin) và `4c. LLM nhỏ few-shot` (`F1 = 0.952`, sai 2/40 tin), trong khi `4c` tốn chi phí cao gấp **4 lần** (`0.016 USD/1k` so với `0.004 USD/1k`) và độ trễ `p95` chậm hơn (`3,030.6 ms` so với `2,659.9 ms`).
   - **Vì sao lệch:** Bài toán phân loại SMS spam trên tập chuẩn vốn đã rất rõ ràng với một LLM đã qua instruction-tuning. Khi đưa thêm 4 ví dụ mẫu (few-shot) hoặc dùng model suy luận lớn hơn (`120B`), mô hình có xu hướng suy diễn quá mức (over-thinking) và nhạy cảm hơn với các tin ham có giọng điệu quảng cáo/hấp dẫn, dẫn đến tăng báo nhầm (False Positive tăng từ 1 lên 2 tin, làm `Precision` giảm từ `0.952` xuống `0.909`) trong khi tốn thêm hàng trăm prompt token cho mỗi lần gọi.

## 4. Phân tích 50 case khó

### Tổng quan kết quả trên 50 case khó (`hard_eval_50.csv`)

| Phương pháp | F1 test thường (`SUB`) | F1 50 case khó | Accuracy 50 case khó | Độ sụt giảm F1 |
| --- | ---: | ---: | ---: | ---: |
| `1. BoW + LogReg` | 0.919 | **0.333** | 0.520 (52%) | **−0.586** |
| `2. BoW + MLP` | 1.000 | **0.462** | 0.580 (58%) | **−0.538** |
| `3a. Emb + kNN` | 1.000 | 0.750 | 0.720 (72%) | −0.250 |
| `3b. Emb + LogReg` | 1.000 | 0.667 | 0.660 (66%) | −0.333 |
| `3c. Emb + MLP head` | 1.000 | 0.625 | 0.640 (64%) | −0.375 |
| `4a. LLM nhỏ zero-shot` | 0.976 | **0.935** | **0.920 (92%)** | −0.041 |
| `4d. LLM nhỏ + prompt phòng thủ` | 0.976 | 0.918 | 0.900 (90%) | −0.058 |

**Độ chính xác theo từng nhóm trong 50 case khó:**

| Nhóm case khó (`category`) | `1. BoW+LR` | `2. BoW+MLP` | `3a. kNN` | `3b. Emb+LR` | `3c. Emb+MLP` | `4a. LLM` | `4d. LLM def` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Tưởng spam nhưng không phải (`ham_looks_spam`) | 100% | 90% | 70% | 80% | 80% | 70% | 70% |
| Thao túng / lừa đảo (`manipulation`) | **12%** | **38%** | 88% | **38%** | **38%** | 100% | 100% |
| Spam giọng tự nhiên (`natural_spam`) | **10%** | **10%** | 50% | **30%** | **30%** | 100% | 90% |
| Né từ khoá (`obfuscation`) | **17%** | **17%** | 83% | 100% | 67% | 100% | 100% |
| Prompt injection (`prompt_injection`) | 67% | 83% | 83% | 100% | 100% | 100% | 100% |
| Nói về spam (`talks_about_spam`) | 100% | 100% | **25%** | **50%** | 75% | 100% | 75% |
| Tin giao dịch hợp lệ (`transactional_ham`) | 83% | 100% | 100% | 83% | 83% | 83% | 100% |
| **TỔNG (50 case)** | **52%** | **58%** | **72%** | **66%** | **64%** | **92%** | **90%** |

*(Ghi chú: Trong notebook cũng đã viết thêm 5 case tự chế `MY_CASES` `#100–#104`; khi gộp thành 55 case, accuracy của `1. BoW+LR` và `2. BoW+MLP` tụt thêm xuống còn `47%` và `53%`, trong khi `4a. LLM` giữ vững `91%`).*

- **Model tụt nhiều nhất và ở nhóm nào:**
  - Hai mô hình dựa trên túi từ (`1. BoW + LogReg` và `2. BoW + MLP`) tụt thê thảm nhất: F1 rơi từ `0.919–1.000` xuống chỉ còn **`0.333`** và **`0.462`**. Nhóm tụt nặng nhất của BoW là **`natural_spam` (10%)**, **`manipulation` (12% – 38%)**, và **`obfuscation` (17%)**.
  - Các mô hình Embedding (`3a`, `3b`, `3c`) giữ được nhóm `obfuscation` tốt hơn (`67%–100%`) nhờ tokenizer subword (`WordPiece`), nhưng lại sụp đổ ở nhóm **`talks_about_spam` (`25%–50%` với `3a`/`3b`)** và **`natural_spam` (`30%–50%`)**.

- **Phân tích 2 case cụ thể và cơ chế khiến model sai:**
  1. **Case #4 (`obfuscation` — nhãn thật: `spam = 1`):**
     - *Nội dung:* `"F R E E entry into our w.e.e.k.l.y draw! R3ply Y3S now to cl4im your g1ft."`
     - *Kết quả dự đoán:* `1. BoW + LogReg` đoán `0 (ham)`, `2. BoW + MLP` đoán `0 (ham)`, trong khi `3a/3b/3c (MiniLM)` và `4a/4d (LLM)` đều đoán đúng `1 (spam)`.
     - *Cơ chế gây lỗi:* Hàm `tokenize()` của BoW dùng biểu thức chính quy `[a-z0-9']+` tách từ theo khoảng trắng và dấu câu. Khi kẻ phát tán chèn khoảng trắng vào `"F R E E"` $\to$ `["f", "r", "e", "e"]`, chèn dấu chấm vào `"w.e.e.k.l.y"` $\to$ `["w", "e", "e", "k", "l", "y"]`, và thay chữ bằng số `"R3ply"`, `"Y3S"`, `"cl4im"`, `"g1ft"`, toàn bộ các token mang trọng số spam cao nhất (`free`, `reply`, `claim`, `gift`, `weekly`) đều biến mất khỏi vector đặc trưng 2.000 chiều (trở thành từ ngoài từ điển OOV hoặc chữ cái đơn vô nghĩa). Chỉ còn lại các từ nối phổ biến (`into`, `our`, `now`, `to`, `your`) mang trọng số âm nghiêng về `ham`, nên tổng $z = w^\top x + b < 0 \Rightarrow \sigma(z) < 0.5$ (Bỏ sót / False Negative).
  2. **Case #32 (`talks_about_spam` — nhãn thật: `ham = 0`):**
     - *Nội dung:* `"Don't ever reply to texts saying you won a free prize or cash reward — that's how my uncle lost $500."`
     - *Kết quả dự đoán:* `3a. Emb + kNN` và `3b. Emb + LogReg` đều đoán sai thành `1 (spam)`, trong khi `4a. LLM` đoán đúng `0 (ham)`.
     - *Cơ chế gây lỗi:* Câu này có **ý định (intent)** là lời khuyên cảnh giác của người thân, nhưng lại chứa dày đặc cụm từ thuộc chủ đề trúng thưởng (`reply to texts`, `won a free prize`, `cash reward`, `$500`). Mô hình Sentence-Transformer (`all-MiniLM-L6-v2`) nén toàn bộ câu thành một vector 384 chiều phản ánh *chủ đề ngữ nghĩa tổng thể* (semantic topic similarity). Do phần lớn diện tích câu nói về chủ đề giải thưởng/tiền mặt, vector biểu diễn của câu bị kéo vào sát cụm các tin nhắn spam trúng thưởng trong không gian 384 chiều, làm $k$ hàng xóm gần nhất (kNN) và siêu phẳng phân chia của `Emb + LogReg` phân loại nhầm thành `spam` (Báo nhầm / False Positive). Chỉ có LLM với cơ chế tự chú ý (self-attention) đọc hiểu cấu trúc phủ định và mệnh đề nhân quả (`"Don't ever reply to... — that's how my uncle lost $500"`) mới phân biệt được đây là tin nhắn *bàn về spam* chứ không phải *là spam*.

## 5. Thử thách cá nhân

- **Đề bài theo `SEED = 4857` (`4857 % 6 = 3` $\to$ Đề #4):**
  > *Embedding · Chạy `knn_predict` với $k = 1, 5, 25$. Báo F1. Vì sao $k$ quá nhỏ hoặc quá lớn đều có hại?*
- **Cách làm:**
  Sử dụng ma trận embedding đã chuẩn hoá L2 trên tập huấn luyện `E_tr` ($2880 \times 384$) và tập kiểm tra `E_te` ($720 \times 384$) của mô hình `sentence-transformers/all-MiniLM-L6-v2`, gọi hàm `knn_predict(E_tr, y_tr, E_te, k=k)` lần lượt với $k \in \{1, 5, 25\}$ và tính `precision`, `recall`, `f1` bằng hàm `metrics(y_te, preds)`.
- **Kết quả thực nghiệm:**

| Tham số $k$ | Precision | Recall | F1 (test) | Nhận xét |
| --- | ---: | ---: | ---: | --- |
| $k = 1$ | **0.882** | 0.938 | **0.909** | F1 cao nhất trong 3 mức, nhưng phụ thuộc hoàn toàn vào 1 điểm gần nhất |
| $k = 5$ | 0.867 | **0.948** | 0.905 | **Recall cao nhất** (bắt được nhiều spam nhất: 91/96 tin spam) |
| $k = 25$ | 0.873 | 0.927 | 0.899 | Cả Recall lẫn F1 đều sụt giảm rõ rệt |

- **Giải thích vì sao $k$ quá nhỏ hoặc quá lớn đều có hại (đánh đổi Bias – Variance):**
  - **Khi $k$ quá nhỏ ($k = 1$ — Phương sai cao / High Variance):** Quyết định phân loại cho một tin nhắn mới phụ thuộc 100% vào duy nhất 1 hàng xóm gần nhất trong tập train. Ranh giới quyết định trở nên răng cưa và cực kỳ nhạy cảm với nhiễu: chỉ cần 1 tin nhắn `ham` ngoại lai (ví dụ tin nhắn hợp lệ có chứa từ `"free"` hoặc `"call"`) hoặc 1 mẫu bị gán nhãn sai nằm sát vùng biên, mọi tin nhắn kiểm tra rơi gần điểm đó đều bị đoán sai theo.
  - **Khi $k$ quá lớn ($k = 25$ — Độ chệch cao / High Bias & Lệch lớp):** Vùng lân cận mở quá rộng khiến mô hình lấy cả những câu không thực sự gần nghĩa vào bỏ phiếu, làm mờ ranh giới cục bộ giữa các cụm nhỏ. Nghiêm trọng hơn, do tập dữ liệu SMS bị **mất cân bằng lớp mạnh (~86.7% là `ham`, chỉ ~13.3% là `spam`)**, một cụm tin nhắn spam đặc thù (ví dụ nhóm lừa đảo bưu kiện hoặc quảng cáo ít gặp chỉ có ~8–10 mẫu trong tập train) khi phải lấy đủ $k = 25$ hàng xóm sẽ bị tràn sang vùng `ham` đông đảo xung quanh. Khi đó số phiếu `ham` áp đảo số phiếu `spam` ($> 12/25$ phiếu), khiến `Recall` tụt từ `0.948` ($k=5$) xuống `0.927` ($k=25$) và kéo `F1` giảm xuống `0.899`.

## 6. Phương án cho sản phẩm thật

**Bài toán:** Nhà mạng cần lọc **10 triệu SMS/ngày** (~116 tin/giây trung bình, giờ cao điểm có thể gấp 3–5 lần), yêu cầu SLA độ trễ **`p95 < 50 ms/tin`**.

1. **Phân tích và loại trừ trên từng phương án đơn lẻ:**
   - **Loại nhóm LLM đơn lẻ (`4a`, `4b`, `4c`):** Dù đạt chất lượng cao nhất trên cả tập thường (`F1 = 0.976`) lẫn 50 case khó (`F1 = 0.935`), độ trễ `p95 = 2,565 – 3,031 ms/tin` chậm hơn ngưỡng SLA `50 ms` tới **50–60 lần** và thông lượng chỉ đạt `0.6 – 1.0 tin/giây`. Về chi phí, với 10 triệu tin/ngày (300 triệu tin/tháng), ngay cả model rẻ nhất `4a` (`0.00408 USD/1k tin`) cũng tiêu tốn **~1,225 USD/tháng** (nếu dùng `4c` few-shot `0.01568 USD/1k tin` sẽ tốn **~4,705 USD/tháng**).
   - **Loại `3a. Encoder + kNN`:** Có `p95 = 67.607 ms > 50 ms` và khi dữ liệu thực tế tăng lên hàng triệu mẫu, chi phí tính khoảng cách tới toàn bộ tập mẫu sẽ phình to tuyến tính $\mathcal{O}(N)$.
   - **Lựa chọn mô hình chính đáp ứng SLA (`p95 < 50 ms`):** Trong 4 mô hình đạt SLA (`1`, `2`, `3b`, `3c`), chọn **`3c. Encoder (MiniLM) + MLP head`** (hoặc **`3b. Encoder (MiniLM) + LogReg`**) làm mô hình chủ lực vì:
     - Đạt **`F1 (test) = 0.963`** (cao nhất trong tất cả các mô hình non-LLM), **`F1 (SUB 40) = 1.000`**, và đặc biệt **`Precision = 0.989`** (cực kỳ quan trọng với nhà mạng vì chặn nhầm tin nhắn thật của khách hàng nguy hiểm hơn để lọt quảng cáo).
     - Độ trễ **`p95 = 10.261 ms`** (`3c`) và **`10.876 ms`** (`3b`), thấp hơn gần **5 lần** so với trần SLA `50 ms`.
     - Thông lượng đạt **`1,227 – 1,661 tin/giây`** trên 1 GPU (dư sức gánh 116 tin/giây trung bình và tải đỉnh), chi phí gọi API bằng **`0.000 USD`**.

2. **Kiến trúc đề xuất cho sản phẩm thật — Hệ thống phân tầng (Cascaded Router 2 tầng):**
   Để khắc phục điểm yếu của `3b`/`3c` trên các tin nhắn lừa đảo tinh vi (`hard_eval_50` F1 đạt `0.625–0.667` so với `0.918–0.935` của LLM), triển khai kiến trúc lai:
   - **Tầng 1 (Fast Path — Online Inline, 100% lưu lượng):** Chạy tiền xử lý chuẩn hoá leetspeak/ký tự ẩn kết hợp **`3b. Encoder (MiniLM) + LogReg`** (hoặc `3c`) trả về xác suất $p = P(\text{spam} \mid x)$ trong `~7–10 ms`:
     - Nếu $p < 0.25 \to$ Cho qua ngay là **`ham`** (~85% lưu lượng).
     - Nếu $p > 0.80 \to$ Chặn/đưa vào mục Spam ngay (~12% lưu lượng).
   - **Tầng 2 (Slow / Gray-zone Path — Async Verification, ~3% lưu lượng lưỡng lự $0.25 \le p \le 0.80$):**
     - Chỉ đẩy ~3% tin nhắn nằm ở vùng biên (nơi tập trung các ca `natural_spam`, `manipulation`, `talks_about_spam`) sang **`4d. LLM nhỏ + prompt phòng thủ`** xử lý bất đồng bộ (hoặc gắn cờ cảnh báo trên thiết bị người nhận).
   - **Đánh đổi chấp nhận (Trade-offs):**
     - **Độ trễ:** 100% tin nhắn trên luồng chính đảm bảo `p95 ~ 10.3–10.9 ms < 50 ms`; ~3% tin nhắn vùng xám có kết quả hậu kiểm sau `~1–2.5 giây`.
     - **Chi phí:** Giảm **97%** chi phí LLM — chỉ tốn $3\% \times 1,225 \approx \mathbf{36.75\text{ USD/tháng}}$ thay vì `1,225 USD/tháng`, nhưng nâng độ bền trước các chiến dịch lừa đảo mới từ `66%` lên tiệm cận `~90%+`.
     - **Vận hành:** Chấp nhận chi phí duy trì máy chủ GPU nhỏ cho `MiniLM` và pipeline thu thập các mẫu vùng xám mà LLM bắt được để định kỳ huấn luyện lại (retrain) lớp phân loại nhẹ `LogReg`/`MLP head` (chỉ mất `~3.2–3.6 giây`).

## 7. AI log và điều đã học

### Bảng nhật ký sử dụng AI (AI Log)

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời / viết gì | Bạn đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| **Phần 1–3** (BoW, MLP & Embedding) | Hỏi cách viết gọn phép chuẩn hoá L2 theo hàng trong NumPy để tránh lỗi chia cho 0 và cách đọc biểu đồ `train_losses` vs `val_losses` để phát hiện thời điểm bắt đầu overfit. | Gợi ý dùng `np.linalg.norm(M, axis=1, keepdims=True) + 1e-12` trước khi nhân ma trận `M_norm @ q_norm`, và chỉ ra điểm uốn nơi `val_loss` ngừng giảm trong khi `train_loss` tiếp tục giảm. | Tự cài đặt các hàm `sigmoid`, `predict_proba`, `fit_logreg`, `MLP`, `top_k_similar`, `knn_predict`; chạy kiểm tra đạt 100% các `check` từ `1.1` đến `3.2` và đối chiếu đường cong loss tại epoch 5. |
| **Phần 4** (LLM API & Prompting) | Hỏi cấu trúc gọi OpenAI SDK tương thích với Groq endpoint (`provider_kwargs`) và thứ tự kiểm tra chuỗi trong hàm `parse_label` để không bắt nhầm `"not spam"` thành `"spam"`. | Gợi ý truyền `base_url="https://api.groq.com/openai/v1"`, kiểm tra `"not spam"` hoặc `"ham"` trước `"spam"` sau khi `.strip().lower()`, và bọc jsonl trong cặp thẻ `<messages>...</messages>`. | Cấu hình `GROQ_API_KEY`, viết hoàn chỉnh `call_openai`, `parse_label`, `classify_zero_shot`, `classify_few_shot`, `classify_batch`, `build_defended_prompt`; kiểm chứng pass toàn bộ `check('4.1')` đến `check('4.4')`. |
| **Phần 5** (Stress Test & Báo cáo) | Hỏi nguyên nhân lỗi `ValueError: too many values to unpack` khi thêm `MY_CASES` vào `hard_eval_50`, cách kiểm tra điều kiện lấn át Pareto (`dominated`), và rà soát cấu trúc báo cáo cá nhân. | Giải thích mỗi phần tử của `MY_CASES` chỉ gồm tuple 3 trường `(label, category, text)` (cột `id` được tự đánh số từ `100` khi tạo DataFrame) và tổng hợp lại các số liệu từ `lab1_report.json`. | Sửa lại `MY_CASES` đúng định dạng 3 trường, chạy thực nghiệm lại toàn bộ bảng 50 case chuẩn + 5 case tự thêm, đối chiếu khớp từng con số với `lab1_report.json`. |

### Điều đã học được hôm nay

Trước buổi lab hôm nay, em từng mặc định rằng mô hình càng lớn và hiện đại (như LLM hàng chục/trăm tỷ tham số kèm few-shot) thì lúc nào cũng vượt trội và nên dùng thay thế hoàn toàn các mô hình cổ điển. Qua việc tự tay code từ `BoW + LogReg` bằng NumPy đến `MLP`, `MiniLM Embedding` và `LLM`, em hiểu ra rằng **cách mô hình biểu diễn văn bản** (đếm token rời rạc $\to$ nén vector ngữ nghĩa $\to$ đọc hiểu ngữ cảnh bằng attention) quyết định trực tiếp **điểm mù** của nó: BoW cực nhanh (`< 1 ms`) nhưng mù trước biến dạng chữ (`F R E E`), Embedding bắt được từ đồng nghĩa và đạt `F1 = 0.963` ở độ trễ `~10 ms` nhưng lại nhầm giữa *"tin nhắn nói về spam"* và *"tin nhắn spam"*, còn LLM hiểu được ý định lừa đảo tinh vi (`92%` trên 50 case khó) nhưng chậm gấp ~250 lần (`p95 > 2.5 s`) và tốn tiền theo từng token. Vì vậy, bài toán kỹ thuật AI thực tế không phải là đi tìm một mô hình "mạnh nhất" duy nhất, mà là thiết kế **kiến trúc lai (Router)** để kết hợp tốc độ, chi phí `0 USD` của `Encoder + LogReg/MLP` cho 95–97% lưu lượng phổ thông với khả năng suy luận sâu của LLM cho 3–5% ca khó ở vùng biên.
