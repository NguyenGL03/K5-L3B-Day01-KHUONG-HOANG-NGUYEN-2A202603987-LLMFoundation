# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603987
- **Họ và tên:** Khương Hoàng Nguyên
- **SEED** 2196

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

| Cách tiếp cận | F1 test | F1 SUB 40 | p95 ms/tin | USD / 1k tin |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.919 | 0.889 | 0.025 | 0 |
| 2. BoW + MLP | 0.939 | 0.974 | 0.128 | 0 |
| 3a. MiniLM + kNN | 0.884 | 0.927 | 23.5 | 0 |
| 3b. MiniLM + LogReg | 0.939 | 0.947 | 11.9 | 0 |
| 3c. MiniLM + MLP head | 0.941 | 0.974 | 11.7 | 0 |
| 4a. LLM nhỏ zero-shot (openai/gpt-6-luna qua Ckey) | — | 0.974 | 44 106 | 0.008 |
| 4b. LLM lớn zero-shot (openai/gpt-6.1-terra qua Ckey) | — | 0.950 | 35 149 | 0.075 |
| 4c. LLM nhỏ few-shot | — | 0.974 | 20 181 | 0.017 |

LLM chỉ chạy trên SUB (40 tin). Cột USD là giá niêm yết của OpenAI, không phải số tiền Ckey trừ thật. Độ trễ LLM tính cả thời gian qua gateway.

## 3. Dự đoán và thực tế

1. llm_models: em đoán model lớn hơn model nhỏ khoảng 0.3 F1. Thực tế model nhỏ đạt 0.974, model lớn 0.950, tức là hoà (chênh đúng 1 tin), trong khi model lớn đắt gấp khoảng 10 lần. Lệch vì phân loại spam là bài toán dễ, model nhỏ đã đủ nên model to hơn không giúp thêm.
2. hard_eval: em đoán LLM còn 80–90%, BoW còn 50–60% trên 50 case khó. Thực tế LLM đạt 98–100%, BoW chỉ đúng 54% (F1 0.378). Em đánh giá thấp khả năng đọc ý định của LLM và đánh giá quá cao BoW, vì điểm test thường của BoW (0.92) che mất việc nó chỉ học thuộc từ khoá.
3. ml_f1: em đoán F1 khoảng 0.7 và recall cao hơn precision. Thực tế F1 = 0.919, precision = 1.000, recall = 0.849, tức là ngược lại. Lý do là dữ liệu lệch (chỉ khoảng 13% spam) và với ngưỡng 0.5 mô hình chỉ chặn khi rất chắc, nên gần như không chặn nhầm nhưng bỏ lọt spam.

## 4. Phân tích 50 case khó

Model tụt nhiều nhất là BoW + LogReg: F1 trên SUB 0.889, trên 50 case khó chỉ còn 0.378. Nó sai nặng nhất ở các nhóm Thao túng / lừa đảo (đúng 12%), Né từ khoá (17%) và Spam giọng tự nhiên (20%).

- Case #29 (né từ khoá): "C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd…" được chấm p = 0.07, tức đoán ham. BoW chỉ đếm token nên "c0ngr4tulat10ns", "w0n" không có trong vocab và bị bỏ qua, các từ spam mạnh như "won", "claim" cũng mất theo. Embedding (83%) và LLM (100%) vẫn nhận ra.
- Case #41 (tin giao dịch hợp lệ, bị chặn nhầm): "Your OTP for login is 482913… call your bank." được chấm p = 0.58, tức đoán spam. Hai từ "call" và "your" nằm trong nhóm từ đẩy về spam mạnh nhất của mô hình, nên một tin OTP thật cũng vượt ngưỡng 0.5. Đây đúng là kiểu lỗi ngân hàng sợ nhất (xem mục 5).

Ngược lại, embedding yếu nhất ở nhóm Nói về spam (25–50%), vì vector nghĩa đo "nói về chủ đề gì" chứ không đo "ý định là gì". LLM dùng prompt phòng thủ chỉ sai 1 case: tin rủ đi xem nhạc bằng "free tickets" của bạn bè bị chặn nhầm, vì prompt làm model nghi ngờ hơn.

## 5. Thử thách cá nhân

Đề #1: đổi ngưỡng quyết định của mô hình Phần 1 sang 0.2 và 0.8. Em dùng lại w, b đã train, tính predict_proba trên tập test rồi so với từng ngưỡng.

| Ngưỡng | Precision | Recall | F1 | Ham bị chặn nhầm | Spam lọt |

| 0.2   | 0.896     | 0.925   | 0.910 | 10            | 7         |
| 0.5   | 1.000     | 0.849   | 0.919 | 0             | 14        |
| 0.8   | 1.000     | 0.667   | 0.800 | 0             | 31        |

Hạ ngưỡng thì recall tăng nhưng chặn nhầm cả tin bình thường như "Shall call now dear having food" (p = 0.28). Nâng ngưỡng thì precision giữ ở 1.0 nhưng spam lọt nhiều hơn. Ngân hàng sợ chặn nhầm tin OTP nên chọn ngưỡng 0.8 để ưu tiên precision. Case #41 ở mục 4 cho thấy một tin OTP thật có p = 0.58: bị chặn ở ngưỡng 0.5 nhưng lọt qua được ở 0.8. Nên kết hợp thêm whitelist sender ID của ngân hàng.

## 6. Phương án cho sản phẩm thật

Em chọn router 2 tầng, lấy 2. BoW + MLP làm lớp chính.

- Tầng 1: BoW + MLP có F1 test 0.939, p95 chỉ 0.13 ms, xử lý khoảng 84.000 tin/s nên dư sức cho mức trung bình 116 tin/s. Trước đó có whitelist sender ID (OTP, ngân hàng) và bước chuẩn hoá chống né từ khoá.
- Tầng 2: chỉ những tin lưỡng lự (p trong khoảng 0.3–0.7, vài %) mới gửi sang LLM nhỏ kèm prompt phòng thủ, xử lý bất đồng bộ để không phá SLA.

Không dùng LLM làm lớp chính vì p95 lên tới 20–44 s, chi phí khoảng $80/ngày (khoảng $2.400/tháng) với gpt-6-luna, hoặc khoảng $22.500/tháng với gpt-6.1-terra.

Đánh đổi chấp nhận:
- Tầng 1 vẫn yếu với spam giọng tự nhiên và lừa đảo (BoW chỉ đúng 54% case khó); tầng 2 chỉ cứu được những tin rơi vào vùng lưỡng lự.
- Phải thu nhãn từ LLM để train lại tầng 1 định kỳ.
- 3c (MiniLM + MLP head) cũng đạt SLA (p95 11.7 ms) nhưng cần thêm worker hoặc GPU, nên chỉ dùng khi có phần cứng.

## 7. AI log và điều đã học

AI log: (tự điền)

Điều em hiểu ra hôm nay:
- Điểm cao trên tập test cùng phân phối không chứng minh model hiểu bài toán: BoW đạt 0.92 nhưng tụt còn 0.38 khi gặp case khó.
- Model to hơn không mặc nhiên tốt hơn: model nhỏ ngang model lớn mà rẻ hơn 10 lần.
- Chọn giải pháp thật phải cân cùng lúc chất lượng, độ trễ, chi phí, và chọn ngưỡng theo loại lỗi nào đắt hơn, chứ không chỉ nhìn F1.
