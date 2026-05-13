# MAAL Reproduction - Active Legibility in Multiagent Reinforcement Learning

**Dự án tái hiện kết quả bài báo "Active legibility in multiagent reinforcement learning" (2025)**

---

## Phân công công việc (3 thành viên)

| Thành viên | Domain                              | File chính                          | Episodes   | Beta values     | Thời gian ước tính |
|------------|-------------------------------------|-------------------------------------|------------|-----------------|--------------------|
| **Duy**    | Lead-Follow Maze (LFM)              | `src/lfm_maal.py`                   | 100.000    | 0.0, 0.01       | 12–20 phút        |
| **Huy**    | Simple Navigation 3 agents          | `src/simple_nav_3agents.py`         | 15.000     | 0.0, 0.01       | 15–25 phút        |
| **Nam**    | Simple Navigation 6 agents 2 goals  | `src/simple_nav_6agents.py`         | 12.000     | 0.0, 0.01       | 20–30 phút        |
| **xxx**    | SMACLite (1c1s_vs_1sc)              | `src/smaclite_maal.py`              | 5.000      | 0.0, 0.05       | 10–15 phút        |

---

## Kết quả mong đợi (đã giảm episodes)

| Domain                        | Beta   | Episodes | Reward mong đợi          | PCR / Win Rate |
|-------------------------------|--------|----------|--------------------------|----------------|
| **Lead-Follow Maze**          | 0.01   | 100k     | **0.82 – 0.87**          | ~0.95+         |
| **Simple Nav 3 agents**       | 0.01   | 15k      | **-1.55 – -1.65**        | ~0.85+         |
| **Simple Nav 6 agents 2 goals**| 0.01  | 12k      | **-2.05 – -2.15**        | ~0.80+         |
| **SMACLite**                  | 0.05   | 5k       | 19.7 – 20.0              | ~0.98+         |

**Ghi chú:** Kết quả trên là giá trị mong đợi sau khi giảm episodes để tiết kiệm thời gian. Vẫn giữ được xu hướng rõ ràng: MAAL (β > 0) tốt hơn baseline (β = 0).

---

## Hướng dẫn chạy chi tiết



## Hướng dẫn chung (tất cả thành viên)

### 1. Clone repo
```powershell
git clone <link-repo-của-bạn>
cd Maal-Reproduction


2. Tạo và kích hoạt Virtual Environment (Python 3.11)
PowerShellpython -m venv env311
.\env311\Scripts\Activate

3. Cài thư viện chung
PowerShellpip install --upgrade pip
pip install -r requirements.txt

#có thể tải thêm phần bên dưới nếu lỗi chạy simple_nav hoặc tự fix 
pip install gymnasium numpy pandas matplotlib tqdm seaborn torch torchvision torchaudio

Hướng dẫn chi tiết theo từng thành viên
Thành viên 1 – Lead-Follow Maze (LFM)
PowerShellpython src/lfm_maal.py

Episodes: 100.000
Chạy 2 lần (β = 0.0 và β = 0.01)
Kết quả mong đợi:
results/lfm/results_beta0.0.csv và results_beta0.01.csv
plots/lfm/plot_beta0.0.png và plot_beta0.01.png
Reward ≈ 0.82 – 0.87 (β=0.01)



Thành viên 2 và 3 – Simple Navigation
Đã cài xong pettingzoo[mpe] trong venv.
PowerShell
python src/simple_nav_3agents.py
python src/simple_nav_6agents.py

3 agents
Episodes: 15.000
Kết quả mong đợi:
results/simple_nav_3agents/results_beta0.0.csv
results/simple_nav_3agents/results_beta0.01.csv
Reward ≈ -1.55 ~ -1.65 (β=0.01)






Kết quả cuối cùng sau khi chạy
Mỗi thành viên sẽ có thư mục:

results/[domain]/ → chứa file .csv
plots/[domain]/ → chứa file .png (3 biểu đồ: Reward, PCR/PTR/Win Rate)

Sau khi chạy xong, mỗi người nén thư mục results + plots của domain mình và gửi cho người tổng hợp slide.



































DomainβEpisodesReward (dự kiến)PCR / Win RateLead-Follow Maze0.01100k0.82 – 0.87~0.95+Simple Nav 3 agents0.0115k-1.55 – -1.65~0.85+SMACLite0.055k~19.7 – 20.0~0.98+
Ghi chú quan trọng:
Chúng ta giảm episodes để tiết kiệm thời gian nhưng vẫn giữ được xu hướng MAAL tốt hơn baseline.