# NCHU-AIoT

歡迎來到中興大學人工智慧物聯網 (AIoT) 課程作業儲存庫。本專案紀錄了從基礎網頁開發、傳統機器學習到進階深度學習與電腦視覺的實作歷程。

## 🚀 專案概覽

本課程涵蓋了 AI 開發的全生命週期，包含數據預處理、模型訓練、效能評估以及最後的應用部署。

### 技術棧
*   **語言**: Python (Jupyter Notebook)
*   **網頁框架**: Flask
*   **機器學習**: Scikit-learn, AutoML (PyCaret)
*   **深度學習**: PyTorch, TensorFlow, Keras, PyTorch Lightning
*   **電腦視覺**: OpenCV, Transfer Learning (VGG16)
*   **工具**: TensorBoard, Seaborn, Matplotlib, Pandas, Numpy

---

## 📂 作業說明 (Homeworks)

### [hw1: Flask 應用與數據回歸](./hw1)
*   **Flask Web App**: 實作個人介紹頁面、BMI 計算器與首頁分流。
*   **Regression Analysis**: 
    *   `50_Startups`: 預測創業公司利潤。
    *   `California House Price`: 房價回歸預測。

### [hw2: Titanic 生存預測](./hw2)
*   探索機器學習經典專案 Titanic。
*   **實作重點**: 特徵工程 (Feature Engineering)、填補缺失值、Label Encoding 與邏輯回歸 (Logistic Regression)。
*   **成果**: 生成 `submission.csv` 以供 Kaggle 競賽提交。

### [hw3: SVM 與邏輯回歸性能對比](./hw3)
*   **模型比較**: 實作 Logistic Regression 與 Support Vector Machine (Linear & RBF Kernel)。
*   **可視化**: 繪製決策邊界 (Decision Boundaries) 以觀察不同模型與 Kernel 在二分類問題上的表現。

### [hw4: AutoML 自動化機器學習](./hw4)
*   使用 **PyCaret** 框架進行自動化模型篩選與訓練。
*   透過 `compare_models()` 快速找出針對特定數據集表現最佳的機器學習演算法。

### [hw5: 深度學習框架實作](./hw5)
*   針對 Iris 數據集，分別使用三種主流框架實作多分類模型：
    1.  **PyTorch**: 底層張量運算與自定義訓練循環。
    2.  **TensorFlow / Keras**: 高階 API 快速開發。
    3.  **PyTorch Lightning**: 結構化與模組化的深度學習實作。
*   **監控**: 整合 **TensorBoard** 進行訓練損失與準確度的即時動態追蹤。

### [hw6: 電腦視覺與遷移學習](./hw6)
*   **專案**: 口罩偵測系統 (Face Mask Detection)。
*   **技術**: 使用 **VGG16** 預訓練模型進行 **Transfer Learning**，調整最後的 Fully Connected 層以適應二分類任務。
*   **功能**: 自動識別影像中的人物是否正確佩戴口罩。

---

## 🛠️ 安裝與安裝環境

1.  克隆此儲存庫:
    ```bash
    git clone https://github.com/danny/NCHU-AIoT.git
    ```
2.  安裝必要依賴:
    ```bash
    pip install -r requirements.txt
    ```
    *(註：請確保已安裝相關深度學習框架如 torch 或 tensorflow)*

3.  啟動 Jupyter Notebook:
    ```bash
    jupyter notebook
    ```

---

## 📈 學習進度快照

| 單元 | 類別 | 重點技術 | 狀態 |
| :--- | :--- | :--- | :--- |
| hw1 | 基礎 | Flask / Regression | ✅ |
| hw2 | ML | Feature Engineering | ✅ |
| hw3 | ML | SVM Kernels | ✅ |
| hw4 | AutoML | PyCaret | ✅ |
| hw5 | DL | Multi-Frameworks / TensorBoard | ✅ |
| hw6 | CV | Transfer Learning / VGG16 | ✅ |

---
*本專案由 [您的名字/ID] 維護，僅供學習參考使用。*
