# Chatbot with TensorFlow

Chatbot đơn giản sử dụng **Python**, **TensorFlow**, và **NLTK**.

---

## 1. Cài đặt

```bash
# Tạo môi trường ảo
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
# venv\Scripts\activate    # Windows

# Cài thư viện
pip install -r requirements.txt

# Tải dữ liệu NLTK
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Huấn luyện chatbot
python train.py

# Bắt đầu chat
python chat.py

https://medium.com/%40abdulelahalwainany/build-your-own-chatbot-with-tensorflow-a-step-by-step-guide-afc374c40958
