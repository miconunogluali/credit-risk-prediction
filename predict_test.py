import os
import joblib
import pandas as pd
import logging

# ==========================================
# 🛑 KURUMSAL LOGGING (GÜNLÜK) YAPILANDIRMASI
# ==========================================
log_filename = "app.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler() # Hem dosyaya yaz hem terminale bas
    ]
)
logger = logging.getLogger("CreditRiskApp")

def calculate_financial_metrics(score):
    """Findeks skoruna göre kredi derecesi ve faiz oranı belirleyen iş motoru."""
    if 1700 <= score <= 1900:
        return 'A', 11.0, 0
    elif 1500 <= score < 1700:
        return 'B', 13.5, 0
    elif 1300 <= score < 1499:
        return 'C', 16.0, 0
    elif 1100 <= score < 1299:
        return 'D', 19.5, 0
    elif 700 <= score < 1099:
        return 'E', 24.0, 0
    elif 1 <= score < 699:
        return 'F', 29.0, 1
    else:
        return 'G', 35.0, 1

# ==========================================
# 🛡️ GÜVENLİ VERİ GİRİŞ FONKSİYONLARI (ROBUST INPUT)
# ==========================================
def get_safe_int(prompt, min_val=0, max_val=2000):
    while True:
        try:
            val = int(input(prompt).strip())
            if min_val <= val <= max_val:
                return val
            print(f"❌ Lütfen {min_val} ile {max_val} arasında bir sayı giriniz.")
        except ValueError:
            logger.warning("Kullanıcı sayısal alana geçersiz karakter girdi.")
            print("❌ Hatalı Giriş! Lütfen sadece tam sayı giriniz.")

def get_safe_float(prompt, min_val=0.0):
    while True:
        try:
            val = float(input(prompt).strip())
            if val >= min_val:
                return val
            print(f"❌ Lütfen {min_val}'den büyük bir değer giriniz.")
        except ValueError:
            logger.warning("Kullanıcı ondalıklı sayısal alana geçersiz karakter girdi.")
            print("❌ Hatalı Giriş! Lütfen geçerli bir miktar giriniz.")

def get_safe_choice(prompt, valid_choices):
    while True:
        val = input(prompt).upper().strip()
        if val in valid_choices:
            return val
        print(f"❌ Geçersiz seçim! Sadece şunları girebilirsiniz: {', '.join(valid_choices)}")

def get_user_inputs():
    logger.info("Yeni bir kredi başvuru giriş süreci başladı.")
    print("\n--- 📝 Lütfen Kredi Başvuru Bilgilerinizi Giriniz ---")
    
    age = get_safe_int("Yaşınız (Örn: 28): ", 18, 100)
    income = get_safe_float("Yıllık Net Geliriniz ($): ")
    emp_length = get_safe_int("Mevcut İşinizdeki Çalışma Süreniz (Yıl): ", 0, 60)
    loan_amnt = get_safe_float("Talep Ettiğiniz Kredi Miktarı ($): ")
    cred_hist_length = get_safe_int("Kredi Geçmişi Süreniz (Yıl): ", 0, 50)
    score = get_safe_int("Findeks Kredi Notunuz (0 - 1900 arası): ", 0, 1900)

    grade_input, loan_int_rate, default_status = calculate_financial_metrics(score)
    loan_percent_income = loan_amnt / income

    home_choices = ['RENT', 'OWN', 'MORTGAGE', 'OTHER']
    home_input = get_safe_choice(f"\n🏠 Ev Sahipliği Durumu {home_choices}: ", home_choices)

    intent_choices = ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION']
    intent_input = get_safe_choice(f"\n🎯 Kredi Kullanım Amacı {intent_choices}: ", intent_choices)

    customer_dict = {
        'person_age': age, 'person_income': income, 'person_emp_length': emp_length,
        'loan_amnt': loan_amnt, 'loan_int_rate': loan_int_rate, 'loan_percent_income': loan_percent_income,
        'cb_person_cred_hist_length': cred_hist_length,
        'person_home_ownership_OTHER': 1 if home_input == 'OTHER' else 0,
        'person_home_ownership_OWN': 1 if home_input == 'OWN' else 0,
        'person_home_ownership_RENT': 1 if home_input == 'RENT' else 0,
        'loan_intent_EDUCATION': 1 if intent_input == 'EDUCATION' else 0,
        'loan_intent_HOMEIMPROVEMENT': 1 if intent_input == 'HOMEIMPROVEMENT' else 0,
        'loan_intent_MEDICAL': 1 if intent_input == 'MEDICAL' else 0,
        'loan_intent_PERSONAL': 1 if intent_input == 'PERSONAL' else 0,
        'loan_intent_VENTURE': 1 if intent_input == 'VENTURE' else 0,
        'loan_grade_B': 1 if grade_input == 'B' else 0,
        'loan_grade_C': 1 if grade_input == 'C' else 0,
        'loan_grade_D': 1 if grade_input == 'D' else 0,
        'loan_grade_E': 1 if grade_input == 'E' else 0,
        'loan_grade_F': 1 if grade_input == 'F' else 0,
        'loan_grade_G': 1 if grade_input == 'G' else 0,
        'cb_person_default_on_file_Y': default_status
    }
    
    logger.info(f"Kullanıcı girdileri başarıyla doğrulandı. Hesaplanan Derece: {grade_input}, Faiz: %{loan_int_rate}")
    return pd.DataFrame([customer_dict])

def interactive_prediction():
    logger.info("Uygulama başlatıldı.")
    model_path = os.path.join("models", "credit_risk_model.joblib")
    columns_path = os.path.join("models", "model_columns.joblib")
    
    if not os.path.exists(model_path) or not os.path.exists(columns_path):
        logger.error("Model veya sütun dosyaları diskte bulunamadı!")
        print("❌ [ERROR] Model dosyaları bulunamadı! Lütfen önce main.py çalıştırın.")
        return

    try:
        model = joblib.load(model_path)
        model_columns = joblib.load(columns_path)
        logger.info("Model ve sütun yapısı diskten başarıyla yüklendi.")
    except Exception as e:
        logger.critical(f"Model yüklenirken kritik hata oluştu: {str(e)}")
        return
    
    df_customer = get_user_inputs()
    df_customer = df_customer.reindex(columns=model_columns, fill_value=0)
    
    try:
        prediction = model.predict(df_customer)
        prediction_proba = model.predict_proba(df_customer)
        
        print("\n=========================================")
        print("       📊 ANLIK BAŞVURU SONUCU          ")
        print("=========================================")
        if prediction[0] == 0:
            print("✅ [TEBRİKLER - KREDİNİZ ONAYLANDI]")
            logger.info("Sonuç: KREDİ ONAYLANDI.")
        else:
            print("❌ [MAALESEF - BAŞVURUNUZ REDDEDİLDİ]")
            logger.info("Sonuç: KREDİ REDDEDİLDİ.")
            
        print(f"-> Banka Skor Güven Oranı: %{round(prediction_proba[0][0] * 100, 2)}")
        print(f"-> Risk Analiz Oranı: %{round(prediction_proba[0][1] * 100, 2)}")
        print("=========================================\n")
    except Exception as e:
        logger.error(f"Tahminleme esnasında hata meydana geldi: {str(e)}")

if __name__ == "__main__":
    interactive_prediction()