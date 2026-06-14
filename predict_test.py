import os
import joblib
import pandas as pd
import numpy as np

def calculate_financial_metrics(score):
    """
    Findeks skoruna göre kredi derecesini (grade), faiz oranını (interest rate)
    ve geçmiş temerrüt durumunu otomatik belirleyen bankacılık iş mantığı motoru.
    """
    if 1700 <= score <= 1900:
        return 'A', 11.0, 0
    elif 1500 <= score < 1700:
        return 'B', 13.5, 0
    elif 1300 <= score < 1500:
        return 'C', 16.0, 0
    elif 1100 <= score < 1300:
        return 'D', 19.5, 0
    elif 700 <= score < 1100:
        return 'E', 24.0, 0
    elif 1 <= score < 700:
        return 'F', 29.0, 1  # Düşük puanlıların yasal takibi var kabul edilir
    else:
        return 'G', 35.0, 1  # 0 Puan alanlar

def get_user_inputs():
    print("\n--- 📝 Lütfen Kredi Başvuru Bilgilerinizi Giriniz ---")
    
    # 1. Temel Sayısal Sorular
    age = int(input("Yaşınız (Örn: 28): "))
    income = float(input("Yıllık Net Geliriniz ($) (Örn: 60000): "))
    emp_length = int(input("Mevcut İşinizdeki Çalışma Süreniz (Yıl): "))
    loan_amnt = float(input("Talep Ettiğiniz Kredi Miktarı ($): "))
    cred_hist_length = int(input("İlk Kredi Kartı/Kredi Kullanımınızdan Bu Yana Kaç Yıl Geçti?: "))
    
    # 2. Findeks Skor Girişi (0 - 1900 Kontrolü)
    while True:
        score = int(input("Findeks Kredi Notunuz (0 - 1900 arası): "))
        if 0 <= score <= 1900:
            break
        print("❌ Hatalı Giriş! Kredi notu 0 ile 1900 arasında olmalıdır.")

    # İş Mantığı Motorunu Tetikliyoruz
    grade_input, loan_int_rate, default_status = calculate_financial_metrics(score)
    loan_percent_income = loan_amnt / income

    # 3. Temel Kategorik Sorular
    print("\n🏠 Ev Sahipliği Durumunuz: RENT (Kiracı), OWN (Ev Sahibi), MORTGAGE (Kredili Ev), OTHER (Diğer)")
    home_input = input("Giriş yapın: ").upper().strip()

    print("\n🎯 Krediyi Ne İçin Kullanacaksınız?: EDUCATION (Eğitim), MEDICAL (Sağlık), VENTURE (Girişim/İş), PERSONAL (Kişisel), HOMEIMPROVEMENT (Ev Tadilatı)")
    intent_input = input("Giriş yapın: ").upper().strip()

    # Modelin beklediği One-Hot biçimine dönüştürme
    customer_dict = {
        'person_age': age, 
        'person_income': income, 
        'person_emp_length': emp_length,
        'loan_amnt': loan_amnt, 
        'loan_int_rate': loan_int_rate, 
        'loan_percent_income': loan_percent_income,
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
    
    # Kullanıcıyı bilgilendirme
    print(f"\n[SİSTEM] Hesaplanan Kredi Derecesi: {grade_input}")
    print(f"[SİSTEM] Bankamızın Size Özel Belirlediği Faiz Oranı: %{loan_int_rate}")
    
    return pd.DataFrame([customer_dict])

def interactive_prediction():
    print("=========================================")
    print("🏦 OTOMATİK KREDİ BAŞVURU VE TEST SİSTEMİ ")
    print("=========================================")
    
    model_path = os.path.join("models", "credit_risk_model.joblib")
    columns_path = os.path.join("models", "model_columns.joblib")
    
    if not os.path.exists(model_path) or not os.path.exists(columns_path):
        print("[ERROR] Model dosyaları bulunamadı! Lütfen önce main.py çalıştırın.")
        return

    model = joblib.load(model_path)
    model_columns = joblib.load(columns_path)
    
    df_customer = get_user_inputs()
    df_customer = df_customer.reindex(columns=model_columns, fill_value=0)
    
    prediction = model.predict(df_customer)
    prediction_proba = model.predict_proba(df_customer)
    
    print("\n=========================================")
    print("       📊 ANLIK BAŞVURU SONUCU          ")
    print("=========================================")
    if prediction[0] == 0:
        print("✅ [TEBRİKLER - KREDİNİZ ONAYLANDI]")
        print("Sistemimiz başvurunuzu düşük riskli olarak değerlendirdi.")
    else:
        print("❌ [MAALESEF - BAŞVURUNUZ REDDEDİLDİ]")
        print("Mevcut finansal verileriniz politikalarımızla uyuşmamaktadır.")
        
    print(f"-> Banka Skor Güven Oranı: %{round(prediction_proba[0][0] * 100, 2)}")
    print(f"-> Risk Analiz Oranı: %{round(prediction_proba[0][1] * 100, 2)}")
    print("=========================================\n")

if __name__ == "__main__":
    interactive_prediction()