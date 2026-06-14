import pandas as pd
import numpy as np

class CreditDataPreprocessor:
    def __init__(self, filepath):
        """
        Sınıf başlatıldığında veri setinin bilgisayardaki yolunu alır.
        """
        self.filepath = filepath
        self.df = None

    def load_data(self):
        """Veriyi belirtilen yoldan diskten okur."""
        self.df = pd.read_csv(self.filepath)
        return self.df

    def clean_anomalies(self):
        """Bankacılık mantığına aykırı uç değerleri (94 yaş, 41 yıl) filtreler."""
        if self.df is None:
            raise ValueError("Veri henüz yüklenmedi! Önce load_data() çağırılmalı.")
        
        # Yaş ve çalışma süresi filtrelerini uyguluyoruz
        self.df = self.df[(self.df['person_age'] < 100) & (self.df['person_emp_length'] < 60)].copy()
        return self.df

    def handle_missing_values(self):
        """Eksik (Null) verileri medyan değerleri ile doldurur."""
        # Çalışma süresi boşluklarını doldurma
        emp_median = self.df['person_emp_length'].median()
        self.df['person_emp_length'] = self.df['person_emp_length'].fillna(emp_median)
        
        # Faiz oranı boşluklarını doldurma
        int_median = self.df['loan_int_rate'].median()
        self.df['loan_int_rate'] = self.df['loan_int_rate'].fillna(int_median)
        return self.df

    def encode_categorical(self):
        """Kategorik değişkenleri One-Hot Encoding ile sayısallaştırır."""
        categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
        
        # dummy_variable trap'i englemek için drop_first=True yapıyoruz
        self.df = pd.get_dummies(self.df, columns=categorical_cols, drop_first=True)
        return self.df

    def run_pipeline(self):
        """Tüm ön işleme adımlarını sırasıyla otomatize eden ana motor fonksiyondur."""
        print("[INFO] Veri ön işleme boru hattı (Pipeline) başladı...")
        self.load_data()
        self.clean_anomalies()
        self.handle_missing_values()
        self.encode_categorical()
        print(f"[SUCCESS] Veri başarıyla hazırlandı. Yeni boyut: {self.df.shape}")
        return self.df