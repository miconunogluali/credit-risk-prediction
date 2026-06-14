import os
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class CreditModelTrainer:
    def __init__(self, df, target_column='loan_status'):
        """
        Model eğitimi, değerlendirilmesi ve diske kaydedilmesinden sorumlu sınıf.
        """
        self.df = df
        self.target_column = target_column
        self.model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def prepare_splits(self):
        """
        Veriyi %80 Eğitim, %20 Test olarak böler. Finansal denge için stratify kullanır.
        """
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"[INFO] Veri bölündü. Eğitim: {self.X_train.shape}, Test: {self.X_test.shape}")

    def train_xgboost(self):
        """
        XGBoost modelini eğitir.
        """
        print("[INFO] XGBoost model eğitimi başlıyor...")
        self.model.fit(self.X_train, self.y_train)
        print("[SUCCESS] Model başarıyla eğitildi.")

    def evaluate_model(self):
        """
        Modelin test seti üzerindeki performansını raporlar.
        """
        y_pred = self.model.predict(self.X_test)
        print("\n--- Model Performans Raporu ---")
        print(classification_report(self.y_test, y_pred))

    def save_model(self):
        """
        Eğitilen modeli ve test setinin sütun yapısını 'models/' klasörüne kaydeder.
        """
        # Proje kök dizininde 'models' klasörü yoksa otomatik oluşturur
        os.makedirs("models", exist_ok=True)
        
        model_path = os.path.join("models", "credit_risk_model.joblib")
        columns_path = os.path.join("models", "model_columns.joblib")
        
        # Modeli ve tahmin anında sütun kontrolü yapmak için özellikleri kaydediyoruz
        joblib.dump(self.model, model_path)
        joblib.dump(list(self.X_train.columns), columns_path)
        
        print(f"[SUCCESS] Model ve sütun yapısı başarıyla kaydedildi:")
        print(f" -> {model_path}")
        print(f" -> {columns_path}")

    def run_training_pipeline(self):
        """
        Tüm eğitim adımlarını sırasıyla tetikler.
        """
        self.prepare_splits()
        self.train_xgboost()
        self.evaluate_model()
        self.save_model() # Kaydetme fonksiyonunu pipeline'a dahil ettik
        return self.model