from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class CreditModelTrainer:
    def __init__(self, df, target_column='loan_status'):
        """
        Sınıf başlatıldığında temizlenmiş ve encode edilmiş veri setini alır.
        """
        self.df = df
        self.target_column = target_column
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4

    def prepare_splits(self, test_size=0.2, random_state=42):
        """Veriyi %80 Eğitim, %20 Test olarak kurumsal standartta böler."""
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        
        # Finansal veri dengesizliği için stratify=y kullanıyoruz
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
        print(f"[INFO] Veri bölündü. Eğitim: {self.X_train.shape}, Test: {self.X_test.shape}")

    def train_xgboost(self, n_estimators=100, max_depth=5, learning_rate=0.1):
        """XGBoost modelini tanımlar ve eğitir."""
        print("[INFO] XGBoost model eğitimi başlıyor...")
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            eval_metric='logloss'
        )
        self.model.fit(self.X_train, self.y_train)
        print("[SUCCESS] Model başarıyla eğitildi.")

    def evaluate_model(self):
        """Modelin test verisi üzerindeki performans raporunu basar."""
        if self.model is None:
            raise ValueError("Önce model eğitilmeli! train_xgboost() çağırılmalı.")
        
        y_pred = self.model.predict(self.X_test)
        print("\n--- Model Performans Raporu ---")
        print(classification_report(self.y_test, y_pred))
        
    def run_training_pipeline(self):
        """Bölme, eğitme ve raporlama adımlarını sırasıyla otomatize eder."""
        self.prepare_splits()
        self.train_xgboost()
        self.evaluate_model()
        return self.model