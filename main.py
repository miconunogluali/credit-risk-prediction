import os
from src.data_preprocessing import CreditDataPreprocessor
from src.model_training import CreditModelTrainer

def main():
    print("="*60)
    print("       CREDIT RISK PREDICTION PIPELINE - PRODUCTION V1      ")
    print("="*60)

    # 1. Veri setinin yolunu belirliyoruz
    # Proje kök dizinindeki data klasörünün altındaki csv dosyasını hedefliyoruz
    data_path = os.path.join("data", "credit_risk_dataset.csv")
    
    if not os.path.exists(data_path):
        print(f"[ERROR] Veri seti bulunamadı! Lütfen şu yolda dosya olduğundan emin olun: {data_path}")
        return

    # 2. Veri Ön İşleme (Preprocessing) Aşaması
    preprocessor = CreditDataPreprocessor(filepath=data_path)
    df_clean = preprocessor.run_pipeline()

    print("-" * 50)

    # 3. Model Eğitimi (Model Training) Aşaması
    trainer = CreditModelTrainer(df=df_clean, target_column='loan_status')
    model = trainer.run_training_pipeline()

    print("="*60)
    print("   PIPELINE SUCCESSFUL: Model trained and evaluated successfully!  ")
    print("="*60)

if __name__ == "__main__":
    main()