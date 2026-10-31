import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import DistilBertTokenizer
from datasets import Dataset
from sklearn.metrics import accuracy_score
import numpy as np

# Kendi yazdığımız adaptör mimarisini içe aktarıyoruz
from luma_adapter import LUMAModel

# ==========================================
# 1. AYARLAR VE VERİ HAZIRLIĞI
# ==========================================
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 2
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def get_synthetic_data():
    print("Veriler hazırlanıyor...")
    medical_examples = [
        ("Patient presents with acute chest pain and dyspnea", 1),
        ("The lab results show elevated glucose levels in the blood", 1),
        ("Administration of 50mg of Atenolol every 12 hours", 1),
        ("MRI scan reveals a small lesion in the left frontal lobe", 1),
        ("Chronic cough and fever for two weeks suggest pneumonia", 1),
        ("Patient exhibits symptoms of type 2 diabetes mellitus", 1),
        ("Heart rate is 85 bpm, blood pressure 120/80 mmHg", 0),
        ("Prescribed antibiotics for a suspected urinary tract infection", 1),
        ("The patient reports severe migraines and nausea", 1),
        ("X-ray of the chest shows clear lung fields", 0),
        ("Regular exercise is recommended for overall wellbeing", 0),
        ("Drink plenty of water and get enough sleep", 0),
        ("The patient is stable and showing signs of recovery", 0),
        ("Patient reports mild discomfort in the lower back", 1),
        ("Blood pressure remains within normal range", 0),
    ]
    texts, labels = [], []
    for _ in range(40): 
        for t, l in medical_examples:
            texts.append(t)
            labels.append(l)
    return texts, labels

# ==========================================
# 2. DENEY FONKSİYONU
# ==========================================
def run_experiment(num_examples):
    print(f"\n🚀 Deney Başlatılıyor: {num_examples}-Shot Adaptasyon")
    
    # Veri Hazırlama
    texts, labels = get_synthetic_data()
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    
    # Eğitim ve Test Setlerini Ayır
    train_texts = texts[:num_examples]
    train_labels = labels[:num_examples]
    test_texts = texts[num_examples:]
    test_labels = labels[num_examples:]
    
    # Tokenize et
    train_enc = tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt")
    test_enc = tokenizer(test_texts, padding=True, truncation=True, return_tensors="pt")
    
    # Model ve Optimizer
    model = LUMAModel(num_labels=NUM_LABELS).to(device)
    optimizer = torch.optim.Adam(
        list(model.luma_adapter.parameters()) + list(model.classifier.parameters()), 
        lr=1e-3
    )
    criterion = nn.CrossEntropyLoss()
    
    # Eğitim Döngüsü (Hızlı Adaptasyon)
    model.train()
    for epoch in range(10): # Küçük veri için kısa eğitim
        optimizer.zero_grad()
        outputs = model(train_enc["input_ids"].to(device), train_enc["attention_mask"].to(device))
        loss = criterion(outputs, torch.tensor(train_labels).to(device))
        loss.backward()
        optimizer.step()
    
    # Değerlendirme
    model.eval()
    with torch.no_grad():
        test_outputs = model(test_enc["input_ids"].to(device), test_enc["attention_mask"].to(device))
        preds = torch.argmax(test_outputs, dim=1)
        acc = accuracy_score(test_labels, preds.cpu().numpy())
    
    print(f"✅ {num_examples}-Shot Accuracy: %{acc*100:.2f}")
    return acc

# ==========================================
# 3. ANA ÇALIŞTIRICI
# ==========================================
if __name__ == "__main__":
    print("="*40)
    print("LUMA ADAPTER PERFORMANCE TEST")
    print("="*40)
    
    # 1 Örnekle ne yapıyor?
    acc_1 = run_experiment(1)
    
    # 5 Örnekle ne yapıyor?
    acc_5 = run_experiment(5)
    
    print("\n" + "="*40)
    print("SİSTEM KARŞILAŞTIRMASI")
    print("="*40)
    print(f"LUMA (1-Shot): %{acc_1*100:.2f}")
    print(f"LUMA (5-Shot): %{acc_5*100:.2f}")
    print("-" * 20)
    print(f"Baseline (KNN/Proto): %90.80 (Ortalama)")
    print("="*40)
    
    if acc_1 > 0.80:
        print("\n🌟 MUHTEŞEM: LUMA sadece 1 örnekle baseline'a yaklaştı!")
    elif acc_5 >= 0.90:
        print("\n✅ BAŞARILI: LUMA 5 örnekle baseline başarısını yakaladı.")
    else:
        print("\n📉 Geliştirme Gerekli: Adaptör parametrelerini optimize etmeliyiz.")