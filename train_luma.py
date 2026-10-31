import torch
import torch.nn as nn
import torch.optim as optim
from luma_adapter import LUMAModel
from transformers import DistilBertTokenizer

model = LUMAModel(num_labels=3)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
optimizer = optim.Adam(model.parameters(), lr=1e-5) 
criterion = nn.CrossEntropyLoss()

# NÜANSLI VERİ SETİ: Sadece konu değil, TON üzerine odaklı
train_data = {
    "Pozitif": [
        "Bu ürün tek kelimeyle harika, bayıldım!", "İnanılmaz bir performans sergiledi", 
        "Kesinlikle tavsiye ediyorum, mükemmel", "Hayatımda gördüğüm en iyi çözüm", 
        "Süper bir deneyimdi, çok mutluyum", "Her kuruşuna değen bir hizmet", 
        "Muhteşem bir başarı hikayesi", "Gerçekten etkileyici bir çalışma"
    ],
    "Negatif": [
        "Tam bir hayal kırıklığı, hiç beğenmedim", "Kullanımı çok zor ve karmaşık", 
        "Parama yazık oldu, sakın almayın", "Sürekli hata veriyor, berbat", 
        "Müşteri hizmetleri çok ilgisizdi", "Beklentilerimin çok altında kaldı", 
        "Çok yavaş çalışıyor, kabul edilemez", "Pişmanlık duyuyorum"
    ],
    "Resmi": [
        "İlgili makamlara arz ederim", "Söz konusu rapor incelenmiş olup", 
        "Yasal mevzuat çerçevesinde değerlendirilmiştir", "Resmi prosedürler uygulanmıştır", 
        "Kurumsal standartlar gereği uygun görülmüştür", "Tutanak altına alınmıştır", 
        "İmzalı nüshalar ekte sunulmuştur", "İdari kararlar yürürlüğe girmiştir"
    ]
}

all_texts = []
all_labels = []
label_map = {label: i for i, label in enumerate(train_data.keys())}

for label, texts in train_data.items():
    all_texts.extend(texts)
    all_labels.extend([label_map[label]] * len(texts))

inputs = tokenizer(all_texts, padding=True, truncation=True, return_tensors="pt")
labels = torch.tensor(all_labels)

model.train()
print("🧠 LUMA Nuance-Training Başlıyor...")

for epoch in range(100):
    optimizer.zero_grad()
    logits = model(inputs['input_ids'], inputs['attention_mask'])
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}/100 - Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "/Users/defnemelisungor/LUMA/luma_trained.pt")
print("\n✅ Nuance-Update tamamlandı!")
