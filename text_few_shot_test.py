import torch
from luma_adapter import LUMAModel
from transformers import DistilBertTokenizer

# 1. Hazırlık: Eğitilmiş Nuance Modelini Yükle
model = LUMAModel(num_labels=3)
model.load_state_dict(torch.load("/Users/defnemelisungor/LUMA/luma_trained.pt"))
model.eval()

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# 2. NÜANSLI Mini Metin Veri Seti (Few-Shot)
data = {
    "Pozitif": [
        "Bu ürün tek kelimeyle harika, bayıldım!", 
        "İnanılmaz bir performans sergiledi", 
        "Kesinlikle tavsiye ediyorum, mükemmel"
    ],
    "Negatif": [
        "Tam bir hayal kırıklığı, hiç beğenmedim", 
        "Kullanımı çok zor ve karmaşık", 
        "Parama yazık oldu, sakın almayın"
    ],
    "Resmi": [
        "İlgili makamlara arz ederim", 
        "Söz konusu rapor incelenmiş olup", 
        "Yasal mevzuat çerçevesinde değerlendirilmiştir"
    ]
}

# Soru Cümleleri (Modelin daha önce görmediği, tonu benzer cümleler)
queries = {
    "Pozitif": "Kesinlikle almalısınız, muazzam bir deneyim.",
    "Negatif": "Sürekli sorun çıkarıyor, çok pişmanım.",
    "Resmi": "Söz konusu talep değerlendirmeye alınmıştır."
}

def get_embeddings(texts, use_adapter=True):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    if use_adapter:
        return model.get_embeddings(inputs['input_ids'], inputs['attention_mask'])
    else:
        with torch.no_grad():
            outputs = model.distilbert(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            return outputs.last_hidden_state[:, 0, :]

print("🚀 LUMA Nüans-Testi Başlıyor...\n")

all_support_texts = []
all_support_labels = []
label_map = {label: i for i, label in enumerate(data.keys())}

for label, texts in data.items():
    all_support_texts.extend(texts)
    all_support_labels.extend([label_map[label]] * len(texts))

support_labels = torch.tensor(all_support_labels)
query_texts = list(queries.values())
query_labels = torch.tensor([label_map[l] for l in queries.keys()])

# A. Ham BERT ile Tahmin
emb_support_raw = get_embeddings(all_support_texts, use_adapter=False)
emb_query_raw = get_embeddings(query_texts, use_adapter=False)
preds_raw = model.prototypical_predict(emb_support_raw, support_labels, emb_query_raw)
acc_raw = (preds_raw == query_labels).float().mean().item()

# B. Eğitilmiş LUMA ile Tahmin
emb_support_luma = get_embeddings(all_support_texts, use_adapter=True)
emb_query_luma = get_embeddings(query_texts, use_adapter=True)
preds_luma = model.prototypical_predict(emb_support_luma, support_labels, emb_query_luma)
acc_luma = (preds_luma == query_labels).float().mean().item()

print(f"📊 Ham BERT Doğruluğu: %{acc_raw*100:.2f}")
print(f"🌟 Eğitilmiş LUMA Doğruluğu: %{acc_luma*100:.2f}")

if acc_luma > acc_raw:
    print("\n✅ ZAFER: LUMA Adapter ton farklarını yakaladı ve ham BERT'i geçti!")
elif acc_luma == acc_raw:
    print("\n⚖️ SONUÇ: Benzer sonuçlar verildi.")
else:
    print("\n⚠️ SONUÇ: Ham BERT hala daha iyi.")