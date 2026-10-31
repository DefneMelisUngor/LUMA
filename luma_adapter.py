import torch
import torch.nn as nn
from transformers import DistilBertModel

# ==========================================
# 1. LUMA ADAPTER KATMANI
# ==========================================
class LUMAAdapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=64):
        super(LUMAAdapter, self).__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, input_dim)
        )
        
    def forward(self, x):
        # Residual connection: orijinal vektör + adapter çıktısı
        return x + self.adapter(x)

# ==========================================
# 2. LUMA MODELİ (Model + Adapter + Proto-Logic)
# ==========================================
class LUMAModel(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", num_labels=2):
        super(LUMAModel, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        
        # BERT katmanlarını dondur (Sadece adapter ve classifier eğitilsin)
        for param in self.distilbert.parameters():
            param.requires_grad = False
            
        self.luma_adapter = LUMAAdapter(768)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        """Standart eğitim ve çıkarım için kullanılır."""
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state[:, 0, :] # [CLS] token
        adapted_state = self.luma_adapter(last_hidden_state)
        logits = self.classifier(adapted_state)
        return logits

    def get_embeddings(self, input_ids, attention_mask):
        """
        Prototip tahmini için gerekli olan 'iyileştirilmiş' vektörleri üretir.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state[:, 0, :]
            adapted_state = self.luma_adapter(last_hidden_state)
        return adapted_state

    def prototypical_predict(self, support_embeddings, support_labels, query_embeddings):
        """
        Deneylerde kanıtlanan Prototip tabanlı tahmin mekanizması.
        Destek setindeki sınıfların ortalamasını alır ve en yakın prototipe atama yapar.
        
        Args:
            support_embeddings: (n_support, dim)
            support_labels: (n_support,)
            query_embeddings: (n_query, dim)
        Returns:
            predictions: (n_query,)
        """
        unique_labels = torch.unique(support_labels)
        prototypes = []
        
        # Her sınıf için prototip (ortalama vektör) hesapla
        for label in unique_labels:
            mask = (support_labels == label)
            proto = support_embeddings[mask].mean(dim=0)
            prototypes.append(proto)
            
        prototypes = torch.stack(prototypes) # (n_ways, dim)
        
        # Öklid mesafesini hesapla (torch.cdist: tüm çiftler arası mesafe)
        # dists: (n_query, n_ways)
        dists = torch.cdist(query_embeddings, prototypes)
        
        # En yakın prototipin indeksini bul
        nearest_idx = torch.argmin(dists, dim=1)
        
        # İndeksi gerçek sınıf etiketine dönüştür
        return unique_labels[nearest_idx]