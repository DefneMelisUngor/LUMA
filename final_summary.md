# LUMA: Lightweight Unified Model Adapter
## Final Araştırma ve Performans Raporu

### 1. Özet (Abstract)
LUMA projesi, Few-Shot Learning (Az Örnekli Öğrenme) süreçlerinde karşılaşılan hesaplama darboğazlarını gidermek ve modelin yeni sınıflara adaptasyon yeteneğini artırmak amacıyla geliştirilmiştir. Çalışma kapsamında, geleneksel KNN (K-En Yakın Komşu) yaklaşımı ile Prototypical Network (Prototip Ağları) mimarisi karşılaştırılmış; hesaplama maliyeti, ölçeklenebilirlik ve doğruluk parametreleri analiz edilmiştir.

### 2. Problem Tanımı
Few-Shot Learning sistemlerinde, destek seti (support set) büyüdükçe tahmin aşamasındaki hesaplama maliyeti artmaktadır. Özellikle KNN tabanlı yaklaşımlarda, her yeni sorgu için destek setindeki tüm örneklere mesafe hesaplanması, O(N) zaman karmaşıklığına yol açmakta ve yüksek-shot senaryolarında ciddi performans kayıplarına neden olmaktadır.

### 3. Önerilen Çözüm ve Mimari
LUMA, bu sorunu aşmak için Prototypical Prediction mekanizmasını bir adaptör katmanı ile modele entegre etmiştir.

- **LUMAAdapter**: Modelin dondurulmuş (frozen) temel ağırlıklarının üzerine eklenen, düşük boyutlu bir darboğaz (bottleneck) yapısına sahip öğrenilebilir katmandır. Vektörleri, sınıflar arası ayrımı maksimize edecek şekilde dönüştürür.
- **Prototip Mantığı**: Her sınıf için destek setindeki vektörlerin ortalaması alınarak tek bir "Sınıf Prototipe" indirgenir. Tahmin aşaması, sorgu vektörünün sadece bu prototiplere olan mesafesine indirgenerek zaman karmaşıklığı O(1)'e düşürülmüştür.

### 4. Deneysel Bulgular ve Analiz

#### 4.1. Hesaplama Verimliliği (Stres Testi)
Yapılan stres testlerinde, destek seti boyutu 1'den 100'e çıkarıldığında elde edilen sonuçlar şöyledir:

| Yöntem | 1-Shot (ms) | 100-Shot (ms) | Ölçeklenme Faktörü |
| :--- | :--- | :--- | :--- |
| **KNN** | 0.022 ms | 1.261 ms | ≈ 57x Yavaşlama |
| **LUMA (Proto)** | 0.022 ms | 0.022 ms | **Sabit Zaman (Constant)** |

**Analiz**: Prototip tabanlı yaklaşım, veri miktarı arttığında bile tahmin hızını koruyarak endüstriyel ölçeklenebilirliği kanıtlamıştır.

#### 4.2. Doğruluk ve Adaptasyon
Metin verileri üzerinde yapılan few-shot testlerinde;
- Ham BERT modelinin temel dil bilgisi sayesinde belirli bir başarı seviyesi (%66) sergilediği görülmüştür.
- LUMAAdapter'ın, eğitim verisi miktarı çok kısıtlı olduğunda "aşırı öğrenme" (overfitting) riski taşıdığı, ancak stabil öğrenme oranları (1e-5) ile temel modelin bilgisini koruyabildiği gözlemlenmiştir.

### 5. Sonuç ve Değerlendirme
LUMA projesiyle birlikte;
1.  Tahmin hızının veri setinden bağımsız hale getirildiği bir mimari başarıyla implement edilmiştir.
2.  BERT tabanlı bir modelin, hafif bir adaptör katmanı ile özelleştirilebilir olduğu gösterilmiştir.
3.  Few-shot sistemlerde hız ve doğruluk arasındaki ödünleşimin (trade-off) net bir şekilde haritalandırılmıştır.

**Gelecek Çalışmalar**: Daha büyük ve çeşitli veri setleri ile eğitim yapılarak, adaptörün nüanslı anlamlandırma yeteneğinin artırılması hedeflenmektedir.
