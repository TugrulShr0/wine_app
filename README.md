# 🍷 Şarap Kimyası Sınıflandırıcısı

Bu proje, Scikit-learn'ün **Wine** veri seti (178 şarap örneği, 13 kimyasal özellik, 3 üzüm çeşidi) ile eğitilmiş bir PyTorch MLP (Çok Katmanlı Algılayıcı) modelini içerir. Model, FastAPI üzerinden sunulan interaktif bir web uygulamasına entegre edilmiştir.

Test setinde **%97.2 doğruluk** oranına sahip olan bu model; şarabın alkol oranı, malik asit, magnezyum ve renk yoğunluğu gibi özelliklerine bakarak hangi üzüm çeşidine ait olduğunu tahmin etmektedir.

## ✨ Özellikler

- **Makine Öğrenmesi (PyTorch):** 13 → 32 → 32 → 3 mimarisine ve ReLU aktivasyonuna sahip, CPU üzerinde eğitilmiş yapay sinir ağı.
- **Hızlı Backend (FastAPI):** Yüksek performanslı ve asenkron web sunucusu[cite: 2]. `/predict` ve `/api/random-sample` uç noktaları üzerinden hizmet verir.
- **İnteraktif Arayüz (Vanilla JS & CSS):** Kimyasal değerleri manuel olarak ayarlayabileceğiniz kaydırıcılar (slider) ve sonucun güven oranını (`%` olarak) gösteren animasyonlu grafikler.
- **Rastgele Örnek Çekme:** Veri setinden rastgele bir gerçek şarap örneği çekip, modelin tahminini gerçek etiketle (true label) doğrudan kıyaslayabilme imkanı.

