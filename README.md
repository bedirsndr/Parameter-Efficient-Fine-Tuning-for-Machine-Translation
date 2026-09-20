# LoRA ve QLoRA ile Makine Çevirisi: İngilizce → Türkçe (WMT16)

**Qwen2.5-7B-Instruct** modelinin, WMT16 (tr-en) verisiyle **LoRA** ve **QLoRA** yöntemleriyle parametre-verimli ince ayarı (PEFT) ve iki yöntemin çeviri kalitesi, eğitim süresi ve bellek kullanımı açısından karşılaştırması.

Kod: [`PEFT_Machine_Translation.ipynb`](PEFT_Machine_Translation.ipynb)

> Bu çalışma bir ders ödevi (Homework 3) kapsamında hazırlanmıştır. Notebook başlığında en↔tr geçse de uygulanan çeviri yönü yalnızca **İngilizce → Türkçe**'dir.

---

## Sonuçlar

Eğitim: 5.000 örnek, 2 epoch. Değerlendirme: WMT16 validation setinden ilk 100 örnek.

| | LoRA | QLoRA |
|---|:---:|:---:|
| Temel model hassasiyeti | bf16 | 4-bit NF4 (double quant) |
| Eğitilebilir parametre | 10.092.544 | 10.092.544 |
| Toplam parametre* | 7.625.709.056 | 4.363.064.832 |
| Eğitilebilir oran* | %0.132 | %0.231 |
| Eğitim süresi | **10.2 dk** | 19.3 dk |
| Maks. VRAM (ölçülen) | 23.73 GB | 27.14 GB |
| BLEU (sacreBLEU) | 8.29 | **9.64** |
| COMET (wmt22-comet-da) | **0.7826** | 0.7790 |

\* QLoRA'da 4-bit ağırlıklar paketlenmiş biçimde saklandığından `numel()` ile sayılan toplam parametre, gerçek parametre sayısının (~7.6B) yaklaşık yarısı kadar çıkar; bu yüzden QLoRA'daki eğitilebilir oran yanıltıcıdır. Her iki yöntemde de eğitilen parametre sayısı aynıdır.

**Yorum:** İki yöntem aynı sayıda parametreyi eğitiyor. QLoRA BLEU'da, LoRA COMET'te önde; farklar küçük ve değerlendirme 100 örnek ve tek çalıştırmayla sınırlı olduğundan sonuç kesin bir üstünlük göstermiyor. QLoRA, eğitimi yaklaşık 1.9 kat yavaşlattı. Bellek kazancı ise bu çalıştırmada ölçülemedi (aşağıdaki nota bakın).

### Örnek çeviriler

| | Metin |
|---|---|
| EN | Norway's rakfisk: Is this the world's smelliest fish? |
| Referans | Norveç'in rakfisk'i: Dünyanın en kokulu balığı bu mu? |
| LoRA | Norveç'in rakfiski: Dünya en kötü kokulu balık mı? |
| QLoRA | Norveç'in rakfiski: Dünya en kötü kokulu balık mı? |

| | Metin |
|---|---|
| EN | Take a selection of over-ripe cheeses. |
| Referans | Bir zamanı geçmiş peynir seçkisi alın. |
| LoRA | Yarım yaramış peynirlerin bir kısımını alın. |
| QLoRA | Üstüne çöken peynirlerin bir bölümünü alın. |

Kısa ve basit cümlelerde her iki model de anlaşılır çeviriler üretiyor; deyimsel veya daha uzun cümlelerde anlam kaymaları ve dilbilgisi hataları görülüyor.

---

## Veri Seti: WMT16 (tr-en)

| Bölüm | Örnek |
|---|---:|
| Train | 205.756 |
| Validation | 1.001 |
| Test | 3.000 |

Bu çalışmada hız için eğitimde ilk **5.000**, kayıp değerlendirmesinde ilk **500**, çeviri kalitesi ölçümünde ilk **100** validation örneği kullanıldı.

Prompt biçimi:

```
### Instruction:
Translate the following English text to Turkish:
{en_text}

### Response:
{tr_text}
```

---

## Yaklaşım

```
WMT16 tr-en (205.756 çift)
      │  ilk 5.000 örnek, instruction prompt, max 256 token
      ▼
Qwen2.5-7B-Instruct
      │
      ├─► LoRA  : taban model bf16 ──────────┐
      │                                       │  q,k,v,o_proj üzerine
      └─► QLoRA : taban model 4-bit NF4 ──────┤  LoRA (r=16, α=32)
                (double quant, paged 8-bit)   │
                                              ▼
                              SFTTrainer, 2 epoch ──► adapter
                                              │
                                              ▼
                         greedy decoding (100 örnek) ──► BLEU + COMET
```

### Yapılandırma

| Parametre | Değer |
|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| LoRA rank (r) / alpha / dropout | 16 / 32 / 0.05 |
| Hedef modüller | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Epoch | 2 |
| Batch size × gradient accumulation | 4 × 4 (efektif 16) |
| Learning rate / Warmup | 2e-4 / 50 adım |
| Optimizer | `paged_adamw_8bit` |
| Precision | bf16 |
| Maks. uzunluk | 256 |
| Decoding | Greedy, `max_new_tokens=128` |
| Donanım | NVIDIA A100 40 GB (Colab) |

### Modelin seçilme gerekçesi

| Kriter | Gerekçe |
|---|---|
| Boyut | 7B parametre; QLoRA ile sınırlı VRAM'de çalışabilir |
| Talimat takibi | Instruct sürümü; `translate X to Turkish` tarzı promptlarla uyumlu |
| Çok dillilik | Türkçe dahil 29+ dil desteği |
| Lisans | Apache 2.0 |

### LoRA ve QLoRA kısaca

- **LoRA:** Ağırlık güncellemesini `ΔW = B·A` şeklinde düşük ranklı iki küçük matrisle öğrenir; taban ağırlıklar dondurulur. Örnek: 4096×4096'lık bir katman için tam ince ayar 16.777.216, LoRA (r=16) 131.072 parametre gerektirir (128 kat azalma).
- **QLoRA:** LoRA'ya ek olarak taban modeli 4-bit **NF4** formatında tutar, nicemleme sabitlerini de sıkıştırır (**double quantization**) ve **paged optimizer** kullanır. Amaç, aynı adapter eğitimini çok daha az bellekle yapabilmektir.

---

## Kurulum ve Çalıştırma

Notebook, Google Colab üzerinde bir **A100 GPU** ile çalıştırılmıştır (bf16 ve 7B model için yaklaşık 24-27 GB VRAM görüldü).

```bash
pip install -U transformers peft trl bitsandbytes accelerate datasets \
    sacrebleu unbabel-comet evaluate sentencepiece
```

1. `PEFT_Machine_Translation.ipynb` dosyasını Colab'da açıp GPU çalışma zamanı seçin.
2. Hücreleri sırayla çalıştırın: veri hazırlama → LoRA eğitimi → QLoRA eğitimi → değerlendirme.
3. Çıktılar `/content` altında oluşur:
   - `lora_adapter/`, `qlora_adapter/`
   - `results_summary.csv`, `all_results.json`, `lora_vs_qlora.png`

Değerlendirme ilk çalıştırmada `Unbabel/wmt22-comet-da` modelini indirir (~1.5 GB).

> Notebook'un ilk kurulum hücresi eski sürümleri sabitleyip kernel'i yeniden başlatır; sonraki hücreler `trl` ve `bitsandbytes`'ı yükseltir. Nihai çalıştırma ortamı Colab (Python 3.12, PyTorch 2.10 + CUDA 12.8) idi. `datasets` yeni sürümlerinde `load_dataset("wmt16", ...)` çağrısındaki `trust_remote_code` argümanı desteklenmez; veri yine de yüklenir, argüman kaldırılabilir.

---

## Notlar ve Sınırlılıklar

- **VRAM karşılaştırması temiz değil.** QLoRA hücresinde LoRA modelini bellekten silen satırlar yorum satırı halindedir; bu yüzden QLoRA eğitimi sırasında LoRA modeli hâlâ GPU'da olabilir ve ölçülen tepe değer (27.14 GB) bunu içerebilir. QLoRA'nın literatürdeki bellek avantajı bu çalıştırmada doğrulanamadı. Temiz ölçüm için her yöntem ayrı oturumda (veya bellek temizlenerek) çalıştırılmalıdır.
- Notebook'taki "Teori" bölümünde verilen VRAM (~14 GB / ~5-6 GB), hız (~%30) ve COMET farkı (0.01-0.03) aralıkları genel literatür beklentileridir; bu çalışmada ölçülen değerler yukarıdaki tablodadır.
- Notebook'un tartışma bölümü "tek epoch" der; yapılandırmada ve gerçek çalıştırmada **2 epoch** kullanılmıştır.
- **Taban model (fine-tune edilmemiş Qwen) referansı yok.** LoRA/QLoRA'nın taban modele göre ne kadar iyileştirme sağladığı bu notebook'ta ölçülmemiştir.
- Değerlendirme yalnızca 100 örnekle ve tek çalıştırmayla yapıldı; BLEU/COMET farkları istatistiksel olarak anlamlı kabul edilmemelidir.
- Etiketler `input_ids`'in kopyasıdır (`labels = input_ids`); prompt ve padding token'ları loss dışında bırakılmamıştır.
- Yalnızca dikkat (attention) projeksiyonlarına LoRA uygulanmış, feed-forward katmanlar dahil edilmemiştir. Eğitim, tam veri setinin ~%2.4'ü ile yapılmıştır.

## Olası Geliştirmeler

- Tüm eğitim verisi, daha fazla epoch ve feed-forward katmanları da hedefleyen LoRA
- Yalnızca yanıt kısmında loss (prompt maskeleme)
- Fine-tune edilmemiş taban model ile karşılaştırma
- Bellek ölçümünü her yöntem için izole oturumlarda tekrarlamak
- Test setinin tamamında (3.000 örnek) değerlendirme ve birden fazla tohum

## Repo Yapısı

```
peft-machine-translation/
├── PEFT_Machine_Translation.ipynb
└── README.md
```
