# Gazetype

GazeType, kamera üzerinden sadece gözlerinizle klavye kullanmayı mümkün kılmak
için oluşturulmuş bir yazılımdır. Windows'ta standart bir web kamerasıyla göz hareketlerini kullanarak
aktif uygulamaya yazı yazmayı amaçlayan erişilebilirlik odaklı bir masaüstü
uygulamasıdır.

> Bu proje bir tıbbi cihaz değildir. İlk sürüm bir kullanılabilirlik ve teknik
> fizibilite prototipidir.

## Özellikler

- Türkçe Q ve İngilizce QWERTY ekran klavyeleri
- Bağlı kameraları ayırt etmek için canlı kamera önizleme kartları
- Çoklu ekran seçimi
- 20–81 arasında ayarlanabilen nokta sayısıyla hassas bakış kalibrasyonu
- Seçili klavyedeki her tuş merkezini hedef alan alternatif kalibrasyon modu
- Hızlı göz sıçramasından sonraki iniş noktasını seçme
- Kamera parazitini azaltmak için 1–30 bakış arasında ayarlanabilen hareketli ortalama
- Kafa dönüşü, eğimi, yuvarlanması ve kameraya uzaklığına göre bakış telafisi
- Hızlı kafa hareketlerinde yanlış tuş basımını önleyen kısa güvenlik duraklatması
- Canlı pencereden değiştirilebilen yatay/dikey kazanç, dikey ofset, kafa telafisi
  ve kafa hareket eşiği deney kontrolleri
- Space, Backspace ve Enter desteği
- Bilinçli göz kırpma veya fare ile açma/kapatma
- Tamamen cihaz üzerinde kamera işleme

## Kullanım

1. Gazetype'i başlatın.
2. Canlı önizleme kartından kamerayı; ardından hedef ekranı, klavye düzenini
   ve hassasiyeti seçin. Kullanılamayan kamera girişleri kart üzerinde belirtilir.
3. Kalibrasyonda sırayla gösterilen her noktaya bakarken farenin sol tuşuyla
   ekranın herhangi bir yerine tıklayın.
4. Kalibrasyon bitince yazmak istediğiniz uygulamayı (örneğin Not Defteri)
   odaklayın.
5. Ekranın sağ üstündeki düğmeye bakıp 250–800 ms boyunca bilinçli olarak
   iki gözünüzü kırpın. Düğmeye fareyle de tıklayabilirsiniz.
6. Harfler arasında gözünüzü hızla hareket ettirin. Yalnızca gözün indiği ve
   doğrulanan tuş yazılır; hareket hattındaki tuşlar yok sayılır.

Sistem tepsisi menüsünden yeniden kalibrasyon yapılabilir veya uygulama
kapatılabilir.

### Hassasiyet seviyeleri

- **Hızlı:** 2 doğrulama karesi ve en az 50 ms
- **Dengeli:** 3 doğrulama karesi ve en az 90 ms
- **Sabit:** 4 doğrulama karesi ve en az 130 ms

Düşük kare hızlı kameralarda gereken gerçek süre uzayabilir.

## Geliştirme

Python 3.12 gereklidir.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m gazetype
```

İlk çalıştırmadan önce MediaPipe modelini indirin:

```powershell
.\scripts\download_model.ps1
```

Testler:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Windows paketi:

```powershell
.\scripts\build_windows.ps1
```

### General gaze model + personal calibration

Gazetype, mevcut kişisel kalibrasyon modelini korurken opsiyonel bir genel gaze
modeli ve onun üstünde kişisel calibration adapter kullanabilir. Genel model
`.npz` formatındadır ve `GAZETYPE_GENERAL_MODEL` ortam değişkeniyle ya da ayar
dosyasındaki `general_gaze_model_path` alanıyla gösterilir. Model dosyası yoksa
uygulama sessizce eski kişisel kalibrasyon davranışına döner.

Genel model eğitimi için başlangıç aracı:

```powershell
python tools/train_gaze_model.py --dataset weyeds --data-dir C:\datasets\weyeds-normalized --out models\gazetype_general.npz
python tools/train_gaze_model.py --quick --out models\smoke_general.npz
```

`models/`, `.npz` ve `.onnx` çıktıları git dışında tutulur. Public datasetlerden
üretilmiş ağırlıklar, dataset/model lisansı netleşmeden bu repo içinde
dağıtılmaz.

### Personal mouse-target dataset

Kendi ekran koordinatlı verinizi toplamak için mouse hedefli collector
kullanılabilir. Tam ekran bir katman açılır; imlecin olduğu noktaya bakıp sol
tıkladığınızda o anki kamera feature'ları ve tıklanan ekran hedefi
`training_samples.jsonl` dosyasına yazılır. Farklı kafa konumlarıyla çok sayıda
örnek almak kişisel model kalitesini doğrudan artırır.

```powershell
python -m gazetype.dataset_collector --camera-index 0
python -m tools.train_gaze_model --dataset weyeds --data-dir $env:LOCALAPPDATA\Gazetype --out models\personal_general.npz --polynomial-degree 2
```

Linux'ta varsayılan kayıt klasörü `~/.local/share/gazetype` olduğu için eğitim
komutu şu şekilde çalıştırılabilir:

```bash
python -m gazetype.dataset_collector --camera-index 0 --moving --duration-seconds 120 --reaction-lag-ms 250 --out ~/.local/share/gazetype/gazetype_manifest.jsonl
python -m tools.train_gaze_model --dataset weyeds --data-dir ~/.local/share/gazetype --out models/personal_general.npz --polynomial-degree 1
GAZETYPE_GENERAL_MODEL=models/personal_general.npz python -m gazetype
```

Uygulamada gelişmiş ayarlardan "Genel gaze modelini kullan" seçeneğini açıp
"Kişisel Modelle Başlat" düğmesine basarsanız, eğitilmiş `.npz` model doğrudan
kalibrasyon objesi olmadan kullanılır.

Ana penceredeki "Hareketli Dataset" düğmesi de aynı akışı başlatır: hedef
ekranı dolaşırken her kamera frame'i kaydedilir ve varsayılan 250 ms takip
gecikmesi hesaba katılır. "Nokta Dataset" modu ise her hedef için Space veya sol
tık bekler. İyi sonuç için her kafa/oturuş pozisyonunda ekranın tamamını
dolaşın; sadece belirli ekran bölgelerini belirli kafa pozisyonlarında toplamak
modelin hedef yerine kafa pozisyonunu öğrenmesine yol açabilir.

### WEyeDS support

Gazetype WEyeDS'i otomatik indirmez. Yerel datasetinizi normalize edilmiş bir
ara manifest formatına dönüştürmeniz gerekir. `tools/datasets/weyeds.py` dataset
klasöründe `gazetype_manifest.csv`, `gazetype_manifest.jsonl`,
`mpiigaze_gazetype_manifest.jsonl`, `manifest.csv`, `manifest.jsonl`,
`samples.csv`, `samples.jsonl` veya `training_samples.jsonl` arar.

Kabul edilen alanlar:

- `target_x`, `target_y`
- `features` JSON listesi veya `feature_0` ... `feature_9`
- `image_path` veya `face_path`
- opsiyonel `left_eye_path`, `right_eye_path`, `screen_width`, `screen_height`

İlk baseline trainer ham görüntüden MediaPipe feature çıkarmaz; manifestte
önceden çıkarılmış 10 Gazetype gaze feature'ı bekler.

### MPIIGaze local conversion

MPIIGaze indirildikten ve yerelde açıldıktan sonra raw görüntülerden Gazetype
feature manifest'i üretilebilir:

```powershell
python -m tools.build_mpiigaze_manifest --data-dir data\MPIIGaze --out data\mpiigaze_gazetype_manifest.jsonl --limit 30000 --eye left --auto-range
python -m tools.train_gaze_model --dataset weyeds --data-dir data --out models\mpiigaze_general.npz --polynomial-degree 2
```

MPIIGaze doğrudan ekran `target_x/target_y` etiketi vermez; gaze vector
değerleri pseudo screen target'a dönüştürülür. Bu nedenle MPIIGaze modeli genel
başlangıç noktasıdır, gerçek kullanımda kişisel kalibrasyon adapter'ı ile
düzeltilmelidir.

## Gizlilik

Kamera kareleri yalnızca bellekte işlenir; kaydedilmez veya ağ üzerinden
gönderilmez. Ayarlar ve kalibrasyon verileri kullanıcının yerel uygulama veri
klasöründe tutulur.

Gelişmiş ayarlardan etkinleştirilirse kalibrasyon örnekleri yerel uygulama veri
klasöründe `training_samples.jsonl` olarak saklanabilir. Her satır hedef x/y,
feature listesi, ekran geometrisi ve kamera indeksini içerir. Raw kamera
görüntüsü varsayılan olarak kaydedilmez; kişisel kalibrasyon verisi cihazda
kalır.

## Sınırlamalar

- İlk sürüm yalnızca Windows 10/11 x64 içindir.
- Yönetici olarak çalışan bir uygulamaya, normal yetkideki Gazetype tuş
  gönderemeyebilir.
- Kelime tahmini, fare kontrolü ve klinik doğruluk iddiası yoktur.

## Lisans

MIT
