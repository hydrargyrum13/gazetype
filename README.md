# Gazetype

Gazetype, Windows'ta standart bir web kamerasıyla göz hareketlerini kullanarak
aktif uygulamaya yazı yazmayı amaçlayan erişilebilirlik odaklı bir masaüstü
uygulamasıdır.

> Bu proje bir tıbbi cihaz değildir. İlk sürüm bir kullanılabilirlik ve teknik
> fizibilite prototipidir.

## Özellikler

- Türkçe Q ve İngilizce QWERTY ekran klavyeleri
- Bağlı kameraları ayırt etmek için canlı kamera önizleme kartları
- Çoklu ekran seçimi
- Dokuz noktalı bakış kalibrasyonu
- Hızlı göz sıçramasından sonraki iniş noktasını seçme
- Space, Backspace ve Enter desteği
- Bilinçli göz kırpma veya fare ile açma/kapatma
- Tamamen cihaz üzerinde kamera işleme

## Kullanım

1. Gazetype'i başlatın.
2. Canlı önizleme kartından kamerayı; ardından hedef ekranı, klavye düzenini
   ve hassasiyeti seçin. Kullanılamayan kamera girişleri kart üzerinde belirtilir.
3. Kalibrasyonda sırayla gösterilen dokuz noktaya bakın.
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

## Gizlilik

Kamera kareleri yalnızca bellekte işlenir; kaydedilmez veya ağ üzerinden
gönderilmez. Ayarlar ve kalibrasyon verileri kullanıcının yerel uygulama veri
klasöründe tutulur.

## Sınırlamalar

- İlk sürüm yalnızca Windows 10/11 x64 içindir.
- Yönetici olarak çalışan bir uygulamaya, normal yetkideki Gazetype tuş
  gönderemeyebilir.
- Kelime tahmini, fare kontrolü ve klinik doğruluk iddiası yoktur.

## Lisans

MIT
