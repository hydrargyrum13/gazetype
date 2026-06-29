# Gazetype

Gazetype, Windows'ta standart bir web kamerasıyla göz hareketlerini kullanarak
aktif uygulamaya yazı yazmayı amaçlayan erişilebilirlik odaklı bir masaüstü
uygulamasıdır.

> Bu proje bir tıbbi cihaz değildir. İlk sürüm bir kullanılabilirlik ve teknik
> fizibilite prototipidir.

## Özellikler

- Türkçe Q ve İngilizce QWERTY ekran klavyeleri
- Çoklu ekran seçimi
- Dokuz noktalı bakış kalibrasyonu
- Hızlı göz sıçramasından sonraki iniş noktasını seçme
- Space, Backspace ve Enter desteği
- Bilinçli göz kırpma veya fare ile açma/kapatma
- Tamamen cihaz üzerinde kamera işleme

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

## Lisans

MIT

