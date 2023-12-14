import numpy as np
import matplotlib.pyplot as plt


def ortalama_hesapla(kume):
    return np.mean(kume, axis=0)


def hata_hesapla(kume, merkez):
    return np.sum(np.square(kume - merkez))


def k_ortalama(X, k, maks_iterasyon=100):
    num_noktalar, num_ozellikler = X.shape
    merkezler = X[np.random.choice(range(num_noktalar), size=k, replace=False)]

    for _ in range(maks_iterasyon):
        etiketler = np.argmin(np.linalg.norm(X[:, np.newaxis] - merkezler, axis=2), axis=1)
        yeni_merkezler = np.array([ortalama_hesapla(X[etiketler == i]) for i in range(k)])

        if np.all(merkezler == yeni_merkezler):
            break

        merkezler = yeni_merkezler

    kume_hatalari = [hata_hesapla(X[etiketler == i], merkezler[i]) for i in range(k)]
    toplam_kare_hata = np.sum(kume_hatalari)

    return etiketler, merkezler, toplam_kare_hata


# Örnek veri oluştur
np.random.seed(42)
X = np.random.randn(300, 2)

# K-Ortalamalar algoritması ile kümeleme
k = 3
etiketler, merkezler, toplam_kare_hata = k_ortalama(X, k)


print("Küme Etiketleri:", etiketler)
print("Merkezler:", merkezler)
print("Toplam Kare Hata:", toplam_kare_hata)


plt.scatter(X[:, 0], X[:, 1], c=etiketler, cmap='viridis', s=50, alpha=0.7)
plt.scatter(merkezler[:, 0], merkezler[:, 1], c='red', marker='X', s=200, label='Merkezler')
plt.title('K-Ortalamalar Kümeleme')
plt.legend()
plt.show()
